import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import cv2
import kornia
import matplotlib.pyplot as plt
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.projection import ProjectionHead
from lib.utils.tools.logger import Logger as Log
from lib.models.modules.contrast import momentum_update, momentum_update_m2, adaptive_momentum_update, l2_normalize, ProjectionHead
from lib.models.modules.sinkhorn import distributed_sinkhorn


class HRNet_W48_Proto_Ultra(nn.Module):
    """
    Prototype Med Ultra Model

    prototype + 边界聚类
    """

    def __init__(self, configer, experiment=None):
        super(HRNet_W48_Proto_Ultra, self).__init__()
        self.configer = configer
        self.gamma = self.configer.get('protoseg', 'gamma')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        self.update_count = 0

        self.experiment = experiment

        # TODO dynamic prototype classes: 当前数值的含义是，对于背景聚5类，对于spine聚23类 还没有正式启用
        self.num_prototypes_per_class = [5, 23]

        in_channels = 720
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

        # 添加边界特征头
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels, self.num_classes,
                      kernel_size=1, stride=1, padding=0)
        )

        self.boundary_prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                                requires_grad=True)
        trunc_normal_(self.boundary_prototypes, std=0.02)

    def calculate_boundary_complexity(self, boundary_mask):
        # 实现边界复杂度计算
        # 这里使用一个简单的方法，可以根据需要进行优化
        return torch.sum(boundary_mask).item() / boundary_mask.numel()

    def boundary_aware_similarity(self, boundary_features, prototypes, boundary_mask):
        similarity = torch.mm(boundary_features, prototypes.t())
        boundary_weight = self.calculate_boundary_weight(boundary_mask)
        boundary_weight = boundary_weight.unsqueeze(1).expand_as(similarity)    # 将 boundary_weight 从 [131072] 扩展到 [131072, 20], 每个类有相同的类别权重
        # boundary_weight = boundary_weight.unsqueeze(1).repeat(1, 20)    # 每个类有不同的类别权重

        # Log.info(f"similarity shape:{similarity.shape}")
        # Log.info(f"boundary_weight shape:{boundary_weight.shape}")

        weighted_similarity = similarity * boundary_weight
        return weighted_similarity

    def calculate_boundary_weight(self, boundary_mask):
        # 实现边界权重计算
        # 这里使用一个简单的方法，可以根据需要进行优化
        return boundary_mask.float() + 1e-5

    def extract_boundary(self, gt_seg, mask):
        boundary_mask = torch.zeros_like(gt_seg)
        for k in range(self.num_classes):
            if torch.sum(gt_seg == k) > 0:
                boundary_mask[k] = torch.from_numpy(
                    cv2.Canny(gt_seg[k].cpu().numpy().astype(np.uint8), 0, 1)).to(gt_seg.device)
        return boundary_mask

    # def extract_boundary(self, gt_seg, mask):
    #     device = gt_seg.device
    #     # batch_size, num_classes, height, width = gt_seg.shape

    #     boundary_mask = torch.zeros_like(gt_seg)

    #     for k in range(self.num_classes):
    #         class_mask = (gt_seg == k).float()

    #         # 使用 Kornia 的 Sobel 滤波器
    #         grad_magnitude = kornia.filters.spatial_gradient(class_mask, mode='sobel', order=1)
    #         grad_magnitude = torch.sqrt(grad_magnitude[:, 0]**2 + grad_magnitude[:, 1]**2)

    #         # 使用自适应阈值
    #         threshold = grad_magnitude.mean() + grad_magnitude.std()

    #         # 创建边界mask
    #         boundary_mask[:, k] = (grad_magnitude > threshold).float().squeeze(1)

    #         # 使用 Kornia 的形态学操作
    #         kernel = torch.ones(3, 3, device=device)
    #         boundary_mask[:, k] = kornia.morphology.dilation(boundary_mask[:, k].unsqueeze(1), kernel).squeeze(1)
    #         boundary_mask[:, k] = kornia.morphology.erosion(boundary_mask[:, k].unsqueeze(1), kernel).squeeze(1)

    #     # 使用 Kornia 的高斯模糊来平滑边界
    #     boundary_mask = kornia.filters.gaussian_blur2d(boundary_mask, (5, 5), (1.5, 1.5))

    #     return boundary_mask

    # def combined_edge_detection(self, image):
    #     # 确保输入是 4D 张量 [B, C, H, W]
    #     if image.dim() == 1:
    #         # 假设原始图像是方形的
    #         side = int(torch.sqrt(image.size(0)))
    #         image = image.view(1, 1, side, side)
    #     elif image.dim() == 2:
    #         image = image.unsqueeze(0).unsqueeze(0)
    #     elif image.dim() == 3:
    #         image = image.unsqueeze(0)

    #     # Sobel
    #     gradients = kornia.filters.spatial_gradient(image, mode='sobel', order=1)
    #     sobel_magnitude = torch.sqrt(gradients[:, :, 0] ** 2 + gradients[:, :, 1] ** 2)

    #     # Laplacian
    #     laplacian = torch.abs(kornia.filters.laplacian(image, kernel_size=3))

    #     # 结合 Sobel 和 Laplacian
    #     combined = (sobel_magnitude + laplacian) / 2
    #     return combined.squeeze()  # 返回 [H, W] 形状

    # seems not working
    # def extract_boundary(self, gt_seg, mask):
    #     device = gt_seg.device

    #     input_size = self.configer.get('train','data_transformer')['input_size']
    #     w = input_size[0]
    #     h = input_size[1]

    #     # 获取原始图像尺寸
    #     gt_seg = gt_seg.view(h, w)

    #     boundary_mask = torch.zeros_like(gt_seg)

    #     for k in range(self.num_classes):
    #         if torch.sum(gt_seg == k) > 0:
    #             class_mask = (gt_seg == k).float()
    #             edges = self.combined_edge_detection(class_mask)
    #             # 可以根据需要应用阈值
    #             edges = (edges > edges.mean()).float()
    #             boundary_mask[gt_seg == k] = edges[gt_seg == k]

    #     return boundary_mask.view(-1)  # 返回展平的张量

        # ******************************************* MAIN PROTOTYPE SECTION *******************************************

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        #! Traditional Prototype Learning section
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(
            _c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + \
                (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

         #! Boundary Prototype Learning section
        boundary_mask = self.extract_boundary(gt_seg, mask)
        boundary_cosine_similarity = self.boundary_aware_similarity(_c,
                                                                    self.boundary_prototypes.view(-1, self.boundary_prototypes.shape[-1]),
                                                                    boundary_mask)
        boundary_proto_logits = boundary_cosine_similarity
        boundary_proto_target = boundary_mask.clone().float()

        boundary_protos = self.boundary_prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[boundary_mask == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = boundary_mask[boundary_mask == k]

            c_k = _c[boundary_mask == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = adaptive_momentum_update(
                    old_value=boundary_protos[k, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.gamma,
                    update_count=self.update_count
                )
                self.update_count += 1

                boundary_protos[k, n != 0, :] = new_value

            boundary_proto_target[boundary_mask ==
                                  k] = indexs.float() + (self.num_prototype * k)

        self.boundary_prototypes = nn.Parameter(l2_normalize(boundary_protos),
                                                requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            boundary_protos = self.boundary_prototypes.data.clone()
            dist.all_reduce(boundary_protos.div_(dist.get_world_size()))
            self.boundary_prototypes = nn.Parameter(
                boundary_protos, requires_grad=False)

        return proto_logits, proto_target, boundary_proto_logits, boundary_proto_target

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):

        #! HRNet backbone section
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')   # TODO b wrong
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        #! Prototype Learning section
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        # self.boundary_prototypes.data.copy_(l2_normalize(self.boundary_prototypes))

        # n: h*w, k: num_class, m: num_prototype
        # 得到相似度矩阵或者是关联度量,其中包含了每个样本与每个类别中每个原型之间的相似度或匹配度
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w",
                            b=feats.shape[0], h=feats.shape[2])  # TODO 计算效率低

        # 边界特征提取-利用最后一个输出头操作
        boundary_feats = self.boundary_head(feats)
        boundary_masks = torch.einsum(
            'bkhw,kmd->bmhw', boundary_feats, self.boundary_prototypes)
        boundary_seg = torch.amax(boundary_masks, dim=1)

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[
                                   2:], mode='nearest').view(-1)
            contrast_logits, contrast_target, boundary_contrast_logits, boundary_contrast_target = self.prototype_learning(
                _c, out_seg, gt_seg, masks)

            return {'seg': out_seg, 'boundary_seg': boundary_seg, 'logits': contrast_logits, 'target': contrast_target,
                    'boundary_logits': boundary_contrast_logits, 'boundary_target': boundary_contrast_target}

        return out_seg
