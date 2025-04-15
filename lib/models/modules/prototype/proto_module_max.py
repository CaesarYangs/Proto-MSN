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
from lib.models.modules.hanet_attention import HANet_Conv
from lib.models.modules.contrast import momentum_update, momentum_update_m2, adaptive_momentum_update, l2_normalize, ProjectionHead
from lib.models.nets.prototype.dynamic_utils import EnhancedSubdomainInteractionModule, GPUOptimizedSubdomainClassifier, SuperpixelSubdomainClassifier, GPUOptimizedSuperpixelSubdomainClassifier
from lib.models.modules.sinkhorn import distributed_sinkhorn, kmeans_clustering
import lib.models.nets.prototype.boundary_utils as boundary_utils
import lib.models.nets.prototype.global_utilis as global_utils


class PrototypeModule_Max_V3(nn.Module):
    def __init__(self, in_channels, configer, type='V3', cluster='regular'):
        super(PrototypeModule_Max_V3, self).__init__()
        self.configer = configer
        self.gamma = self.configer.get('protoseg', 'gamma')
        # self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.update_prototype = self.configer.get('protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get('protoseg', 'pretrain_prototype')
        self.num_classes = self.configer.get('data', 'num_classes')
        self.num_boundary_prototype = self.configer.get('protoseg', 'num_boundary_prototype')
        self.boundary_line_threshold = self.configer.get('protoseg', 'boundary_line_threshold')
        self.cluster_method = cluster

        if self.configer.exists('protoseg', 'distance_measure'):
            self.distance_measure = self.configer.get('protoseg', 'distance_measure')
        else:
            self.distance_measure = 'cosine_similarity'

        Log.info(f"using distance meausrement:{self.distance_measure}")

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        #! boundary prototype learning section
        self.boundary_cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.boundary_prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_boundary_prototype, in_channels),
                                                requires_grad=True)

        if type == 'V3':
            self.boundary_attention = boundary_utils.BoundaryAttention(self.num_classes)
        elif type == 'V3s':
            Log.info(f"using V3s ImprovedBoundaryAttention")
            self.boundary_attention = boundary_utils.ImprovedBoundaryAttention(self.num_classes)

        trunc_normal_(self.boundary_prototypes, std=0.02)

        #! debug for sub-domain prototype
        self.num_subdomains = configer.get('protoseg', 'num_subdomains')
        self.num_prototypes_per_subdomain = configer.get('protoseg', 'num_prototypes_per_subdomain')

        Log.info(f"num_subdomains:{self.num_subdomains}, num_prototypes_per_subdomain:{self.num_prototypes_per_subdomain}")
        Log.info(f"boundary_prototypes:{self.num_boundary_prototype}")

        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_classes, self.num_prototypes_per_subdomain, in_channels))  # kmd
            for _ in range(self.num_subdomains)
        ])

        self.subdomain_prob_threshold = configer.get('protoseg', 'subdomain_prob_threshold')
        self.subdomain_classifier = GPUOptimizedSuperpixelSubdomainClassifier(
            in_channels=in_channels,
            num_subdomains=self.num_subdomains,
        )

        self.prototype_combine = nn.Sequential(
            nn.Conv2d(self.num_prototypes_per_subdomain * self.num_subdomains, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )

        self.dropout = nn.Dropout(0.5)

        for proto in self.prototypes:
            trunc_normal_(proto, std=0.02)

        Log.info(f"subdomain_prob_threshold:{self.subdomain_prob_threshold}")

        #! For Analysis
        self.features_list = []
        self.labels_list = []
        self.gt_list = []
        self.test_img_id = 0

    def extract_boundary(self, func, gt_seg, folder_name="b_gt_seg_viz_siim_test", debug=False):
        boundary_feature = func(gt_seg, line_threshold=self.boundary_line_threshold)  # DDD thought: dependency injection/functional programming

        #! for debug here:
        if debug:
            Log.info(f"size of gt_semantic_seg:{gt_seg.size()}")
            Log.info(f"size of boundary_feature:{boundary_feature.size()}")
            test_boundary_save_dir = "/DATA/yangyeqing/Project-Brown/methods/protoseg_dev/res/others/" + folder_name
            boundary_utils.boundary_gt_seg_vis(boundary_feature, test_boundary_save_dir)
            exit(0)

        return boundary_feature

    # ******************************************* MAIN PROTOTYPE SECTION *******************************************
    def prototype_learning(self, _c, out_seg, gt_seg, masks, subdomain_probs):
        b, _, h, w = out_seg.shape
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        # Initialize variables to store results for all subdomains
        all_proto_logits = []
        all_proto_targets = []

        _c = _c.view(b, h, w, -1)  # [b, h, w, 720]
        gt_seg = gt_seg.view(b, h, w)

        # Log.info(f"shape of gt_seg:{gt_seg.shape}, shape of _c:{_c.shape}")

        # Process each subdomain
        for i in range(self.num_subdomains):
            subdomain_mask = subdomain_probs[:, i] > self.subdomain_prob_threshold  # [b, h, w]

            # Filter features and labels for current subdomain
            # Log.info(f"shape of subdomain_probs[i]:{subdomain_probs[i].shape}")
            # Log.info(f"shape of _c:{_c.shape}, shape of subdomain_mask:{subdomain_mask.shape}")
            _c_subdomain = _c[subdomain_mask]
            gt_seg_subdomain = gt_seg[subdomain_mask]
            mask_subdomain = mask.view(b, h, w)[subdomain_mask]

            cosine_similarity = torch.mm(_c_subdomain, self.prototypes[i].view(-1, self.prototypes[i].shape[-1]).t())
            proto_logits = cosine_similarity

            proto_target = gt_seg_subdomain.clone().float()

            # clustering for each class within the subdomain
            protos = self.prototypes[i].data.clone()
            for k in range(self.num_classes):
                class_mask = gt_seg_subdomain == k
                if not class_mask.any():
                    # Log.info(f"skip here at {k}")
                    continue

                init_q = mask_subdomain[class_mask].unsqueeze(1).float()

                if self.cluster_method == 'kmeans':
                    q, indexs = kmeans_clustering(init_q)
                else:
                    q, indexs = distributed_sinkhorn(init_q)

                c_k = _c_subdomain[class_mask]
                m_k_tile = mask_subdomain[class_mask].unsqueeze(1).repeat(1, self.num_prototypes_per_subdomain)

                m_q = q * m_k_tile
                c_q = c_k.unsqueeze(1).repeat(1, self.num_prototypes_per_subdomain, 1)

                f = torch.sum(m_q.unsqueeze(-1) * c_q, dim=0)
                n = torch.sum(m_q, dim=0)

                if torch.sum(n) > 0 and self.update_prototype:
                    f = F.normalize(f, p=2, dim=-1)
                    new_value = momentum_update(
                        old_value=protos[k, n != 0, :],
                        new_value=f[n != 0, :],
                        momentum=self.gamma,
                        debug=False
                    )
                    protos[k, n != 0, :] = new_value

                proto_target[class_mask] = indexs.float() + (self.num_prototypes_per_subdomain * k)

            self.prototypes[i] = nn.Parameter(l2_normalize(protos), requires_grad=False)

            all_proto_logits.append(proto_logits)
            all_proto_targets.append(proto_target)

        # Combine results from all subdomains
        combined_proto_logits = torch.cat(all_proto_logits, dim=0)
        combined_proto_targets = torch.cat(all_proto_targets, dim=0)

        return combined_proto_logits, combined_proto_targets

    def boundary_prototype_learning(self, _c, boundary_out_seg, boundary_gt_seg, boundary_masks):
        """Boundary Prototype Learning Core Func

        """
        pred_seg = torch.max(boundary_out_seg, 1)[1]
        mask = (boundary_gt_seg == pred_seg.view(-1))

        boundary_cosine_similarity = torch.mm(
            _c, self.boundary_prototypes.view(-1, self.boundary_prototypes.shape[-1]).t())
        boundary_proto_logits = boundary_cosine_similarity
        boundary_proto_target = boundary_gt_seg.clone().float()

        boundary_protos = self.boundary_prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = boundary_masks[..., k]
            init_q = init_q[boundary_gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            if self.cluster_method == 'kmeans':
                q, indexs = kmeans_clustering(init_q)
            else:
                q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[boundary_gt_seg == k]

            c_k = _c[boundary_gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_boundary_prototype)

            m_q = q * m_k_tile

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile

            f = m_q.transpose(0, 1) @ c_q

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=boundary_protos[k, n != 0, :], new_value=f[n != 0, :], momentum=self.gamma, debug=False)
                boundary_protos[k, n != 0, :] = new_value

            boundary_proto_target[boundary_gt_seg == k] = indexs.float() + (self.num_boundary_prototype * k)

        self.boundary_prototypes = nn.Parameter(l2_normalize(boundary_protos), requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            boundary_protos = self.boundary_prototypes.data.clone()
            dist.all_reduce(boundary_protos.div_(dist.get_world_size()))
            self.boundary_prototypes = nn.Parameter(boundary_protos, requires_grad=False)

        return boundary_proto_logits, boundary_proto_target

    def forward(self, feats, gt_semantic_seg=None, pretrain_prototype=False, current_epoch=None, analysis_mode=False, ori_img=None):
        c = self.cls_head(feats)
        c = self.proj_head(c)

        #! debug and building now: sub-domain prototype learning
        # Subdomain classification: [12, 3, 128, 64]
        # subdomain_probs = self.subdomain_classifier(c)
        subdomain_probs = F.softmax(self.subdomain_classifier(c), dim=1)

        if torch.isnan(subdomain_probs).any() or torch.isinf(subdomain_probs).any():
            print("NaN or Inf detected in subdomain_probs")

        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        #! BPL part: boundary prototype learning section
        boundary_c = self.boundary_cls_head(feats)
        boundary_c = self.proj_head(boundary_c)
        _boundary_c = rearrange(boundary_c, ' b c h w -> (b h w) c')
        _boundary_c = self.feat_norm(_boundary_c)
        _boundary_c = l2_normalize(_boundary_c)
        self.boundary_prototypes.data.copy_(l2_normalize(self.boundary_prototypes))

        b_masks = torch.einsum('nd,kmd->nmk', _boundary_c, self.boundary_prototypes)

        b_out_seg = torch.amax(b_masks, dim=1)
        b_out_seg = self.mask_norm(b_out_seg)
        b_out_seg = rearrange(b_out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])

        #! DMCPL part
        # Compute similarities with prototypes from all subdomains
        all_masks = []
        for _, prototypes in enumerate(self.prototypes):
            prototypes.data.copy_(l2_normalize(prototypes))
            masks = torch.einsum('nd,kmd->nmk', _c, prototypes)
            all_masks.append(masks)

        # Combine masks based on subdomain probabilities
        subdomain_probs_reshaped = rearrange(subdomain_probs, 'b c h w -> c (b h w)')  # [2, 131072] 2 is sub-domain
        combined_masks = torch.stack(all_masks, dim=0)  # (num_subdomains, n(whole pixel number), num_prototypes, num_classes) [2, 131072, 10, 2]
        subdomain_probs_expanded = subdomain_probs_reshaped.unsqueeze(-1).unsqueeze(-1)   # add dim to the rear side [2, 131072,1,1]

        weighted_masks = combined_masks * subdomain_probs_expanded
        final_masks = weighted_masks.sum(dim=0)  # (n, num_prototypes, num_classes)

        out_seg = torch.amax(final_masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])

        # Apply boundary information as attention mechanism
        refined_out_seg = self.boundary_attention(out_seg, b_out_seg)

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)

            b_gt_seg_gen = self.extract_boundary(func=boundary_utils.extract_boundary_morphology_advanced, gt_seg=gt_semantic_seg,
                                                 folder_name="b_gt_seg_viz_cvc_test", debug=False)
            b_gt_seg_gen_1d = F.interpolate(b_gt_seg_gen.float(), size=feats.size()[2:], mode='nearest').view(-1)

            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks, subdomain_probs)
            boundary_contrast_logits, boundary_contrast_target = self.boundary_prototype_learning(_c, b_out_seg, b_gt_seg_gen_1d, b_masks)

            return {'seg': refined_out_seg, 'boundary_seg': b_out_seg, 'boundary_gt': b_gt_seg_gen, 'logits': contrast_logits, 'target': contrast_target, 'subdomain_probs': subdomain_probs, 'boundary_logits': boundary_contrast_logits, 'boundary_target': boundary_contrast_target}
