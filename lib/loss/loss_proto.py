from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSAuxCELoss, FSCELossOriginal, FSAuxRMILoss, FSCELoss, FSCELoss_Release
from lib.loss.loss_utils import DiceCELoss, DiceLoss, FocalLoss, WeightFocalLoss, BoundaryLoss
from lib.utils.tools.logger import Logger as Log


class EssientailLoss(nn.Module):
    def __init__(self, configer=None):
        super(EssientailLoss, self).__init__()
        self.configer = configer

        self.loss_core = self.configer.get('loss', 'loss_core')

        Log.info(f"EssientailLoss using:{self.loss_core}")

        if self.loss_core == "ce":
            self.criterion = FSCELoss(configer=configer)
        elif self.loss_core == "dice":
            self.criterion = DiceLoss(configer=configer)
        elif self.loss_core == "focal":
            self.criterion == FocalLoss(configer=configer)
        elif self.loss_core == "weight_focal":
            self.criterion = WeightFocalLoss(configer=configer)
        elif self.loss_core == "dice_ce":
            self.criterion = DiceCELoss(configer=configer)

    def forward(self, inputs, target):
        return self.criterion(inputs, target)


class PPL_a(nn.Module, ABC):
    """Constrastive Loss (Inter)

    Args:
        nn (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, configer):
        super(PPL_a, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(
            contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class BPC(nn.Module, ABC):
    def __init__(self, configer):
        super(BPC, self).__init__()

        self.configer = configer
        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, boundary_logits, boundary_target):
        # 创建权重图
        weight_map = torch.ones_like(boundary_target, dtype=torch.float)
        weight_map[boundary_target != self.ignore_label] = 2.0  # 设置边界像素权重

        # 使用加权交叉熵损失函数
        loss_inter_boundary = F.cross_entropy(
            boundary_logits,
            boundary_target.long(),
            ignore_index=self.ignore_label
        )
        return loss_inter_boundary


class PPL_b(nn.Module, ABC):
    """Distance(Intra)

    Args:
        nn (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, configer):
        super(PPL_b, self).__init__()

        self.configer = configer

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class BPD(nn.Module, ABC):
    def __init__(self, configer):
        super(BPD, self).__init__()

        self.configer = configer
        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')[
                'ce_ignore_index']

    def forward(self, boundary_logits, boundary_target, debug=False):
        if debug:
            Log.info(f"type of boundary_target:{boundary_target.dtype}, type of ignore_label:{type(self.ignore_label)}")
            Log.info(f"boundary_logits shape:{boundary_logits.shape}")

        boundary_logits = boundary_logits[boundary_target != self.ignore_label, :]
        boundary_target = boundary_target[boundary_target != self.ignore_label]

        logits = torch.gather(boundary_logits, 1, boundary_target[:, None].long())
        loss_intra_boundary = (1 - logits).pow(2).mean()

        return loss_intra_boundary


class PBI(nn.Module, ABC):
    """Prototype Boundary Interactive Loss

    Args:
        nn (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(self, configer):
        super(PBI, self).__init__()

        self.configer = configer
        self.ignore_label = -1

    def forward(self, p_seg_out, boundary_p_seg_out, smooth=1e-6):
        """
        计算多类别分割结果 A 和 B 之间的反向 IoU 损失。

        Args:
            A: 整体分割结果，形状为 (batch_size, num_classes, height, width)。
            B: 边界关注分割结果，形状为 (batch_size, num_classes, height, width)。
            smooth: 平滑因子，避免除以零。

        Returns:
            反向 IoU 损失。
        """
        # [b,c,h,w]
        # Log.info(f"regular out size:{p_seg_out.size()}, type is:{type(p_seg_out)}")
        # Log.info(f"boudnary out size:{boundary_p_seg_out.size()}, type is:{type(boundary_p_seg_out)}")

        p_seg_out_c = p_seg_out.reshape(-1, 1, p_seg_out.shape[2], p_seg_out.shape[3])
        boundary_p_seg_out_c = boundary_p_seg_out.reshape(-1, 1, boundary_p_seg_out.shape[2], boundary_p_seg_out.shape[3])

        intersection = (p_seg_out_c * boundary_p_seg_out_c).sum(dim=(2, 3))
        union = (p_seg_out_c + boundary_p_seg_out_c).sum(dim=(2, 3)) - intersection

        # 计算 IoU
        iou = (intersection + smooth) / (union + smooth)

        # 计算反向 IoU 损失并求平均
        loss = 1 - iou
        return loss.mean()


class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)
            # 对应的权重设计：loss(seg loss) + loss_ppc + loss_ppd(prototype learning loss)
            # Log.info(f"seg_loss:{loss}, ppc:{loss_ppc},ppd:{loss_ppd}")
            return loss + self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELossPlus(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELossPlus, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        self.dice_weight = 0.5
        self.ce_weight = 0.5

        self.loss_scale = 10

        self.seg_criterion = EssientailLoss(configer=configer)

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            # proto_dist = preds['dist']

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # proto_reg_loss = torch.mean(torch.exp(-proto_dist))
            proto_reg_loss = 0

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)
            # Log.info(f"seg_loss:{loss}, ppc:{loss_ppc},ppd:{loss_ppd}")
            return loss + self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd + proto_reg_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELoss_Ultra_Dynamic_Test(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss_Ultra_Dynamic_Test, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        # * regular prototype loss weight
        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        self.diversity_weight = self.configer.get('protoseg', 'diversity_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)
            self.boundary_seg_criterion = BoundaryLoss()

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds
            assert "subdomain_probs" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            subdomain_probs = preds['subdomain_probs']

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # diversity loss
            # Encourage diverse subdomain assignments
            mean_probs = subdomain_probs.mean(dim=[0, 2, 3])
            diversity_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-5))

            prototype_loss = loss + self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd
            dynamic_subdomain_loss = self.diversity_weight * diversity_loss

            return prototype_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELoss_Ultra_Dynamic_Test(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss_Ultra_Dynamic_Test, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        # * regular prototype loss weight
        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        self.diversity_weight = self.configer.get('protoseg', 'diversity_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)
            self.boundary_seg_criterion = BoundaryLoss()

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds
            assert "subdomain_probs" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            subdomain_probs = preds['subdomain_probs']

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # diversity loss
            # Encourage diverse subdomain assignments
            mean_probs = subdomain_probs.mean(dim=[0, 2, 3])
            diversity_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-5))

            prototype_loss = loss + self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd
            dynamic_subdomain_loss = self.diversity_weight * diversity_loss

            return prototype_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELoss_Ultra_Dynamic_Test_SuperPixel(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss_Ultra_Dynamic_Test_SuperPixel, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.debug = self.configer.get('loss', 'loss_debug')

        # * regular prototype loss weight
        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        self.diversity_weight = self.configer.get('protoseg', 'diversity_weight')
        self.dice_weight = 0.7
        self.ce_weight = 0.3

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')
        self.ce_criterion = FSCELoss(configer=configer)

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            # self.seg_criterion = FSCELoss(configer=configer)
            self.seg_criterion = DiceLoss(configer=configer)

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds
            assert "subdomain_probs" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            subdomain_probs = preds['subdomain_probs']

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.dice_weight * self.seg_criterion(pred, target) + self.ce_weight * self.ce_criterion(pred, target)

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # diversity loss
            # Encourage diverse subdomain assignments
            # mean_probs = subdomain_probs.mean(dim=[0, 2, 3])
            # diversity_loss = -torch.sum(mean_probs * torch.log(mean_probs + 1e-5))

            prototype_loss = loss + self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd

            if self.debug:
                Log.info(f"seg_loss:{loss}, ppc:{loss_ppc},ppd:{loss_ppd}")

            return prototype_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELoss_Ultra_Final_1(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss_Ultra_Final_1, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.debug = self.configer.get('loss', 'loss_debug')

        # * regular prototype loss weight
        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        # * boundary loss weight
        self.loss_bpc_weight = self.configer.get('protoseg', 'loss_bpc_weight')
        self.loss_bpd_weight = self.configer.get('protoseg', 'loss_bpd_weight')
        self.loss_pbi_weight = self.configer.get('protoseg', 'loss_pbi_weight')
        self.loss_boundary_weight = self.configer.get('protoseg', 'loss_boundary_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        # loss core selection
        self.essientail_criterion = EssientailLoss(configer=configer)

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)
        self.boundary_seg_criterion = BoundaryLoss()
        self.bpc_criterion = BPC(configer=configer)
        self.bpd_criterion = BPD(configer=configer)
        self.pbi_criterion = PBI(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds
            assert "boundary_logits" in preds
            assert "boundary_target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            boundary_logits = preds['boundary_logits']
            boundary_target = preds['boundary_target']

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)

            # core loss
            loss = self.essientail_criterion(pred, target)

            # prototype loss
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            loss_bpc = self.bpc_criterion(boundary_logits, boundary_target)
            loss_bpd = self.bpd_criterion(boundary_logits, boundary_target)

            prototype_loss = self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd
            boundary_prototype_loss = self.loss_bpc_weight * loss_bpc + self.loss_bpd_weight * loss_bpd

            if self.debug:
                Log.info(f"seg_loss:{loss}, ppc:{loss_ppc},ppd:{loss_ppd}")

            return loss + prototype_loss + boundary_prototype_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELoss_Ultra_Final_2(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss_Ultra_Final_2, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.debug = self.configer.get('loss', 'loss_debug')

        # * regular prototype loss weight
        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        # * boundary loss weight
        self.loss_bpc_weight = self.configer.get('protoseg', 'loss_bpc_weight')
        self.loss_bpd_weight = self.configer.get('protoseg', 'loss_bpd_weight')
        self.loss_pbi_weight = self.configer.get('protoseg', 'loss_pbi_weight')
        self.loss_boundary_weight = self.configer.get('protoseg', 'loss_boundary_weight')

        self.dice_weight = 1
        self.ce_weight = 0

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        # loss core selection
        self.essientail_criterion = EssientailLoss(configer=configer)

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)
        self.boundary_seg_criterion = BoundaryLoss()
        self.bpc_criterion = BPC(configer=configer)
        self.bpd_criterion = BPD(configer=configer)
        self.pbi_criterion = PBI(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)

            # core loss
            loss = self.essientail_criterion(pred, target)

            # prototype loss
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            prototype_loss = self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd
            boundary_prototype_loss = 0

            if self.debug:
                Log.info(f"seg_loss:{loss}, ppc:{loss_ppc},ppd:{loss_ppd}")

            return loss + prototype_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss


class PixelPrototypeCELoss_Ultra_V3(nn.Module, ABC):
    def __init__(self, configer=None):
        super(PixelPrototypeCELoss_Ultra_V3, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.debug = self.configer.get('loss', 'loss_debug')

        # * regular prototype loss weight
        self.loss_ppl_a_weight = self.configer.get('protoseg', 'loss_ppl_a_weight')
        self.loss_ppl_b_weight = self.configer.get('protoseg', 'loss_ppl_b_weight')
        # * boundary loss weight
        self.loss_bpc_weight = self.configer.get('protoseg', 'loss_bpc_weight')
        self.loss_bpd_weight = self.configer.get('protoseg', 'loss_bpd_weight')
        self.loss_pbi_weight = self.configer.get('protoseg', 'loss_pbi_weight')
        self.loss_boundary_weight = self.configer.get('protoseg', 'loss_boundary_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        # loss core selection
        self.essientail_criterion = EssientailLoss(configer=configer)

        self.ppc_criterion = PPL_a(configer=configer)
        self.ppd_criterion = PPL_b(configer=configer)
        self.boundary_seg_criterion = FSCELoss(configer=configer)
        self.bpc_criterion = BPC(configer=configer)
        self.bpd_criterion = BPD(configer=configer)
        self.pbi_criterion = PBI(configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds
            assert "boundary_logits" in preds
            assert "boundary_target" in preds
            assert "boundary_seg" in preds
            assert "boundary_gt" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            boundary_logits = preds['boundary_logits']
            boundary_target = preds['boundary_target']
            boundary_seg = preds['boundary_seg']
            boundary_gt = preds['boundary_gt']

            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            boundary_seg_pred = F.interpolate(input=boundary_seg, size=(h, w), mode='bilinear', align_corners=True)
            boundary_seg_target = boundary_gt.squeeze(1)

            # core loss
            loss = self.essientail_criterion(pred, target)
            boundary_loss = self.boundary_seg_criterion(boundary_seg_pred, boundary_seg_target)

            # prototype loss
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            loss_bpc = self.bpc_criterion(boundary_logits, boundary_target)
            loss_bpd = self.bpd_criterion(boundary_logits, boundary_target)

            prototype_loss = self.loss_ppl_a_weight * loss_ppc + self.loss_ppl_b_weight * loss_ppd
            boundary_prototype_loss = self.loss_bpc_weight * loss_bpc + self.loss_bpd_weight * loss_bpd

            if self.debug:
                Log.info(f"seg_loss:{loss}, ppc:{loss_ppc},ppd:{loss_ppd}")

            return loss + prototype_loss + boundary_prototype_loss + boundary_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss
