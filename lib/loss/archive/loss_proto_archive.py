from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSAuxCELoss, FSCELossOriginal, FSAuxRMILoss, FSCELoss, FSCELoss_Release
from lib.loss.loss_utils import DiceLoss, BoundaryLoss
from lib.utils.tools.logger import Logger as Log


class PixelPrototypeCELoss_MED(nn.Module, ABC):
    """ 自定义loss proto function

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(self, configer=None, experiment=None):
        super(PixelPrototypeCELoss_MED, self).__init__()

        self.configer = configer

        self.experiment = experiment

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = self.configer.get('protoseg', 'loss_ppc_weight')
        self.loss_ppd_weight = self.configer.get('protoseg', 'loss_ppd_weight')
        self.loss_inter_boundary_weight = self.configer.get(
            'protoseg', 'loss_inter_boundary_weight')
        self.loss_intra_boundary_weight = self.configer.get(
            'protoseg', 'loss_intra_boundary_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            # 一般情况下都是用基本的FSCE loss
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)
        self.inter_boundary_criterion = InterClassBoundaryLoss(
            configer=configer)
        self.intra_boundary_criterion = IntraClassBoundaryLoss(
            configer=configer)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            boundary_logits = preds['boundary_logits']
            boundary_target = preds['boundary_target']

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
            loss_inter_boundary = self.inter_boundary_criterion(
                boundary_logits, boundary_target)
            loss_intra_boundary = self.intra_boundary_criterion(
                boundary_logits, boundary_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

            if self.configer.get('wandb', 'use_wandb') and self.experiment is not None is True:
                try:
                    self.experiment.log({
                        "train_epoch": self.configer.get('epoch'),
                        "FSCELoss": loss,
                        "PPCLoss": self.loss_ppc_weight * loss_ppc,
                        "PPDLoss": self.loss_ppd_weight * loss_ppd,
                        "InterBoundaryLoss": self.loss_inter_boundary_weight * loss_inter_boundary,
                        "IntraBoundaryLoss": self.loss_intra_boundary_weight * loss_intra_boundary,
                        "TotalLoss": loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd + self.loss_inter_boundary_weight * loss_inter_boundary + self.loss_intra_boundary_weight * loss_intra_boundary
                    })
                except Exception as e:
                    Log.info(e)

            # print loss for test
            # Log.info(f"[[[[[[print loss]]]]]]: base FSCELoss:{loss}, PPCLoss:{self.loss_ppc_weight*loss_ppc}, PPDLoss:{self.loss_ppd_weight*loss_ppd}, InterBoundaryLoss:{self.loss_inter_boundary_weight * loss_inter_boundary}, IntraBoundaryLoss:{self.loss_intra_boundary_weight * loss_intra_boundary}, ALL:{loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd + self.loss_inter_boundary_weight * loss_inter_boundary + self.loss_intra_boundary_weight * loss_intra_boundary}")
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd + self.loss_inter_boundary_weight * loss_inter_boundary + self.loss_intra_boundary_weight * loss_intra_boundary

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        # Log.info(f"[[[[[[print loss]]]]]]: base FSCELoss:{loss}")
        return loss


class PixelPrototypeCELoss_Max(nn.Module, ABC):
    """ 自定义loss proto function

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(self, configer=None, experiment=None):
        super(PixelPrototypeCELoss_Max, self).__init__()

        self.configer = configer

        self.experiment = experiment

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = self.configer.get('protoseg', 'loss_ppc_weight')
        self.loss_ppd_weight = self.configer.get('protoseg', 'loss_ppd_weight')

        # * 边界损失
        self.loss_inter_boundary_weight = self.configer.get(
            'protoseg', 'loss_inter_boundary_weight')
        self.loss_intra_boundary_weight = self.configer.get(
            'protoseg', 'loss_intra_boundary_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            # 一般情况下都是用基本的FSCE loss
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)
        self.inter_boundary_criterion = InterClassBoundaryLoss(
            configer=configer)
        self.intra_boundary_criterion = IntraClassBoundaryLoss(
            configer=configer)

        # 添加Boundary Loss
        self.boundary_criterion = BoundaryLoss()
        self.loss_boundary_weight = self.configer.get('protoseg', 'loss_boundary_weight')

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        # Log.info(f"loss h:{h}, w:{w}")

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            boundary_logits = preds['boundary_logits']
            boundary_target = preds['boundary_target']

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
            loss_inter_boundary = self.inter_boundary_criterion(
                boundary_logits, boundary_target)
            loss_intra_boundary = self.intra_boundary_criterion(
                boundary_logits, boundary_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

            # 计算新的Boundary Loss
            loss_boundary = self.boundary_criterion(pred, target)

            total_loss = (loss +
                          self.loss_ppc_weight * loss_ppc +
                          self.loss_ppd_weight * loss_ppd +
                          self.loss_inter_boundary_weight * loss_inter_boundary +
                          self.loss_intra_boundary_weight * loss_intra_boundary +
                          self.loss_boundary_weight * loss_boundary)

            if self.configer.get('wandb', 'use_wandb') and self.experiment is not None:
                try:
                    self.experiment.log({
                        "train_epoch": self.configer.get('epoch'),
                        "FSCELoss": loss,
                        "PPCLoss": self.loss_ppc_weight * loss_ppc,
                        "PPDLoss": self.loss_ppd_weight * loss_ppd,
                        "InterBoundaryLoss": self.loss_inter_boundary_weight * loss_inter_boundary,
                        "IntraBoundaryLoss": self.loss_intra_boundary_weight * loss_intra_boundary,
                        "BoundaryLoss": self.loss_boundary_weight * loss_boundary,
                        "TotalLoss": total_loss
                    })
                except Exception as e:
                    Log.info(e)

            # Log.info(f"[[[[[[print loss]]]]]]: base FSCELoss:{loss}, PPCLoss:{self.loss_ppc_weight*loss_ppc}, PPDLoss:{self.loss_ppd_weight*loss_ppd}, InterBoundaryLoss:{self.loss_inter_boundary_weight * loss_inter_boundary}, IntraBoundaryLoss:{self.loss_intra_boundary_weight * loss_intra_boundary}, BoundaryLoss:{self.loss_boundary_weight * loss_boundary}, ALL:{total_loss}")
            return total_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        # Log.info(f"[[[[[[print loss]]]]]]: base FSCELoss:{loss}")
        return loss


class PixelPrototypeCELoss_Ultra(nn.Module, ABC):
    """ 自定义loss proto function

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(self, configer=None, experiment=None):
        super(PixelPrototypeCELoss_Ultra, self).__init__()

        self.configer = configer

        self.experiment = experiment

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')[
                'ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = self.configer.get('protoseg', 'loss_ppc_weight')
        self.loss_ppd_weight = self.configer.get('protoseg', 'loss_ppd_weight')

        # * 边界损失
        self.loss_inter_boundary_weight = self.configer.get(
            'protoseg', 'loss_inter_boundary_weight')
        self.loss_intra_boundary_weight = self.configer.get(
            'protoseg', 'loss_intra_boundary_weight')

        self.use_rmi = self.configer.get('protoseg', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            # 一般情况下都是用基本的FSCE loss
            self.seg_criterion = FSCELoss(configer=configer)

        self.ppc_criterion = PPC(configer=configer)
        self.ppd_criterion = PPD(configer=configer)
        self.inter_boundary_criterion = InterClassBoundaryLoss(
            configer=configer)
        self.intra_boundary_criterion = IntraClassBoundaryLoss(
            configer=configer)

        # 添加Boundary Loss
        self.boundary_criterion = BoundaryLoss()
        self.loss_boundary_weight = self.configer.get('protoseg', 'loss_boundary_weight')

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        # Log.info(f"loss h:{h}, w:{w}")

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            # boundary_logits = preds['boundary_logits']
            # boundary_target = preds['boundary_target']

            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)
            # loss_inter_boundary = self.inter_boundary_criterion(
            #     boundary_logits, boundary_target)
            # loss_intra_boundary = self.intra_boundary_criterion(
            #     boundary_logits, boundary_target)

            pred = F.interpolate(input=seg, size=(
                h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

            # 计算新的Boundary Loss
            loss_boundary = self.boundary_criterion(pred, target)

            total_loss = (loss +
                          self.loss_ppc_weight * loss_ppc +
                          self.loss_ppd_weight * loss_ppd +
                          self.loss_boundary_weight * loss_boundary)

            if self.configer.get('wandb', 'use_wandb') and self.experiment is not None:
                try:
                    self.experiment.log({
                        "train_epoch": self.configer.get('epoch'),
                        "FSCELoss": loss,
                        "PPCLoss": self.loss_ppc_weight * loss_ppc,
                        "PPDLoss": self.loss_ppd_weight * loss_ppd,
                        "BoundaryLoss": self.loss_boundary_weight * loss_boundary,
                        "TotalLoss": total_loss
                    })
                except Exception as e:
                    Log.info(e)

            # Log.info(f"[[[[[[print loss]]]]]]: base FSCELoss:{loss}, PPCLoss:{self.loss_ppc_weight*loss_ppc}, PPDLoss:{self.loss_ppd_weight*loss_ppd}, InterBoundaryLoss:{self.loss_inter_boundary_weight * loss_inter_boundary}, IntraBoundaryLoss:{self.loss_intra_boundary_weight * loss_intra_boundary}, BoundaryLoss:{self.loss_boundary_weight * loss_boundary}, ALL:{total_loss}")
            return total_loss

        seg = preds
        pred = F.interpolate(input=seg, size=(
            h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        # Log.info(f"[[[[[[print loss]]]]]]: base FSCELoss:{loss}")
        return loss
