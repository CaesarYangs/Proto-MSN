# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
# Microsoft Research
# yuyua@microsoft.com
# Copyright (c) 2019
##
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from lib.loss.loss_helper import FSAuxOhemCELoss, FSOhemCELoss, FSRMILoss
from lib.loss.loss_helper import FSCELoss, FSCELossOriginal, FSCELoss_Release, FSCELoss_BINARY, FSAuxCELoss, FSAuxRMILoss, FSCELOVASZLoss, MSFSAuxRMILoss, FSAuxCELossDSN
from lib.loss.loss_helper import SegFixLoss
from lib.loss.rmi_loss import RMILoss
from lib.loss.loss_contrast import ContrastAuxCELoss, ContrastCELoss
from lib.loss.loss_contrast_mem import ContrastCELoss as MemContrastCELoss
from lib.loss.loss_proto import PixelPrototypeCELossOriginal, PixelPrototypeCELoss, PixelPrototypeCELossPlus, PixelPrototypeCELoss_Ultra_Test, PixelPrototypeCELoss_Ultra_Dynamic_Test, PixelPrototypeCELoss_Ultra_Dynamic_Test_SuperPixel, PixelPrototypeCELoss_Ultra_Final_1, PixelPrototypeCELoss_Ultra_Final_2, PixelPrototypeCELoss_Ultra_V3, PixelPrototypeCELoss_Pro_wo_BPL, PixelPrototypeCELoss_Pro_wo_DMCPL
from lib.loss.loss_utils import DiceLoss, DiceCELoss

# connect_loss
from lib.loss.others.DTMFormer_loss import DTMFormerLoss

# MEGANet_loss
from lib.loss.others.MEGANet_loss import MEGANetLoss

# PLHN_Loss
from lib.loss.others.PLHN_loss import PLHN_Loss

# Pionono Loss
from lib.loss.others.pionono_loss import PiononoLoss

from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import is_distributed


SEG_LOSS_DICT = {
    'fs_ce_loss': FSCELoss,
    'fs_ce_loss_original': FSCELossOriginal,
    'fs_ce_loss_release': FSCELoss_Release,
    'fs_ce_loss_binary': FSCELoss_BINARY,
    'fs_ohemce_loss': FSOhemCELoss,
    'fs_auxce_loss': FSAuxCELoss,
    'fs_aux_rmi_loss': FSAuxRMILoss,
    'fs_auxohemce_loss': FSAuxOhemCELoss,
    'segfix_loss': SegFixLoss,
    'rmi_loss': RMILoss,
    'fs_rmi_loss': FSRMILoss,
    'contrast_auxce_loss': ContrastAuxCELoss,
    'contrast_ce_loss': ContrastCELoss,
    'fs_ce_lovasz_loss': FSCELOVASZLoss,
    'ms_fs_aux_rmi_loss': MSFSAuxRMILoss,
    'fs_auxce_dsn_loss': FSAuxCELossDSN,
    'mem_contrast_ce_loss': MemContrastCELoss,
    # others
    'DTMFormerLoss': DTMFormerLoss,
    'MEGANetLoss': MEGANetLoss,
    'PLHN_Loss': PLHN_Loss,
    'PiononoLoss': PiononoLoss,
    # prototype
    'dice_loss': DiceLoss,
    'dice_ce_loss': DiceCELoss,
    'PixelPrototypeCELoss_Ultra_V3': PixelPrototypeCELoss_Ultra_V3,
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if is_distributed():
            Log.info('use distributed loss')
            return loss

        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            Log.info('use DataParallelCriterion loss')
            from lib.extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_seg_loss(self, loss_type=None, experiment=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        Log.info('use loss is :{}.'.format(key))
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)
        Log.info('use loss: {}.'.format(key))

        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)
