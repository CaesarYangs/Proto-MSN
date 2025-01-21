import torch
from einops import rearrange
from torch import nn
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.tools.logger import Logger as Log
from lib.loss.loss_utils import DiceCELoss, DiceLoss, FocalLoss, WeightFocalLoss, BoundaryLoss


class MEGANetLoss(nn.Module):
    def __init__(self, configer):
        super(MEGANetLoss, self).__init__()
        self.configer = configer

        self.criterion = DiceCELoss(configer=self.configer)

    def forward(self, pred, gt):
        d0, d1, d2, d3, d4 = pred[0:]
        gt = gt.float()

        loss0 = self.criterion(torch.sigmoid(d0), gt)
        gt = F.avg_pool2d(gt.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        loss1 = self.criterion(torch.sigmoid(d1), gt)
        gt = F.avg_pool2d(gt.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        loss2 = self.criterion(torch.sigmoid(d2), gt)
        gt = F.avg_pool2d(gt.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        loss3 = self.criterion(torch.sigmoid(d3), gt)
        gt = F.avg_pool2d(gt.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        loss4 = self.criterion(torch.sigmoid(d4), gt)

        return loss0 + loss1 + loss2 + loss3 + loss4
