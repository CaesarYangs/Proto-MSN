import torch
from einops import rearrange
from torch import nn
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
from lib.utils.tools.logger import Logger as Log
from lib.loss.loss_utils import DiceCELoss, DiceLoss, FocalLoss, WeightFocalLoss, BoundaryLoss


class ClusterSegLoss(nn.Module):
    def __init__(self, configer):
        super(ClusterSegLoss, self).__init__()
        self.configer = configer

        mask_loss, boundary_loss, cluster_loss = LovaszSoftmax(), LovaszSoftmax(), LovaszSoftmax()

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


class LovaszSoftmax(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
