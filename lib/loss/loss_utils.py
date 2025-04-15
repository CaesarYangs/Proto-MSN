
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.ndimage import distance_transform_edt

from lib.utils.tools.logger import Logger as Log
from lib.loss.rmi_loss import RMILoss
from lib.loss.loss_helper import FSAuxCELoss, FSCELossOriginal, FSAuxRMILoss, FSCELoss, FSCELoss_Release
from lib.models.nets.prototype.boundary_utils import extract_boundary_morphology_advanced


class DiceCELoss(nn.Module):
    def __init__(self, configer=None):
        super(DiceCELoss, self).__init__()
        self.configer = configer

        self.dice_weight = self.configer.get('network', 'loss_weights')['dice_loss']
        self.ce_weight = self.configer.get('network', 'loss_weights')['ce_loss']

        Log.info(f"DiceCELoss: dice_weight:{self.dice_weight},ce_weight:{self.ce_weight}")

        self.dice_loss = DiceLoss(configer=configer)
        self.ce_loss = FSCELoss(configer=configer)

        Log.info(f"using DICE_CE as essientail loss")

    def forward(self, inputs, target):
        dice_loss = self.dice_loss(inputs, target)
        ce_loss = self.ce_loss(inputs, target)

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class FocalLoss(nn.Module):
    """Focal Loss for binary and multi-class segmentation"""

    def __init__(self, configer=None):
        super(FocalLoss, self).__init__()
        self.configer = configer

        self.alpha = 0.8
        self.gamma = 3.0
        self.epsilon = 1e-6
        self.weight = None
        self.multiclass = False

        self.foreground_weight = 3.0  # 可调整
        self.background_weight = 1.0  # 可调整

        if self.configer.exists('loss', 'params'):
            params = self.configer.get('loss', 'params')
            if 'focal_alpha' in params:
                self.alpha = params['focal_alpha']
            if 'focal_gamma' in params:
                self.gamma = params['focal_gamma']
            if 'focal_weight' in params:
                self.weight = params['focal_weight']

        Log.info(f"using FOCAL as essientail loss")

    def forward(self, inputs, target):
        # 选择第一个通道的预测结果
        inputs = inputs[:, 0, :, :]

        # 将 target 转换为二进制值
        target = (target > 0).float()

        # 应用 sigmoid
        pred = torch.sigmoid(inputs)

        # 计算 focal loss
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)

        loss = -focal_weight * torch.log(pt + self.epsilon)

        # 如果提供了权重，则应用权重
        if self.weight is not None:
            loss = loss * self.weight

        return loss.mean()


class WeightFocalLoss(nn.Module):
    def __init__(self, configer):
        super(WeightFocalLoss, self).__init__()
        self.configer = configer
        self.epsilon = 1e-6
        self.alpha = 0.8
        self.gamma = 3.0
        self.weight = 10
        self.foreground_weight = 3.0  # 默认前景权重
        self.background_weight = 1.0  # 默认背景权重

        if self.configer.exists('loss', 'params'):
            params = self.configer.get('loss', 'params')
            if 'focal_alpha' in params:
                self.alpha = params['focal_alpha']
            if 'focal_gamma' in params:
                self.gamma = params['focal_gamma']
            if 'focal_weight' in params:
                self.weight = params['focal_weight']
            if 'foreground_weight' in params:
                self.foreground_weight = params['foreground_weight']
            if 'background_weight' in params:
                self.background_weight = params['background_weight']

        Log.info(f"using WEIGHT FOCAL as essientail loss, alpha:{self.alpha}, gamma:{self.gamma}")
        Log.info(f"weight:{self.weight},for_weight:{self.foreground_weight}, back_weight:{self.background_weight}")

    def forward(self, inputs, target):
        # 选择第一个通道的预测结果
        inputs = inputs[:, 0, :, :]

        # 将 target 转换为二进制值
        target = (target > 0).float()

        # 应用 sigmoid
        pred = torch.sigmoid(inputs)

        # 计算 focal loss
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)

        loss = -focal_weight * torch.log(pt + self.epsilon)

        # 应用前景和背景权重
        weight_map = torch.where(target == 1, self.foreground_weight, self.background_weight)
        loss = loss * weight_map

        # 如果提供了额外的权重，则应用它
        if self.weight is not None:
            loss = loss * self.weight

        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for binary and multi-class segmentation"""

    def __init__(self, configer=None):
        super(DiceLoss, self).__init__()
        self.configer = configer

        self.smooth = 1e-6
        self.weight = None
        self.multiclass = False

        if self.configer.exists('loss', 'params') and 'dice_smooth' in self.configer.get('loss', 'params'):
            self.smooth = self.configer.get('loss', 'params')['dice_smooth']
            self.weight = self.configer.get('loss', 'params')['dice_weight']

        Log.info(f"using DICE as essientail loss")

    def forward(self, inputs, target):
        # if self.multiclass:
        #     # 如果是多分类任务，将 target 转换为 one-hot 编码
        #     target = torch.nn.functional.one_hot(target.long(), num_classes=pred.shape[1])
        #     target = target.permute(0, 3, 1, 2)  # 将维度调整为 NCHW
        # else:
        #     # 对于二分类，我们只需要正类的预测概率
        #     pred = torch.softmax(pred, dim=1)[:, 1]  # 现在 pred 形状为 [16, 512, 256]

        # 选择第一个通道的预测结果
        inputs = inputs[:, 0, :, :]

        # 将 target 转换为二进制值
        target = (target > 0).float()

        # 应用 sigmoid 而不是 softmax，因为我们只有一个通道
        pred = torch.sigmoid(inputs)

        # 计算 Dice coefficient
        intersection = (pred * target).sum(dim=(1, 2))
        pred_sum = pred.sum(dim=(1, 2))
        target_sum = target.sum(dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # 计算 Dice Loss
        dice_loss = 1 - dice.mean()

        return dice_loss


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    def forward(self, pred, target):
        # pred shape: [B, 1, H, W], target shape: [B, H, W]
        boundary_target = self.create_boundary_target(target)
        boundary_pred = self.create_boundary_pred(pred)
        return self.bce_loss(boundary_pred, boundary_target)

    def create_boundary_target(self, target):
        prepared_target = target[:, None, ...]  # [B, 1, H, W]
        return extract_boundary_morphology_advanced(prepared_target)

    def create_boundary_pred(self, pred):
        class_index = torch.argmax(pred, dim=1)
        extracted_result = class_index.unsqueeze(1)  # [B, 1, H, W]
        return extract_boundary_morphology_advanced(extracted_result)


class BoundaryLoss2(nn.Module):
    def __init__(self):
        super(BoundaryLoss2, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=None, reduction='mean')

    def forward(self, boundary_pred, target):
        # pred shape: [B, 1, H, W], target shape: [B, H, W]
        boundary_target = self.create_boundary_target(target)
        return self.bce_loss(boundary_pred, boundary_target)

    def create_boundary_target(self, target):
        prepared_target = target[:, None, ...]  # [B, 1, H, W]
        return extract_boundary_morphology_advanced(prepared_target)

    def create_boundary_pred(self, pred):
        class_index = torch.argmax(pred, dim=1)
        extracted_result = class_index.unsqueeze(1)  # [B, 1, H, W]
        return extract_boundary_morphology_advanced(extracted_result)


def extract_boundary_morphology_advanced(gt_seg, num_classes=2, line_threshold=300):
    boundary_mask = torch.zeros_like(gt_seg, dtype=torch.float)

    for k in range(num_classes):
        mask = (gt_seg == k).float()
        if torch.sum(mask) > 0:
            dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
            eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
            boundary = dilated - eroded

            # 对每个批次单独处理
            for i in range(boundary.shape[0]):
                boundary[i] = remove_long_lines(boundary[i], line_threshold)

            boundary_mask += boundary

    return boundary_mask


def remove_long_lines(boundary, threshold):
    """
    移除完全水平或垂直的长直线
    """
    # 转换为NumPy数组
    boundary_np = boundary.cpu().numpy()

    # 检测并移除水平线
    horizontal_sum = np.sum(boundary_np, axis=1)
    horizontal_lines = np.where(horizontal_sum >= threshold)[0]
    for line in horizontal_lines:
        boundary_np[line, :] = 0

    # 检测并移除垂直线
    vertical_sum = np.sum(boundary_np, axis=0)
    vertical_lines = np.where(vertical_sum >= threshold)[0]
    for line in vertical_lines:
        boundary_np[:, line] = 0

    # 转回PyTorch张量
    return torch.from_numpy(boundary_np).to(boundary.device)

# class BoundaryLoss(nn.Module):
#     def __init__(self):
#         super(BoundaryLoss, self).__init__()

#     def forward(self, pred, target):
#         # pred shape: [B, C, H, W]
#         # target shape: [B, H, W]
#         pred = F.softmax(pred, dim=1)
#         boundary_target = self.create_boundary_target(target)
#         boundary_pred = self.create_boundary_pred(pred)
#         return F.binary_cross_entropy(boundary_pred, boundary_target)

#     def create_boundary_target(self, target):
#         # target shape: [B, H, W]
#         target_dt = torch.zeros_like(target, dtype=torch.float32)
#         for i in range(target.shape[0]):  # iterate over batch
#             dt = torch.from_numpy(distance_transform_edt(target[i].cpu().numpy())).to(target.device)
#             target_dt[i] = dt
#         return torch.exp(-target_dt).unsqueeze(1)  # [B, 1, H, W]

#     def create_boundary_pred(self, pred):
#         # pred shape: [B, C, H, W]
#         return 1 - pred[:, 0].unsqueeze(1)  # Use background class probability


class BoundaryConsistencyLoss(nn.Module):
    def __init__(self):
        super(BoundaryConsistencyLoss, self).__init__()

    def hausdorff_loss(seg, gt):
        pred_boundary = extract_boundary_morphology_advanced(seg)
        gt_boundary = extract_boundary_morphology_advanced(gt)

        d1 = torch.max(torch.min(torch.cdist(pred_boundary, gt_boundary), dim=1)[0])
        d2 = torch.max(torch.min(torch.cdist(gt_boundary, pred_boundary), dim=1)[0])

        return torch.max(d1, d2)

    def forward(seg, gt):
        pred_boundary = extract_boundary_morphology_advanced(seg)
        gt_boundary = extract_boundary_morphology_advanced(gt)

        consistency_loss = F.mse_loss(pred_boundary, gt_boundary)

        return consistency_loss


class ProtoytpeLogitsEntropyLoss(nn.Module):
    def __init__(self):
        super(ProtoytpeLogitsEntropyLoss, self).__init__()

    def forward(self, contrast_logits):
        """
        计算 Entropy Minimization Loss.

        Args:
            contrast_logits: 形状为 (N, K) 的张量，表示每个像素属于每个 prototype 的未归一化得分。
                            N 是像素数量，K 是 prototype 数量。

        Returns:
            标量，表示 entropy loss.
        """

        probs = F.softmax(contrast_logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        return entropy.mean()


def focal_loss(contrast_logits, contrast_target, gamma=2.0, alpha=0.25):
    """
    Focal Loss: 对难分类的样本给予更高的权重
    """
    ce_loss = F.cross_entropy(contrast_logits, contrast_target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt)**gamma * ce_loss
    return focal_loss.mean()


def kl_divergence_loss(contrast_logits, contrast_target):
    """
    KL 散度损失：衡量 contrast_logits 和 contrast_target 之间的分布差异
    """
    log_probs = F.log_softmax(contrast_logits, dim=1)
    return F.kl_div(log_probs, contrast_target, reduction='batchmean')


def top_k_loss(contrast_logits, contrast_target, k=3):
    """
    Top-k Loss：只考虑 top-k 个最相似的原型
    """
    top_k_logits, _ = torch.topk(contrast_logits, k, dim=1)
    top_k_target, _ = torch.topk(contrast_target, k, dim=1)
    return F.mse_loss(top_k_logits, top_k_target)


def margin_ranking_loss(contrast_logits, contrast_target, margin=1.0):
    """
    Margin Ranking Loss：鼓励正确原型的得分高于错误原型
    """
    _, target_indices = torch.max(contrast_target, dim=1)
    positive_logits = contrast_logits[torch.arange(contrast_logits.size(0)), target_indices]
    negative_logits = contrast_logits[torch.arange(contrast_logits.size(0)), 1 - target_indices]
    return F.margin_ranking_loss(positive_logits, negative_logits,
                                 torch.ones_like(positive_logits), margin=margin)


def label_smoothing_loss(contrast_logits, contrast_target, smoothing=0.1):
    """
    Label Smoothing Loss：通过软化 contrast_target 来提高模型的鲁棒性
    """
    confidence = 1.0 - smoothing
    smoothed_target = torch.full_like(contrast_logits, smoothing / (contrast_logits.size(-1) - 1))
    smoothed_target.scatter_(1, contrast_target.unsqueeze(1), confidence)
    return torch.mean(torch.sum(-smoothed_target * F.log_softmax(contrast_logits, dim=1), dim=1))
