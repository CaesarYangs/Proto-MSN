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
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from torch_geometric.nn import GCNConv
from skimage.segmentation import slic
from skimage.measure import regionprops
from lib.utils.tools.logger import Logger as Log
from functools import lru_cache


class CombineGlobalLocal(nn.Module):
    def __init__(self, num_classes, feature_dim, combination_mode='concat'):
        super(CombineGlobalLocal, self).__init__()
        self.combination_mode = combination_mode

        if combination_mode == 'concat':
            self.combine = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU()
            )
        elif combination_mode == 'sum':
            self.global_weight = nn.Parameter(torch.FloatTensor([0.5]))
            self.local_weight = nn.Parameter(torch.FloatTensor([0.5]))
        elif combination_mode == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes, kernel_size=1),
                nn.Sigmoid()
            )
        elif combination_mode == 'gated':
            self.gate = nn.Sequential(
                nn.Conv2d(feature_dim, num_classes, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, local_features, global_features, original_features=None):
        if self.combination_mode == 'concat':
            combined = torch.cat([local_features, global_features], dim=1)
            return self.combine(combined)

        elif self.combination_mode == 'sum':
            return self.global_weight * global_features + self.local_weight * local_features

        elif self.combination_mode == 'attention':
            attention_weights = self.attention(torch.cat([local_features, global_features], dim=1))
            return attention_weights * global_features + (1 - attention_weights) * local_features

        elif self.combination_mode == 'gated':
            assert original_features is not None, "Original features are required for gated mode"
            gate = self.gate(original_features)
            return gate * global_features + (1 - gate) * local_features


class GlobalPrototype(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(GlobalPrototype, self).__init__()
        self.global_prototypes = nn.Parameter(torch.zeros(num_classes, feature_dim))
        nn.init.normal_(self.global_prototypes, std=0.02)

    def forward(self, features):
        # features: [B, C, H, W]
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # Compute similarities
        similarities = F.cosine_similarity(features_flat.unsqueeze(2),
                                           self.global_prototypes.unsqueeze(0).unsqueeze(0),
                                           dim=3)  # [B, HW, num_classes]

        return similarities.permute(0, 2, 1).view(B, -1, H, W)
