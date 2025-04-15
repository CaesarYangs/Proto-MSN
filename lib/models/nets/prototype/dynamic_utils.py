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


class SubdomainInteractionModule(nn.Module):
    def __init__(self, in_channels, num_subdomains, interaction_type='attention'):
        super(SubdomainInteractionModule, self).__init__()
        self.in_channels = in_channels
        self.num_subdomains = num_subdomains
        self.interaction_type = interaction_type

        if interaction_type == 'attention':
            self.attention = nn.MultiheadAttention(in_channels, num_heads=8)
        elif interaction_type == 'graph':
            self.graph_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
            self.adjacency = nn.Parameter(torch.ones(num_subdomains, num_subdomains))
        else:
            raise ValueError(f"Unsupported interaction type: {interaction_type}")

        self.norm = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        # x shape: (batch_size, num_subdomains, in_channels, height, width)
        b, s, c, h, w = x.shape
        x_flat = x.view(b, s, c, -1).permute(0, 3, 1, 2).reshape(-1, s, c)  # (b*h*w, s, c)

        if self.interaction_type == 'attention':
            attended, _ = self.attention(x_flat, x_flat, x_flat)
            x_interacted = attended + x_flat
        elif self.interaction_type == 'graph':
            adj = F.softmax(self.adjacency, dim=1)
            x_graph = torch.bmm(adj.expand(x_flat.size(0), -1, -1), x_flat)
            x_interacted = self.graph_conv(x_graph.transpose(1, 2)).transpose(1, 2) + x_flat

        x_normed = self.norm(x_interacted)
        x_out = self.ffn(x_normed) + x_normed

        return x_out.view(b, h, w, s, c).permute(0, 3, 4, 1, 2)  # (b, s, c, h, w)


class EnhancedSubdomainInteractionModule(nn.Module):
    def __init__(self, in_channels, num_subdomains, interaction_type='attention'):
        super(EnhancedSubdomainInteractionModule, self).__init__()
        self.in_channels = in_channels
        self.num_subdomains = num_subdomains
        self.interaction_type = interaction_type

        if interaction_type == 'attention':
            # self.attention = nn.MultiheadAttention(in_channels, num_heads=8)
            # self.cross_attention = nn.MultiheadAttention(in_channels, num_heads=8)
            pass
        elif interaction_type == 'graph':
            self.graph_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
            self.adjacency = nn.Parameter(torch.ones(num_subdomains, num_subdomains))
        else:
            raise ValueError(f"Unsupported interaction type: {interaction_type}")

        self.norm = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.gated_fusion = nn.Linear(in_channels * 2, in_channels)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x_flat = x.view(b, s, c, -1).permute(0, 3, 1, 2).reshape(-1, s, c)  # (b*h*w, s, c)

        if self.interaction_type == 'attention':
            # attended, _ = self.attention(x_flat, x_flat, x_flat)
            # cross_attended, _ = self.cross_attention(x_flat.mean(dim=1, keepdim=True), x_flat, x_flat)
            # x_interacted = self.gated_fusion(torch.cat([attended, cross_attended.expand_as(attended)], dim=-1))
            pass
        elif self.interaction_type == 'graph':
            adj = F.softmax(self.adjacency, dim=1)
            x_graph = torch.bmm(adj.expand(x_flat.size(0), -1, -1), x_flat)
            x_conv = self.graph_conv(x_graph.transpose(1, 2)).transpose(1, 2)
            # Instead of using GCN, we'll use a simple graph convolution
            x_gcn = torch.bmm(adj.expand(x_flat.size(0), -1, -1), x_flat)
            x_interacted = self.gated_fusion(torch.cat([x_conv, x_gcn], dim=-1))

        x_interacted = x_interacted + x_flat
        x_normed = self.norm(x_interacted)
        x_out = self.ffn(x_normed) + x_normed

        return x_out.view(b, h, w, s, c).permute(0, 3, 4, 1, 2)  # (b, s, c, h, w)


class DynamicSubdomainAllocationModule(nn.Module):
    def __init__(self, in_channels, num_subdomains, hidden_dim=256):
        super(DynamicSubdomainAllocationModule, self).__init__()
        self.in_channels = in_channels
        self.num_subdomains = num_subdomains
        self.hidden_dim = hidden_dim

        # Local feature extraction
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Global context
        self.attention = nn.MultiheadAttention(in_channels, num_heads=2)

        # Combine local and global features
        self.conv_combine = nn.Conv2d(hidden_dim + in_channels, hidden_dim, kernel_size=1)
        self.bn_combine = nn.BatchNorm2d(hidden_dim)

        # Final classification
        self.conv3 = nn.Conv2d(hidden_dim, num_subdomains, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.shape

        # Local feature extraction
        local_feat = F.relu(self.bn1(self.conv1(x)))
        local_feat = F.relu(self.bn2(self.conv2(local_feat)))

        # Global context
        global_feat = x.view(batch_size, self.in_channels, -1).permute(2, 0, 1)  # (h*w, batch_size, in_channels)
        global_feat, _ = self.attention(global_feat, global_feat, global_feat)
        global_feat = global_feat.permute(1, 2, 0).view(batch_size, self.in_channels, height, width)

        # Combine local and global information
        combined_feat = torch.cat([local_feat, global_feat], dim=1)
        combined_feat = F.relu(self.bn_combine(self.conv_combine(combined_feat)))

        # Final classification
        logits = self.conv3(combined_feat)

        # Soft assignment
        subdomain_probs = F.softmax(logits, dim=1)

        return subdomain_probs, logits


class AdaptiveSubdomainModule(nn.Module):
    def __init__(self, in_channels, max_subdomains):
        super(AdaptiveSubdomainModule, self).__init__()
        self.max_subdomains = max_subdomains

        # Complexity estimation network
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Subdomain classifier
        self.subdomain_classifier = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, max_subdomains, kernel_size=1)
        )

    def forward(self, x):
        # Estimate complexity
        complexity = self.complexity_estimator(x)

        # Determine number of subdomains based on complexity
        num_subdomains = torch.round(complexity * (self.max_subdomains - 1) + 1).long()

        # Get subdomain logits
        subdomain_logits = self.subdomain_classifier(x)

        # Apply adaptive masking
        mask = torch.arange(self.max_subdomains, device=x.device).unsqueeze(0) < num_subdomains
        masked_logits = subdomain_logits * mask.float().unsqueeze(-1).unsqueeze(-1)

        # Get subdomain probabilities
        subdomain_probs = F.softmax(masked_logits, dim=1)

        return subdomain_probs, num_subdomains


class SuperpixelSubdomainClassifier(nn.Module):
    def __init__(self, in_channels, num_subdomains, superpixel_params):
        super(SuperpixelSubdomainClassifier, self).__init__()
        self.num_subdomains = num_subdomains
        self.superpixel_params = superpixel_params

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_subdomains)
        )

    def generate_superpixels(self, x):
        # Detach x from the computation graph and move to CPU
        x_np = x.detach().cpu().numpy()

        # Generate superpixels for each image in the batch
        superpixels = []
        for i in range(x_np.shape[0]):
            img = x_np[i].transpose(1, 2, 0)
            segments = slic(img, **self.superpixel_params)
            superpixels.append(segments)

        # Convert back to tensor and move to the same device as x
        superpixels_tensor = torch.from_numpy(np.stack(superpixels)).to(x.device)
        return superpixels_tensor

    def extract_centroids(self, features, superpixels):
        centroids = []
        for i in range(superpixels.shape[0]):  # Iterate over batch
            props = regionprops(superpixels[i].cpu().numpy())
            batch_centroids = []
            for prop in props:
                y, x = prop.centroid
                centroid_features = features[i, :, int(y), int(x)]
                batch_centroids.append(centroid_features)
            centroids.append(torch.stack(batch_centroids))
        return torch.cat(centroids, dim=0)

    def assign_probabilities(self, logits, superpixels, shape):
        probs = F.softmax(logits, dim=1)
        subdomain_probs = torch.zeros((shape[0], self.num_subdomains, shape[2], shape[3])).to(logits.device)
        start_idx = 0
        for i in range(superpixels.shape[0]):  # Iterate over batch
            num_superpixels = len(torch.unique(superpixels[i]))
            end_idx = start_idx + num_superpixels
            for j in range(num_superpixels):
                mask = (superpixels[i] == j)
                subdomain_probs[i, :, mask] = probs[start_idx + j].unsqueeze(-1)
            start_idx = end_idx
        return subdomain_probs

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Generate superpixels
        superpixels = self.generate_superpixels(x)

        # Extract superpixel centroids
        centroids = self.extract_centroids(features, superpixels)

        # Classify centroids
        logits = self.classifier(centroids)

        # Assign subdomain probabilities to original image size
        subdomain_probs = self.assign_probabilities(logits, superpixels, x.shape)

        return subdomain_probs


class SuperpixelSubdomainClassifier(nn.Module):
    def __init__(self, in_channels, num_subdomains, grid_size=16):
        super(SuperpixelSubdomainClassifier, self).__init__()
        self.num_subdomains = num_subdomains
        self.grid_size = grid_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_subdomains)
        )

    def gpu_segmentation(self, x):
        B, C, H, W = x.shape
        device = x.device

        # Create grid
        grid_h, grid_w = H // self.grid_size, W // self.grid_size
        y_grid, x_grid = torch.meshgrid(torch.arange(grid_h, device=device),
                                        torch.arange(grid_w, device=device))
        grid = (y_grid * grid_w + x_grid).repeat_interleave(self.grid_size, dim=0).repeat_interleave(self.grid_size, dim=1)

        # Compute color similarity
        x_avg = F.avg_pool2d(x, self.grid_size, stride=self.grid_size)
        x_upsample = F.interpolate(x_avg, size=(H, W), mode='nearest')
        color_diff = torch.sum((x - x_upsample) ** 2, dim=1, keepdim=True)

        # Combine grid and color information
        segments = grid.unsqueeze(0).repeat(B, 1, 1)
        color_threshold = color_diff.mean() * 2  # Adjust this threshold as needed
        segments[color_diff.squeeze(1) > color_threshold] += grid_h * grid_w

        return segments

    def extract_centroids(self, features, segments):
        B, C, H, W = features.shape
        device = features.device

        # Create position tensor
        pos_y, pos_x = torch.meshgrid(torch.arange(H, device=device),
                                      torch.arange(W, device=device))
        pos = torch.stack((pos_y, pos_x), dim=0).float()

        centroids = []
        for i in range(B):
            seg = segments[i]
            num_segments = seg.max().item() + 1

            # Compute centroids
            seg_onehot = F.one_hot(seg.long(), num_segments).permute(2, 0, 1).float()
            seg_size = seg_onehot.sum(dim=(1, 2), keepdim=True)
            seg_centroid = (seg_onehot.unsqueeze(1) * pos.unsqueeze(0)).sum(dim=(2, 3)) / seg_size

            # Extract features at centroid locations
            centroid_y = seg_centroid[:, 0].long().clamp(0, H-1)
            centroid_x = seg_centroid[:, 1].long().clamp(0, W-1)
            batch_centroids = features[i, :, centroid_y, centroid_x].t()
            centroids.append(batch_centroids)

        return torch.cat(centroids, dim=0)

    def assign_probabilities(self, logits, segments, shape):
        probs = F.softmax(logits, dim=1)
        subdomain_probs = torch.zeros((shape[0], self.num_subdomains, shape[2], shape[3]), device=logits.device)

        batch_size = segments.shape[0]
        start_idx = 0
        for i in range(batch_size):
            num_segments = segments[i].max().item() + 1
            end_idx = start_idx + num_segments
            subdomain_probs[i] = probs[start_idx:end_idx, :, None, None] * F.one_hot(segments[i], num_segments).permute(2, 0, 1)
            start_idx = end_idx

        return subdomain_probs

    def forward(self, x):
        features = self.feature_extractor(x)
        segments = self.gpu_segmentation(x)
        centroids = self.extract_centroids(features, segments)
        logits = self.classifier(centroids)
        subdomain_probs = self.assign_probabilities(logits, segments, x.shape)
        return subdomain_probs


class GPUOptimizedSuperpixelSubdomainClassifier(nn.Module):
    def __init__(self, in_channels, num_subdomains, n_segments=10):
        super(GPUOptimizedSuperpixelSubdomainClassifier, self).__init__()
        self.num_subdomains = num_subdomains
        self.n_segments = n_segments
        Log.info(f"n_segments:{self.n_segments}")
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, num_subdomains, kernel_size=1),
            # nn.Tanh()  # 添加 tanh 激活函数
        )

        # Learnable centers for superpixel-like segmentation
        self.register_parameter('centers', nn.Parameter(torch.randn(n_segments, in_channels)))

    def generate_superpixels(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)

        # Normalize both input and centers
        x_norm = F.normalize(x_flat, dim=1, eps=1e-8)
        centers_norm = F.normalize(self.centers, dim=1, eps=1e-8)

        # Compute distances
        distances = torch.cdist(x_norm.permute(0, 2, 1), centers_norm)

        # Assign each pixel to the nearest center
        segments = distances.argmin(dim=2).view(B, H, W)

        return segments

    def pool_within_segments(self, logits, segments):
        B, C, H, W = logits.shape
        segments_flat = segments.view(B, -1)
        logits_flat = logits.view(B, C, -1)

        segment_sums = torch.zeros(B, C, self.n_segments, dtype=logits.dtype, device=logits.device)
        segment_counts = torch.zeros(B, 1, self.n_segments, dtype=logits.dtype, device=logits.device)

        segment_sums.scatter_add_(2, segments_flat.unsqueeze(1).expand(-1, C, -1), logits_flat)
        segment_counts.scatter_add_(2, segments_flat.unsqueeze(1), torch.ones_like(segments_flat, dtype=logits.dtype).unsqueeze(1))

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-5
        pooled_logits = segment_sums / (segment_counts + epsilon)

        return pooled_logits

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Generate superpixel-like segments
        segments = self.generate_superpixels(x)

        # Classify segments
        logits = self.classifier(features)

        # Pool logits within segments
        pooled_logits = self.pool_within_segments(logits, segments)

        # Reshape pooled_logits to 4D tensor
        B, C, _ = pooled_logits.shape
        pooled_logits_4d = pooled_logits.view(B, C, 1, -1)

        # Upsample pooled logits to original size
        subdomain_probs = F.interpolate(pooled_logits_4d, size=x.shape[2:], mode='nearest')

        return subdomain_probs


class GPUOptimizedSubdomainClassifier(nn.Module):
    def __init__(self, in_channels, num_subdomains, grid_size=16):
        super(GPUOptimizedSubdomainClassifier, self).__init__()
        self.num_subdomains = num_subdomains
        self.grid_size = grid_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, num_subdomains, kernel_size=1)
        )

    def generate_grid_segments(self, x):
        B, _, H, W = x.shape
        grid_h, grid_w = H // self.grid_size, W // self.grid_size
        segments = torch.arange(grid_h * grid_w, device=x.device).view(1, 1, grid_h, grid_w)
        segments = segments.repeat(B, 1, 1, 1)
        return F.interpolate(segments.float(), size=(H, W), mode='nearest').long()

    def pool_within_segments(self, logits, segments):
        B, C, H, W = logits.shape
        segments_flat = segments.view(B, -1)
        logits_flat = logits.view(B, C, -1)

        pooled_logits = []
        for b in range(B):
            segment_ids = segments_flat[b].unique()
            batch_pooled = torch.stack([
                logits_flat[b, :, segments_flat[b] == seg_id].mean(dim=1)
                for seg_id in segment_ids
            ])
            pooled_logits.append(batch_pooled)

        max_segments = max(len(p) for p in pooled_logits)
        padded_pooled_logits = torch.stack([
            F.pad(p, (0, 0, 0, max_segments - len(p))) for p in pooled_logits
        ])

        return padded_pooled_logits.permute(0, 2, 1).unsqueeze(-1)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)

        # Generate grid-based segments
        segments = self.generate_grid_segments(x)

        # Classify segments
        logits = self.classifier(features)

        # Pool logits within segments
        pooled_logits = self.pool_within_segments(logits, segments)

        # Upsample pooled logits to original size
        subdomain_probs = F.interpolate(pooled_logits, size=x.shape[2:], mode='nearest')

        return subdomain_probs
