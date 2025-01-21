import os
import torch
import numpy as np
import cv2
import cupy as cp
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.tools.logger import Logger as Log


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BoundaryAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x, boundary):
        attention = self.conv(boundary)
        attention = self.norm(attention)
        attention = torch.sigmoid(attention)
        return x * attention + x * (1 - attention)


class ImprovedBoundaryAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels

        # Ensure the reduction doesn't result in 0 channels
        self.reduced_channels = max(1, in_channels // reduction_ratio)

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, self.reduced_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.reduced_channels, in_channels, 1)

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Boundary attention
        self.conv_boundary = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm_boundary = nn.BatchNorm2d(in_channels)

        # Gating mechanism
        self.conv_gate = nn.Conv2d(in_channels * 2, 2, kernel_size=1)

    def forward(self, x, boundary):
        # Channel attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = torch.sigmoid(avg_out + max_out)
        x_channel = x * channel_out

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.cat([avg_out, max_out], dim=1)
        spatial_out = torch.sigmoid(self.conv_spatial(spatial_out))
        x_spatial = x * spatial_out

        # Boundary attention
        boundary_out = self.conv_boundary(boundary)
        boundary_out = self.norm_boundary(boundary_out)
        boundary_out = torch.sigmoid(boundary_out)
        x_boundary = x * boundary_out

        # Gating mechanism
        gate_input = torch.cat([x_channel + x_spatial, x_boundary], dim=1)
        gate = torch.sigmoid(self.conv_gate(gate_input))

        out = gate[:, 0:1, :, :] * (x_channel + x_spatial) + gate[:, 1:2, :, :] * x_boundary

        # Residual connection
        return out + x


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seg, boundary_seg):
        combined = torch.cat([seg, boundary_seg], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return seg * attention + boundary_seg * (1 - attention)


class AttentionFusionBalanced(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的平衡参数

    def forward(self, seg, boundary_seg):
        combined = torch.cat([seg, boundary_seg], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return seg * (self.alpha * attention + (1 - self.alpha)) + boundary_seg * ((1 - self.alpha) * attention + self.alpha)


class BoundaryRefinement(nn.Module):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def forward(self, x, boundary):
        boundary_mask = (boundary > self.threshold).float()
        refined_seg = x * (1 - boundary_mask) + boundary * boundary_mask
        return refined_seg


def sharpen_image(image, type=1):
    """
    对图像应用锐化滤波器。
    """
    if type == 1:
        kernel = np.array([[0, -1,  0],
                           [-1,  5, -1],
                           [0, -1,  0]])

    if type == 2:
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


def boundary_gt_seg_vis(boudnary_gt_seg, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Log.info(f"boundary_gt_seg shape:{boudnary_gt_seg.shape}")

    B, N, h, w = boudnary_gt_seg.shape

    for batch_idx in range(B):
        for class_idx in range(N):
            boundary_map = boudnary_gt_seg[batch_idx, class_idx, :, :]
            boundary_map_np = boundary_map.cpu().numpy()
            boundary_mask = (boundary_map_np > 0).astype("uint8") * 255

            Log.info(f"saving seg viz: batch_{batch_idx}_class_{class_idx}.png")

            save_path = os.path.join(
                save_dir, f"batch_{batch_idx}_class_{class_idx}.png")
            cv2.imwrite(save_path, boundary_mask)


def extract_boundary_v5(gt_seg):
    boundary_mask = torch.zeros_like(gt_seg)
    B, N, h, w = gt_seg.shape

    for batch_idx in range(B):
        for class_idx in range(N):
            if torch.sum(gt_seg[batch_idx, class_idx] > 0):  # 检查每个类别
                gt_seg_np = gt_seg[batch_idx, class_idx, :, :].cpu().numpy().astype(np.uint8)

                gt_seg_np = sharpen_image(gt_seg_np)

                # 使用高斯模糊进行预处理
                blurred_gt = cv2.GaussianBlur(gt_seg_np, (5, 5), 0)

                # 使用 Sobel 算子提取边界
                sobelx = cv2.Sobel(blurred_gt, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(blurred_gt, cv2.CV_64F, 0, 1, ksize=5)
                abs_grad_x = cv2.convertScaleAbs(sobelx)
                abs_grad_y = cv2.convertScaleAbs(sobely)
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

                # 使用膨胀操作连接断裂的边界
                kernel = np.ones((3, 3), np.uint8)
                dilated_grad = cv2.dilate(grad, kernel, iterations=1)

                # 使用腐蚀操作去除小的噪声点
                eroded_grad = cv2.erode(dilated_grad, kernel, iterations=1)

                # 移除图像边框
                # 1. 检测并移除直线
                edges = cv2.Canny(eroded_grad, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(
                    edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(eroded_grad, (x1, y1),
                                 (x2, y2), (0, 0, 0), 5)

                # 2. 移除边缘区域
                border = 5  # 边框宽度
                eroded_grad[:border, :] = 0
                eroded_grad[-border:, :] = 0
                eroded_grad[:, :border] = 0
                eroded_grad[:, -border:] = 0

                # 3. 只保留最大的连通区域
                num_labels, labels = cv2.connectedComponents(eroded_grad)
                if num_labels > 1:
                    largest_label = 1 + \
                        np.argmax([np.sum(labels == i)
                                  for i in range(1, num_labels)])
                    eroded_grad = np.uint8(labels == largest_label) * 1  # need to be very clear when assigning the border pixel
                else:
                    # 处理没有连通区域的情况
                    eroded_grad = np.zeros_like(eroded_grad)  # 或者其他处理方式

                boundary_mask[batch_idx, class_idx] = torch.from_numpy(eroded_grad).to(gt_seg.device)

    return boundary_mask


def extract_boundary_v5s(gt_seg):
    boundary_mask = torch.zeros_like(gt_seg)
    B, N, h, w = gt_seg.shape

    for batch_idx in range(B):
        for class_idx in range(N):
            if torch.sum(gt_seg[batch_idx, class_idx] > 0):
                gt_seg_np = gt_seg[batch_idx, class_idx, :, :].cpu().numpy().astype(np.uint8)

                gt_seg_np = sharpen_image(gt_seg_np)

                # 使用Canny边缘检测
                edges = cv2.Canny(gt_seg_np, 50, 150)

                # 使用形态学操作细化边界 (可选)
                kernel = np.ones((2, 2), np.uint8)  # 可以尝试调整核的大小
                thin_boundary = cv2.morphologyEx(
                    edges, cv2.MORPH_CLOSE, kernel, iterations=1)

                # **减少边界宽度:**
                # 使用腐蚀操作缩小边界
                erosion_kernel = np.ones((2, 2), np.uint8)  # 调整核大小控制腐蚀程度
                eroded_boundary = cv2.erode(
                    thin_boundary, erosion_kernel, iterations=1)  # 调整迭代次数

                # 移除图像边框
                # 1. 检测并移除直线
                edges = cv2.Canny(eroded_boundary, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(
                    edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(eroded_boundary, (x1, y1),
                                 (x2, y2), (0, 0, 0), 5)

                # 2. 移除边缘区域
                border = 5  # 边框宽度
                eroded_boundary[:border, :] = 0
                eroded_boundary[-border:, :] = 0
                eroded_boundary[:, :border] = 0
                eroded_boundary[:, -border:] = 0

                # 3. 只保留最大的连通区域
                num_labels, labels = cv2.connectedComponents(
                    eroded_boundary)
                if num_labels > 1:
                    largest_label = 1 + \
                        np.argmax([np.sum(labels == i)
                                  for i in range(1, num_labels)])
                    eroded_boundary = np.uint8(labels == largest_label) * 255
                else:
                    # 处理没有连通区域的情况
                    eroded_boundary = np.zeros_like(
                        eroded_boundary)  # 或者其他处理方式

                boundary_mask[batch_idx, class_idx] = torch.from_numpy(eroded_boundary).to(gt_seg.device)

    return boundary_mask

# TODO: 对于lung数据集，会将随机裁剪缩放的图像框也当作边界提取出来


def extract_boundary_v4(gt_seg):
    boundary_mask = torch.zeros_like(gt_seg)
    B, N, h, w = gt_seg.shape

    for batch_idx in range(B):
        for class_idx in range(N):
            if torch.sum(gt_seg[batch_idx, class_idx] > 0):  # 检查每个类别
                gt_seg_np = gt_seg[batch_idx, class_idx,
                                   :, :].cpu().numpy().astype(np.uint8)

                # 应用开运算来消除小的噪声，但使用更小的核
                kernel = np.ones((3, 3), np.uint8)
                gt_seg_np = cv2.morphologyEx(
                    gt_seg_np, cv2.MORPH_OPEN, kernel)

                gt_seg_np = sharpen_image(gt_seg_np)

                # 使用Canny边缘检测，降低阈值使其更敏感
                edges = cv2.Canny(gt_seg_np, 30, 100)

                # 使用轮廓检测
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 创建一个空白掩码来绘制轮廓
                contour_mask = np.zeros_like(gt_seg_np)

                # 降低最小轮廓面积阈值
                min_contour_area = 10  # 可以根据需要进一步调整此值
                for contour in contours:
                    if cv2.contourArea(contour) > min_contour_area:
                        cv2.drawContours(
                            contour_mask, [contour], 0, 255, 1)

                # 如果轮廓掩码是空的，我们可以尝试直接使用Canny边缘
                if np.sum(contour_mask) == 0:
                    print(
                        "Warning: No contours found. Using Canny edges directly.")
                    contour_mask = edges

                boundary_mask[batch_idx, class_idx] = torch.from_numpy(
                    contour_mask).to(gt_seg.device)

    return boundary_mask


def extract_boundary_v1(gt_seg, num_classes):
    boundary_mask = torch.zeros_like(gt_seg)
    for k in range(num_classes):
        if torch.sum(gt_seg == k) > 0:
            boundary_mask[k] = torch.from_numpy(
                cv2.Canny(gt_seg[k].cpu().numpy().astype(np.uint8), 0, 1)).to(gt_seg.device)
    return boundary_mask


def extract_boundary_dilate(gt_seg, num_classes):
    boundary_mask = torch.zeros_like(gt_seg)
    for k in range(num_classes):
        if torch.sum(gt_seg == k) > 0:
            # 使用cv2.dilate对Canny边缘进行膨胀操作
            edges = cv2.Canny(
                gt_seg[k].cpu().numpy().astype(np.uint8), 0, 1)
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            boundary_mask[k] = torch.from_numpy(
                dilated_edges).to(gt_seg.device)
    return boundary_mask


def extract_boundary_sobel(gt_seg):
    boundary_mask = torch.zeros_like(gt_seg)
    B, N, h, w = gt_seg.shape

    for batch_idx in range(B):
        for class_idx in range(N):
            if torch.sum(gt_seg[batch_idx, class_idx] > 0):  # 检查每个类别
                gt_seg_np = gt_seg[batch_idx, class_idx,
                                   :, :].cpu().numpy().astype(np.uint8)

                # 使用高斯模糊进行预处理
                blurred_gt = cv2.GaussianBlur(gt_seg_np, (5, 5), 0)

                # 使用 Sobel 算子提取边界
                sobelx = cv2.Sobel(blurred_gt, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(blurred_gt, cv2.CV_64F, 0, 1, ksize=5)
                abs_grad_x = cv2.convertScaleAbs(sobelx)
                abs_grad_y = cv2.convertScaleAbs(sobely)
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

                boundary_mask[batch_idx, class_idx] = torch.from_numpy(
                    grad).to(gt_seg.device)
    return boundary_mask


def extract_boundary_v3(gt_seg):
    boundary_mask = torch.zeros_like(gt_seg)
    B, N, h, w = gt_seg.shape

    for batch_idx in range(B):
        for class_idx in range(N):
            if torch.sum(gt_seg[batch_idx, class_idx] > 0):  # 检查每个类别
                gt_seg_np = gt_seg[batch_idx, class_idx,
                                   :, :].cpu().numpy().astype(np.uint8)

                gt_seg_np = sharpen_image(gt_seg_np)

                # 使用高斯模糊进行预处理
                blurred_gt = cv2.GaussianBlur(gt_seg_np, (5, 5), 0)

                # 使用 Sobel 算子提取边界
                sobelx = cv2.Sobel(blurred_gt, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(blurred_gt, cv2.CV_64F, 0, 1, ksize=5)
                abs_grad_x = cv2.convertScaleAbs(sobelx)
                abs_grad_y = cv2.convertScaleAbs(sobely)
                grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

                # 使用膨胀操作连接断裂的边界
                kernel = np.ones((3, 3), np.uint8)
                dilated_grad = cv2.dilate(grad, kernel, iterations=1)

                # 使用腐蚀操作去除小的噪声点
                eroded_grad = cv2.erode(dilated_grad, kernel, iterations=1)

                boundary_mask[batch_idx, class_idx] = torch.from_numpy(eroded_grad).to(
                    gt_seg.device
                )

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


def extract_boundary_morphology_advanced(gt_seg, num_classes=2, line_threshold=150):
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


def extract_boundary_morphology(gt_seg, num_classes=2):
    boundary_mask = torch.zeros_like(gt_seg, dtype=torch.float)

    # 将图像边界设置为背景值
    gt_seg[:, :, 0, :] = 0  # 上边界
    gt_seg[:, :, -1, :] = 0  # 下边界
    gt_seg[:, :, :, 0] = 0  # 左边界
    gt_seg[:, :, :, -1] = 0  # 右边界

    for k in range(num_classes):
        mask = (gt_seg == k).float()
        if torch.sum(mask) > 0:
            dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
            eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
            boundary_mask += (dilated - eroded)

    return boundary_mask


def extract_boundary_base(gt_seg, num_classes=2):
    boundary_mask = torch.zeros_like(gt_seg)
    for k in range(num_classes):
        if torch.sum(gt_seg == k) > 0:
            boundary_mask[k] = torch.from_numpy(
                cv2.Canny(gt_seg[k].cpu().numpy().astype(np.uint8), 0, 1)).to(gt_seg.device)
    return boundary_mask


def get_gt_seg(gt_seg):
    return gt_seg
