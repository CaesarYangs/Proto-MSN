
class Proto_Pro_BPL_T1_HRNet_W48(nn.Module):
    """
    首个coding结束的基础prototype模型
    base prototype + boundary prototype modeling
    backup model
    """

    def __init__(self, configer, experiment=None):
        super(Proto_Pro_BPL_T1_HRNet_W48, self).__init__()
        self.configer = configer
        self.gamma = self.configer.get('protoseg', 'gamma')
        self.num_prototype = self.configer.get('protoseg', 'num_prototype')
        self.use_prototype = self.configer.get('protoseg', 'use_prototype')
        self.num_boundary_prototype = self.configer.get(
            'protoseg', 'num_boundary_prototype')
        self.update_prototype = self.configer.get(
            'protoseg', 'update_prototype')
        self.pretrain_prototype = self.configer.get(
            'protoseg', 'pretrain_prototype')
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.experiment = experiment

        # backbone section
        in_channels = 720
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.10)
        )

        # regular prototype learning section
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

        # boundary prototype learning section
        self.boundary_cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(
                in_channels, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        self.boundary_prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                                requires_grad=True)
        trunc_normal_(self.boundary_prototypes, std=0.02)

    def extract_boundary_v5s(self, gt_seg):
        boundary_mask = torch.zeros_like(gt_seg)
        B, N, h, w = gt_seg.shape

        for batch_idx in range(B):
            for class_idx in range(N):
                if torch.sum(gt_seg[batch_idx, class_idx] > 0):
                    gt_seg_np = gt_seg[batch_idx, class_idx, :, :].cpu().numpy().astype(np.uint8)

                    gt_seg_np = self.sharpen_image(gt_seg_np)

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

                    # # 改进的连通组件分析
                    # num_labels, labels = cv2.connectedComponents(eroded_grad)

                    # # 计算每个连通组件的面积
                    # areas = [np.sum(labels == i) for i in range(1, num_labels)]

                    # # 找出最大的连通组件
                    # largest_label = np.argmax(areas) + 1
                    # largest_area = areas[largest_label - 1]

                    # # 创建一个掩码，初始时只包含最大的连通组件
                    # mask = (labels == largest_label)

                    # # 添加其他足够大的连通组件
                    # for i, area in enumerate(areas, start=1):
                    #     if i != largest_label and area > 0.1 * largest_area:  # 保留面积大于最大组件10%的组件
                    #         mask = mask | (labels == i)

                    # # 应用掩码
                    # eroded_grad = np.uint8(mask) * 255

                    # boundary_mask[batch_idx, class_idx] = torch.from_numpy(eroded_grad).to(gt_seg.device)

        return boundary_mask

    def boundary_gt_seg_vis(self, boudnary_gt_seg, save_dir):
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

    # ******************************************* MAIN PROTOTYPE SECTION *******************************************
    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(
            _c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + \
                (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

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

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[boundary_gt_seg == k]

            c_k = _c[boundary_gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile',
                              tile=self.num_boundary_prototype)

            m_q = q * m_k_tile

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile

            f = m_q.transpose(0, 1) @ c_q

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=boundary_protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                boundary_protos[k, n != 0, :] = new_value

            boundary_proto_target[boundary_gt_seg == k] = indexs.float(
            ) + (self.num_boundary_prototype * k)

        self.boundary_prototypes = nn.Parameter(l2_normalize(boundary_protos),
                                                requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            boundary_protos = self.boundary_prototypes.data.clone()
            dist.all_reduce(boundary_protos.div_(dist.get_world_size()))
            self.boundary_prototypes = nn.Parameter(
                boundary_protos, requires_grad=False)

        return boundary_proto_logits, boundary_proto_target

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):
        # ^ backbone feat handle section
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(
            h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(
            h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(
            h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)

        c = self.cls_head(feats)
        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        boundary_c = self.boundary_cls_head(feats)
        boundary_c = self.proj_head(boundary_c)
        _boundary_c = rearrange(boundary_c, ' b c h w -> (b h w) c')
        _boundary_c = self.feat_norm(_boundary_c)
        _boundary_c = l2_normalize(_boundary_c)

        # ^ prototype prepare section
        self.prototypes.data.copy_(l2_normalize(self.prototypes))
        self.boundary_prototypes.data.copy_(
            l2_normalize(self.boundary_prototypes))

        # ^ mask & outseg generating section
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w",
                            b=feats.shape[0], h=feats.shape[2])

        b_masks = torch.einsum('nd,kmd->nmk', _boundary_c,
                               self.boundary_prototypes)

        b_out_seg = torch.amax(b_masks, dim=1)
        b_out_seg = self.mask_norm(b_out_seg)
        b_out_seg = rearrange(b_out_seg, "(b h w) k -> b k h w",
                              b=feats.shape[0], h=feats.shape[2])

        # ^ prototype learning and update section
        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)

            b_gt_seg_gen = self.extract_boundary_v5(gt_semantic_seg)
            b_gt_seg_gen_1d = F.interpolate(b_gt_seg_gen.float(), size=feats.size()[2:], mode='nearest').view(-1)

            # #! for debug here:
            # Log.info(f"size of gt_semantic_seg:{gt_semantic_seg.size()}")
            # Log.info(f"size of b_gt_seg_gen:{b_gt_seg_gen.size()}")
            # test_boundary_save_dir = "/DATA/yangyeqing/Project-Brown/methods/protoseg_dev/res/others/b_gt_seg_viz_lung_test"
            # self.boundary_gt_seg_vis(b_gt_seg_gen,test_boundary_save_dir)
            # exit(0)

            contrast_logits, contrast_target = self.prototype_learning(
                _c, out_seg, gt_seg, masks)
            boundary_contrast_logits, boundary_contrast_target = self.boundary_prototype_learning(
                _c, b_out_seg, b_gt_seg_gen_1d, b_masks)

            return {'seg': out_seg, 'boundary_seg': b_out_seg, 'logits': contrast_logits, 'target': contrast_target,
                    'boundary_logits': boundary_contrast_logits, 'boundary_target': boundary_contrast_target}

        return out_seg
