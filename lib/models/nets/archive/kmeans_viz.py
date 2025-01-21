#! debug and building now: sub-domain prototype learning
        # section1: get sub-domain
        feats_reshaped = feats.view(batch_size * h * w, num_features)
        kmeans = KMeans(n_clusters=self.num_subdomains, random_state=0)
        cluster_labels = kmeans.fit_predict(feats_reshaped.cpu().detach())  # 对每个像素进行聚类
        cluster_labels = cluster_labels.reshape(batch_size, h, w)
        Log.info(f"content of cluster_labels:{cluster_labels}")
        self.visualize_cluster_res_2_pic(cluster_labels, feats)
        exit(0)
        #! debug ---


def visualize_cluster_res(self, cluster_labels, feats, save_dir='/DATA/yangyeqing/Project-Brown/methods/protoseg_dev/res/others/cluster_viz'):
        """
        可视化 KMeans 聚类结果并将图像保存到磁盘.

        Args:
            cluster_labels (torch.Tensor): 形状为 (batch_size,) 的张量，包含每个像素的聚类标签。
            save_dir (str, optional): 保存图像的目录。默认为 './cluster_visualizations'。
        """

        # 创建保存目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 选择要用于可视化的两个特征维度
        feat_x = 0  # 例如，使用第一个特征维度
        feat_y = 1  # 例如，使用第二个特征维度

        h = 128  # 特征图高度
        w = 64  # 特征图宽度

        # 创建颜色映射，为每个簇分配不同的颜色
        cmap = ListedColormap(plt.cm.get_cmap("tab20", self.num_subdomains).colors)  # 使用 "tab20" 颜色映射

        for idx in range(cluster_labels.shape[0]):  # 遍历 batch
            # 从特征图中提取选择的特征
            feature_map = feats[idx].detach().cpu().numpy()
            x = feature_map[feat_x].reshape(h, w)
            y = feature_map[feat_y].reshape(h, w)

            # 将聚类标签转换为 NumPy 数组
            labels = cluster_labels[idx].reshape(h, w)

            # 创建图形和轴
            fig, ax = plt.subplots()

            # 绘制聚类结果
            for cluster_id in range(self.num_subdomains):
                # 找到属于当前簇的像素
                cluster_mask = labels == cluster_id

                # 绘制属于当前簇的像素点
                ax.scatter(x[cluster_mask], y[cluster_mask], c=cmap(cluster_id), label=f'Cluster {cluster_id}', s=1)

            # 设置图形标题和轴标签
            ax.set_title(f'Clustering Results (Batch {idx + 1})')
            ax.set_xlabel(f'Feature {feat_x + 1}')
            ax.set_ylabel(f'Feature {feat_y + 1}')

            # 添加图例
            ax.legend()

            # 保存图像到磁盘
            plt.savefig(os.path.join(save_dir, f'cluster_visualization_batch_{idx + 1}.png'))
            plt.close(fig)  # 关闭图形以释放内存

    def visualize_cluster_res_2_pic(self, cluster_labels, feats, save_dir='/DATA/yangyeqing/Project-Brown/methods/protoseg_dev/res/others/cluster_viz'):
        """
        将 KMeans 聚类结果可视化到原始图像上，并将图像保存到磁盘.

        Args:
            cluster_labels (torch.Tensor): 形状为 (batch_size, h, w) 的张量，包含每个像素的聚类标签。
            original_image (torch.Tensor): 形状为 (batch_size, channels, original_h, original_w) 的原始图像张量。
            save_dir (str, optional): 保存图像的目录。默认为 './cluster_visualizations'。
        """

        # 创建保存目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 获取特征图尺寸和原始图像尺寸
        batch_size, _, h, w = feats.shape

        # 创建颜色映射，为每个簇分配不同的颜色
        cmap = ListedColormap(plt.cm.get_cmap("tab20", self.num_subdomains).colors)

        for idx in range(batch_size):
            # 获取当前 batch 的聚类标签
            labels = cluster_labels[idx]

            # 将聚类标签映射回原始图像尺寸
            labels_resized = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

            # 创建一个空白图像，用于绘制聚类结果
            clustered_image = np.zeros((h, w, 3), dtype=np.uint8)

            # 为每个像素绘制其所属簇的颜色
            for cluster_id in range(self.num_subdomains):
                clustered_image[labels_resized == cluster_id] = (np.array(cmap(cluster_id)[:3]) * 255).astype(np.uint8)

            # 创建图形和轴
            fig, ax = plt.subplots()

            # 显示聚类后的图像
            ax.imshow(clustered_image)

            # 设置图形标题
            ax.set_title(f'Clustering Results (Batch {idx + 1})')
            ax.axis('off')  # 隐藏坐标轴

            # 保存图像到磁盘
            plt.savefig(os.path.join(save_dir, f'cluster_visualization_batch_{idx + 1}.png'))
            plt.close(fig)  # 关闭图形以释放内存
