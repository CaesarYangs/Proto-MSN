import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from lib.utils.tools.logger import Logger as Log


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t()  # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)  # each point/sample's maximum prototype index(每个样本点归属的最大置信度类别)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)  # one-hot mode of indexes, convenient way of discribe indxes above

    return L, indexs


def kmeans_clustering(x, n_clusters=10, max_iters=100, tol=1e-4):
    # 随机初始化聚类中心
    c = x[torch.randperm(x.shape[0])[:n_clusters]]

    for _ in range(max_iters):
        # 计算每个点到聚类中心的距离
        distances = torch.cdist(x, c)

        # 分配每个点到最近的聚类中心
        labels = torch.argmin(distances, dim=1)

        # 更新聚类中心
        new_c = torch.stack([x[labels == k].mean(0) for k in range(n_clusters)])

        # 检查收敛
        if torch.all(torch.abs(new_c - c) < tol):
            break

        c = new_c

    # 计算软分配
    distances = torch.cdist(x, c)
    L = F.softmax(-distances / 0.05, dim=1)

    return L, labels


def dbscan_clustering(x, eps=0.5, min_samples=5):
    # 计算距离矩阵
    distances = torch.cdist(x, x)

    # 找到核心点
    core_points = (distances <= eps).sum(dim=1) >= min_samples

    # 初始化标签
    labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device) - 1

    cluster_id = 0
    for i in range(x.shape[0]):
        if labels[i] != -1 or not core_points[i]:
            continue

        # 扩展聚类
        queue = [i]
        labels[i] = cluster_id
        while queue:
            q = queue.pop(0)
            neighbors = torch.where((distances[q] <= eps) & (labels == -1))[0]
            labels[neighbors] = cluster_id
            queue.extend(neighbors[core_points[neighbors]].tolist())

        cluster_id += 1

    # 计算软分配（这里使用到聚类中心的距离的倒数作为软分配）
    unique_labels = torch.unique(labels[labels != -1])
    centers = torch.stack([x[labels == label].mean(dim=0) for label in unique_labels])
    soft_distances = 1 / (torch.cdist(x, centers) + 1e-5)
    L = soft_distances / soft_distances.sum(dim=1, keepdim=True)

    return L, labels


def gmm_clustering(x, n_components=10, n_iterations=100):
    n_samples, n_features = x.shape

    # 初始化参数
    means = x[torch.randperm(n_samples)[:n_components]]
    covariances = torch.eye(n_features, device=x.device).unsqueeze(0).repeat(n_components, 1, 1)
    weights = torch.ones(n_components, device=x.device) / n_components

    for _ in range(n_iterations):
        # E-step
        log_likelihoods = torch.stack([
            -0.5 * (torch.sum((x.unsqueeze(1) - means) ** 2 / covariances.diagonal(dim1=-2, dim2=-1).unsqueeze(1), dim=-1) +
                    torch.log(covariances.diagonal(dim1=-2, dim2=-1).prod(-1)) +
                    n_features * torch.log(torch.tensor(2 * torch.pi)))
        ], dim=-1).squeeze()

        log_responsibilities = log_likelihoods + torch.log(weights)
        responsibilities = torch.softmax(log_responsibilities, dim=-1)

        # M-step
        N = responsibilities.sum(0)
        means = (responsibilities.unsqueeze(-1) * x.unsqueeze(1)).sum(0) / N.unsqueeze(-1)
        covariances = torch.stack([
            ((x.unsqueeze(1) - means).unsqueeze(-1) * (x.unsqueeze(1) - means).unsqueeze(-2) *
             responsibilities.unsqueeze(-1).unsqueeze(-1)).sum(0) / N.unsqueeze(-1).unsqueeze(-1)
            for _ in range(n_components)])
        weights = N / n_samples

    return responsibilities, torch.argmax(responsibilities, dim=1)


def spectral_clustering(x, n_clusters=10, sigma=1.0):
    # 计算相似度矩阵
    distances = torch.cdist(x, x)
    affinity = torch.exp(-distances ** 2 / (2 * sigma ** 2))

    # 计算拉普拉斯矩阵
    degree = affinity.sum(1)
    laplacian = torch.diag(degree) - affinity

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    # 选择前 k 个特征向量
    k = n_clusters
    features = eigenvectors[:, :k]

    # 在低维空间中应用 K-means
    L, indexs = kmeans_clustering(features, n_clusters)

    return L, indexs


def random_sampling_clustering(x, n_clusters=10):
    n_samples = x.shape[0]

    # 随机选择聚类中心
    center_indices = torch.randperm(n_samples)[:n_clusters]
    centers = x[center_indices]

    # 计算每个点到聚类中心的距离
    distances = torch.cdist(x, centers)

    # 分配每个点到最近的聚类中心
    labels = torch.argmin(distances, dim=1)

    # 计算软分配（可选）
    L = torch.softmax(-distances / 0.1, dim=1)

    return L, labels


def distributed_greenkhorn(out, sinkhorn_iterations=100, epsilon=0.05):
    L = torch.exp(out / epsilon).t()
    K = L.shape[0]
    B = L.shape[1]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    r = torch.ones((K,), dtype=L.dtype).to(L.device) / K
    c = torch.ones((B,), dtype=L.dtype).to(L.device) / B

    r_sum = torch.sum(L, axis=1)
    c_sum = torch.sum(L, axis=0)

    r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
    c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    for _ in range(sinkhorn_iterations):
        i = torch.argmax(r_gain)
        j = torch.argmax(c_gain)
        r_gain_max = r_gain[i]
        c_gain_max = c_gain[j]

        if r_gain_max > c_gain_max:
            scaling = r[i] / r_sum[i]
            old_row = L[i, :]
            new_row = old_row * scaling
            L[i, :] = new_row

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)
        else:
            scaling = c[j] / c_sum[j]
            old_col = L[:, j]
            new_col = old_col * scaling
            L[:, j] = new_col

            L = L / torch.sum(L)
            r_sum = torch.sum(L, axis=1)
            c_sum = torch.sum(L, axis=0)

            r_gain = r_sum - r + r * torch.log(r / r_sum + 1e-5)
            c_gain = c_sum - c + c * torch.log(c / c_sum + 1e-5)

    L = L.t()

    indexs = torch.argmax(L, dim=1)
    G = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs
