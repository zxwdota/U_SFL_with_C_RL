import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons,make_circles,make_swiss_roll
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist



from sklearn.datasets import make_classification

X_unbalence, y_unbalence = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,   # ✅ 显式设置为 1
    weights=[0.8, 0.2],
    random_state=9
)


from sklearn.neighbors import NearestNeighbors

# class KMeans:
#     def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, random_state=None):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.tol = tol
#         self.random_state = random_state
#         self.cluster_centers_ = None
#         self.labels_ = None
#
#     def _kmeans_plus_plus_init(self, X):
#         np.random.seed(self.random_state)
#         n_samples = X.shape[0]
#
#         # 第一个中心点随机选择
#         centers = [X[np.random.choice(n_samples)]]
#
#         # 选取其余的中心点
#         for _ in range(1, self.n_clusters):
#             # 计算每个点到已选中心中最近一个的距离
#             distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centers), axis=2)**2, axis=1)
#             probabilities = distances / distances.sum()
#             cumulative_prob = np.cumsum(probabilities)
#             r = np.random.rand()
#
#             # 选择一个新的点作为中心
#             next_center = X[np.searchsorted(cumulative_prob, r)]
#             centers.append(next_center)
#
#         return np.array(centers)
#
#     def _compute_distances(self, X):
#         return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
#
#     def fit(self, X):
#         if self.random_state is not None:
#             np.random.seed(self.random_state)
#
#         # 使用 KMeans++ 初始化中心
#         self.cluster_centers_ = self._kmeans_plus_plus_init(X)
#
#         for i in range(self.max_iter):
#             # 分配标签
#             distances = self._compute_distances(X)
#             labels = np.argmin(distances, axis=1)
#
#             # 更新中心
#             new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
#
#             # 检查是否收敛
#             shift = np.linalg.norm(self.cluster_centers_ - new_centers)
#             if shift < self.tol:
#                 break
#             self.cluster_centers_ = new_centers
#
#         self.labels_ = labels
# class CSKMeans:
#     def __init__(self, n_clusters=2, max_iter=100, tol_kmeans=1e-4, tol_overlap=0.1, random_state=None):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.tol_kmeans = tol_kmeans
#         self.tol_overlap = tol_overlap
#         self.random_state = random_state
#         self.cluster_centers_ = None
#         self.labels_ = None
#
#     def _smart_init(self, X):
#         n_samples = X.shape[0]
#         dist_matrix = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
#         sum_dists = dist_matrix.sum(axis=1)
#         first_center = X[np.argmin(sum_dists)]
#         second_center = X[np.argmax(sum_dists)]
#         return np.array([first_center, second_center])
#
#     def _compute_distances(self, X):
#         return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
#
#     def _compute_bounds(self, X, labels):
#         bounds = []
#         for i in range(self.n_clusters):
#             points = X[labels == i]
#             if len(points) == 0:
#                 bounds.append(None)
#             else:
#                 bounds.append((points.min(axis=0), points.max(axis=0)))
#         return bounds
#
#     def _compute_overlap_volume(self, bounds1, bounds2):
#         if bounds1 is None or bounds2 is None:
#             return 0.0
#         min1, max1 = bounds1
#         min2, max2 = bounds2
#         overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
#         overlap = np.maximum(overlap, 0)  # 负值变 0
#         return np.prod(overlap)
#
#     def _compute_total_overlap(self, bounds):
#         max_overlap = 0
#         for i in range(self.n_clusters):
#             for j in range(i + 1, self.n_clusters):
#                 vol = self._compute_overlap_volume(bounds[i], bounds[j])
#                 max_overlap = max(max_overlap, vol)
#         return max_overlap
#
#     def fit(self, X):
#         if self.random_state is not None:
#             np.random.seed(self.random_state)
#
#         self.cluster_centers_ = self._smart_init(X)
#         self.n_clusters = 2
#         prev_overlap = float('inf')
#         center_history = []
#
#         for iter_num in range(self.max_iter):
#             for i in range(self.max_iter):
#                 # 标准 K-means 步骤
#                 print('a')
#                 distances = self._compute_distances(X)
#                 labels = np.argmin(distances, axis=1)
#
#                 # 更新中心
#                 new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
#
#
#                 shift = np.linalg.norm(self.cluster_centers_ - new_centers)
#                 if shift < self.tol_kmeans:
#                     break
#                 self.cluster_centers_ = new_centers
#                 # plot_clusters(X, labels, self.cluster_centers_)
#             self.labels_ = labels
#
#             # NSK 特有退出判断：计算重叠空间
#             bounds = self._compute_bounds(X, labels)
#             curr_overlap = self._compute_total_overlap(bounds)
#             print(f"迭代 {iter_num + 1}: 最大簇重叠体积 = {curr_overlap:.5f}")
#
#             # if abs(prev_overlap - curr_overlap) < self.tol_overlap or curr_overlap < self.tol_overlap:
#             #     print("满足退出条件，结束迭代")
#             #     break
#
#             # prev_overlap = curr_overlap
#
#             # === Step 2-b: 从密度最低簇中，选一个与簇内点总距离最大的点作为新中心 ===
#
#             # 1. 计算每个簇的边界体积（lengths in all dimensions）
#             bounds = self._compute_bounds(X, self.labels_)
#             densities = []
#             for i in range(self.n_clusters):
#                 pts = X[self.labels_ == i]
#                 if len(pts) == 0 or bounds[i] is None:
#                     densities.append(np.inf)
#                     continue
#                 length = bounds[i][1] - bounds[i][0]  # max - min
#                 volume = np.prod(length + 1e-8)  # 避免乘积为 0
#                 density = len(pts) / volume
#                 densities.append(density)
#
#             # 2. 找出密度最低的簇
#             low_density_idx = np.argmin(densities)
#             pts = X[self.labels_ == low_density_idx]
#
#             # 3. 在该簇中选出与所有点距离和最大的点
#             dist_matrix = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
#             sum_dists = dist_matrix.sum(axis=1)
#             new_center = pts[np.argmax(sum_dists)]
#
#             # 4. 加入新的中心点
#             self.cluster_centers_ = np.vstack([self.cluster_centers_, new_center])
#             self.n_clusters = self.cluster_centers_.shape[0]
#
#             # =========================================
#
#             # # ==== 判断是否有重复中心点 ====
#             # center_history.append(tuple(new_center))
#             # from collections import Counter
#             # counts = Counter(center_history)
#             # if any(v > 2 for v in counts.values()):
#             #     print("检测到初始中心重复超过两次，提前终止迭代")
#             #     break
#
#             # 计算本轮最大重叠体积和总重叠体积
#             bounds = self._compute_bounds(X, self.labels_)
#             S_all = 0
#             S_max = 0
#
#             for i in range(self.n_clusters):
#                 for j in range(i + 1, self.n_clusters):
#                     vol = self._compute_overlap_volume(bounds[i], bounds[j])
#                     S_all += vol
#                     S_max = max(S_max, vol)
#
#             print(f"迭代 {iter_num + 1}: 最大重叠体积 = {S_max:.5f}, 总重叠体积 = {S_all:.5f}")
#
#             # 停止条件判断
#             if S_all == 0:
#                 print("满足理想停止条件（重叠空间为0），终止迭代。")
#                 break
#             if S_max <= self.tol_overlap:
#                 print("满足非理想停止条件（最大重叠体积小于阈值），终止迭代。")
#                 break
#             if iter_num > 0 and S_max == prev_S_max:
#                 print("满足非理想停止条件（最大重叠体积不再变化），终止迭代。")
#                 break
#
#             prev_S_max = S_max
#
#
#         # --- 后处理阶段：CSK-means 的重叠点归属逻辑 ---
#         bounds = self._compute_bounds(X, self.labels_)
#
#         for i in range(self.n_clusters):
#             for j in range(i + 1, self.n_clusters):
#                 pts_i = X[self.labels_ == i]
#                 pts_j = X[self.labels_ == j]
#                 pts = np.vstack([pts_i, pts_j])
#
#                 # 计算交集边界
#                 if bounds[i] is None or bounds[j] is None:
#                     continue
#                 min_bound = np.maximum(bounds[i][0], bounds[j][0])
#                 max_bound = np.minimum(bounds[i][1], bounds[j][1])
#
#                 mask = np.all((pts >= min_bound) & (pts <= max_bound), axis=1)
#                 overlap_pts = pts[mask]
#
#                 for pt in overlap_pts:
#                     # 模拟将 pt 加入 i 后的密度
#                     pts_i_plus = np.vstack([pts_i, pt])
#                     vol_i = np.prod(bounds[i][1] - bounds[i][0] + 1e-8)
#                     rho_i_new = len(pts_i_plus) / vol_i
#
#                     # 模拟将 pt 加入 j 后的密度
#                     pts_j_plus = np.vstack([pts_j, pt])
#                     vol_j = np.prod(bounds[j][1] - bounds[j][0] + 1e-8)
#                     rho_j_new = len(pts_j_plus) / vol_j
#
#                     # 归属到密度更大的那个簇
#                     self.labels_[np.all(X == pt, axis=1)] = i if rho_i_new > rho_j_new else j
# class CSKMedoids:
#     def __init__(self, n_clusters=2, max_iter=100, tol_kmeans=1e-4, tol_overlap=1e-4, random_state=None):
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#         self.tol_kmeans = tol_kmeans
#         self.tol_overlap = tol_overlap
#         self.random_state = random_state
#         self.cluster_centers_ = None
#         self.labels_ = None
#
#     def _smart_init(self, X):
#         n_samples = X.shape[0]
#         dist_matrix = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
#         sum_dists = dist_matrix.sum(axis=1)
#         first_center = X[np.argmin(sum_dists)]
#         second_center = X[np.argmax(sum_dists)]
#         return np.array([first_center, second_center])
#
#     def _compute_distances(self, X):
#         return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
#
#     def _compute_bounds(self, X, labels):
#         bounds = []
#         for i in range(self.n_clusters):
#             points = X[labels == i]
#             if len(points) == 0:
#                 bounds.append(None)
#             else:
#                 bounds.append((points.min(axis=0), points.max(axis=0)))
#         return bounds
#
#     def _compute_overlap_volume(self, bounds1, bounds2):
#         if bounds1 is None or bounds2 is None:
#             return 0.0
#         min1, max1 = bounds1
#         min2, max2 = bounds2
#         overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
#         overlap = np.maximum(overlap, 0)  # 负值变 0
#         return np.prod(overlap)
#
#     def _compute_total_overlap(self, bounds):
#         max_overlap = 0
#         for i in range(self.n_clusters):
#             for j in range(i + 1, self.n_clusters):
#                 vol = self._compute_overlap_volume(bounds[i], bounds[j])
#                 max_overlap = max(max_overlap, vol)
#         return max_overlap
#
#     def fit(self, X):
#         if self.random_state is not None:
#             np.random.seed(self.random_state)
#
#         self.cluster_centers_ = self._smart_init(X)
#         self.n_clusters = 2
#         prev_overlap = float('inf')
#         center_history = []
#
#         for iter_num in range(self.max_iter):
#             for i in range(self.max_iter):
#                 # 标准 K-means 步骤
#                 print('a')
#                 distances = self._compute_distances(X)
#                 labels = np.argmin(distances, axis=1)
#
#                 # 更新中心
#                 new_centers = []
#                 for j in range(self.n_clusters):
#                     cluster_points = X[labels == j]
#                     if len(cluster_points) == 0:
#                         new_centers.append(self.cluster_centers_[j])  # 保留原中心
#                         continue
#                     # 计算每个点到簇内所有点的距离和，取最小值的点
#                     dist_matrix = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points[np.newaxis, :], axis=2)
#                     sum_dists = dist_matrix.sum(axis=1)
#                     medoid = cluster_points[np.argmin(sum_dists)]
#                     new_centers.append(medoid)
#                 new_centers = np.array(new_centers)
#
#
#                 shift = np.linalg.norm(self.cluster_centers_ - new_centers)
#                 if shift < self.tol_kmeans:
#                     break
#                 self.cluster_centers_ = new_centers
#                 # plot_clusters(X, labels, self.cluster_centers_)
#             self.labels_ = labels
#
#             # NSK 特有退出判断：计算重叠空间
#             bounds = self._compute_bounds(X, labels)
#             curr_overlap = self._compute_total_overlap(bounds)
#             print(f"迭代 {iter_num + 1}: 最大簇重叠体积 = {curr_overlap:.5f}")
#
#             # if abs(prev_overlap - curr_overlap) < self.tol_overlap or curr_overlap < self.tol_overlap:
#             #     print("满足退出条件，结束迭代")
#             #     break
#
#             # prev_overlap = curr_overlap
#
#             # === Step 2-b: 从密度最低簇中，选一个与簇内点总距离最大的点作为新中心 ===
#
#             # 1. 计算每个簇的边界体积（lengths in all dimensions）
#             bounds = self._compute_bounds(X, self.labels_)
#             densities = []
#             for i in range(self.n_clusters):
#                 pts = X[self.labels_ == i]
#                 if len(pts) == 0 or bounds[i] is None:
#                     densities.append(np.inf)
#                     continue
#                 length = bounds[i][1] - bounds[i][0]  # max - min
#                 volume = np.prod(length + 1e-8)  # 避免乘积为 0
#                 density = len(pts) / volume
#                 densities.append(density)
#
#             # 2. 找出密度最低的簇
#             low_density_idx = np.argmin(densities)
#             pts = X[self.labels_ == low_density_idx]
#
#             # 3. 在该簇中选出与所有点距离和最大的点
#             dist_matrix = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
#             sum_dists = dist_matrix.sum(axis=1)
#             new_center = pts[np.argmax(sum_dists)]
#
#             # 4. 加入新的中心点
#             self.cluster_centers_ = np.vstack([self.cluster_centers_, new_center])
#             self.n_clusters = self.cluster_centers_.shape[0]
#
#             # =========================================
#
#             # # ==== 判断是否有重复中心点 ====
#             # center_history.append(tuple(new_center))
#             # from collections import Counter
#             # counts = Counter(center_history)
#             # if any(v > 2 for v in counts.values()):
#             #     print("检测到初始中心重复超过两次，提前终止迭代")
#             #     break
#
#             # 计算本轮最大重叠体积和总重叠体积
#             bounds = self._compute_bounds(X, self.labels_)
#             S_all = 0
#             S_max = 0
#
#             for i in range(self.n_clusters):
#                 for j in range(i + 1, self.n_clusters):
#                     vol = self._compute_overlap_volume(bounds[i], bounds[j])
#                     S_all += vol
#                     S_max = max(S_max, vol)
#
#             print(f"迭代 {iter_num + 1}: 最大重叠体积 = {S_max:.5f}, 总重叠体积 = {S_all:.5f}")
#
#             # 停止条件判断
#             if S_all == 0:
#                 print("满足理想停止条件（重叠空间为0），终止迭代。")
#                 break
#             if S_max <= self.tol_overlap:
#                 print("满足非理想停止条件（最大重叠体积小于阈值），终止迭代。")
#                 break
#             if iter_num > 0 and S_max == prev_S_max:
#                 print("满足非理想停止条件（最大重叠体积不再变化），终止迭代。")
#                 break
#
#             prev_S_max = S_max
#
#
#         # --- 后处理阶段：CSK-means 的重叠点归属逻辑 ---
#         bounds = self._compute_bounds(X, self.labels_)
#
#         for i in range(self.n_clusters):
#             for j in range(i + 1, self.n_clusters):
#                 pts_i = X[self.labels_ == i]
#                 pts_j = X[self.labels_ == j]
#                 pts = np.vstack([pts_i, pts_j])
#
#                 # 计算交集边界
#                 if bounds[i] is None or bounds[j] is None:
#                     continue
#                 min_bound = np.maximum(bounds[i][0], bounds[j][0])
#                 max_bound = np.minimum(bounds[i][1], bounds[j][1])
#
#                 mask = np.all((pts >= min_bound) & (pts <= max_bound), axis=1)
#                 overlap_pts = pts[mask]
#
#                 for pt in overlap_pts:
#                     # 模拟将 pt 加入 i 后的密度
#                     pts_i_plus = np.vstack([pts_i, pt])
#                     vol_i = np.prod(bounds[i][1] - bounds[i][0] + 1e-8)
#                     rho_i_new = len(pts_i_plus) / vol_i
#
#                     # 模拟将 pt 加入 j 后的密度
#                     pts_j_plus = np.vstack([pts_j, pt])
#                     vol_j = np.prod(bounds[j][1] - bounds[j][0] + 1e-8)
#                     rho_j_new = len(pts_j_plus) / vol_j
#
#                     # 归属到密度更大的那个簇
#                     self.labels_[np.all(X == pt, axis=1)] = i if rho_i_new > rho_j_new else j
#


def plot_clusters(X, labels, centroids):
    plt.figure(figsize=(4.5, 4), dpi=300)
    plt.axes([0.17, 0.12, 0.8, 0.85])    # 绘制数据点，根据聚类标签用不同颜色标记
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        plt.scatter(
            X[mask, 0], X[mask, 1],
            s=50, alpha=0.7, edgecolors='k',
            label=f'Cluster {label}'
        )
    # 绘制聚类中心，用红色“X”标出
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.legend()
    plt.show()



def try_merge_clusters(i, j, X, labels, cluster_centers, margin=0.1, dist_threshold=0.5):
    # 提取两个簇的点
    pts_i = X[labels == i]
    pts_j = X[labels == j]

    if len(pts_i) == 0 or len(pts_j) == 0:
        return False, labels, cluster_centers

    # 计算边界并放大重叠空间
    min_i, max_i = pts_i.min(axis=0), pts_i.max(axis=0)
    min_j, max_j = pts_j.min(axis=0), pts_j.max(axis=0)

    min_bound = np.maximum(min_i, min_j) - margin
    max_bound = np.minimum(max_i, max_j) + margin

    # 获取落在放大重叠区域的点
    mask = np.all((X >= min_bound) & (X <= max_bound), axis=1)
    overlap_pts = X[mask]

    if len(overlap_pts) == 0:
        return False, labels, cluster_centers

    # 分别提取这两个簇中落入 overlap 的点
    overlap_i = pts_i[np.all((pts_i >= min_bound) & (pts_i <= max_bound), axis=1)]
    overlap_j = pts_j[np.all((pts_j >= min_bound) & (pts_j <= max_bound), axis=1)]

    if len(overlap_i) == 0 or len(overlap_j) == 0:
        return False, labels, cluster_centers

    # 计算两个簇之间在重叠区内点的平均距离
    dist_matrix = cdist(overlap_i, overlap_j)
    avg_dist = np.mean(dist_matrix)

    if avg_dist < dist_threshold:
        print(f"合并簇 {i} 和 {j}，重叠区点距={avg_dist:.4f}")
        labels[labels == j] = i

        # 重新编号
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        labels = np.array([label_map[l] for l in labels])

        # 更新中心点
        cluster_centers = np.array([
            X[labels == l].mean(axis=0) for l in range(len(np.unique(labels)))
        ])
        return True, labels, cluster_centers

    return False, labels, cluster_centers

def greedy_merge_by_overlap_distance(X, labels, K, assign_k, margin=0.1, dist_threshold=0.8):

    cluster_ids = np.unique(labels)
    n_clusters = len(cluster_ids)
    merged = False
    best_pair = None
    best_dist = np.inf
    # cdist_dict = np.full((n_clusters, n_clusters), np.inf)
    cdist_dict = {(i, j): np.inf for i in range(n_clusters) for j in range(i+1,n_clusters+1)}
    # 遍历所有簇对，找出扩展重叠区域平均距离最小的一对
    while True:
        for idx_i in range(n_clusters):
            for idx_j in range(idx_i + 1, n_clusters):
                ci = cluster_ids[idx_i]
                cj = cluster_ids[idx_j]
                pts_i = X[labels == ci]
                pts_j = X[labels == cj]

                if len(pts_i) == 0 or len(pts_j) == 0:
                    continue

                # 计算扩展重叠区域
                min_i, max_i = pts_i.min(axis=0), pts_i.max(axis=0)
                min_j, max_j = pts_j.min(axis=0), pts_j.max(axis=0)
                min_bound = np.maximum(min_i, min_j) - margin
                max_bound = np.minimum(max_i, max_j) + margin

                # 获取两个簇中落入扩展区域的点
                overlap_i = pts_i[np.all((pts_i >= min_bound) & (pts_i <= max_bound), axis=1)]
                overlap_j = pts_j[np.all((pts_j >= min_bound) & (pts_j <= max_bound), axis=1)]

                if len(overlap_i) == 0 or len(overlap_j) == 0:
                    continue

                # 计算扩展重叠区中跨簇点的平均距离

                avg_dist = np.mean(cdist(overlap_i, overlap_j))
                cdist_dict[(idx_i, idx_j)] = avg_dist
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_pair = (ci, cj)

        best_pair, best_dist = min(cdist_dict.items(), key=lambda x: x[1])
        # if best_dist < dist_threshold:
        if assign_k:
            if len(np.unique(labels)) > K:
                ci, cj = best_pair
                print(f"合并簇 {ci} 和 {cj}，扩展重叠区平均距离 = {best_dist:.4f}")
                labels[labels == cj] = ci

                # 重新编号 labels 为连续整数
                unique_labels = np.unique(labels)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                labels = np.array([label_map[l] for l in labels])
                cdist_dict[(ci, cj)] = np.inf
                cluster_centers = np.array([
                    X[labels == l].mean(axis=0) for l in np.unique(labels)
                ])
                # plot_clusters(X, labels, cluster_centers)

            else:
                print(f"最小平均距离 {best_dist:.4f} ≥ 阈值 {dist_threshold}，停止合并")
                break
        else:
            if best_dist < dist_threshold:
                ci, cj = best_pair
                print(f"合并簇 {ci} 和 {cj}，扩展重叠区平均距离 = {best_dist:.4f}")
                labels[labels == cj] = ci

                # 重新编号 labels 为连续整数
                unique_labels = np.unique(labels)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                labels = np.array([label_map[l] for l in labels])
                cdist_dict[(ci, cj)] = np.inf
                cluster_centers = np.array([
                        X[labels == l].mean(axis=0) for l in np.unique(labels)
                    ])
                # plot_clusters(X, labels, cluster_centers)
            else:
                print(f"最小平均距离 {best_dist:.4f} ≥ 阈值 {dist_threshold}，停止合并")
                break


    # 重新计算中心
    cluster_centers = np.array([
        X[labels == l].mean(axis=0) for l in np.unique(labels)
    ])
    return labels, cluster_centers
class NSKMeans:
    def __init__(self,n_clusters=2, max_iter=100, tol_kmeans=1e-4, tol_overlap=1e-2, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol_kmeans = tol_kmeans
        self.tol_overlap = tol_overlap
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def merge_clusters_by_silhouette(self, X, threshold=0.0):
        while True:
            merged = False
            for i in range(self.n_clusters):
                for j in range(i + 1, self.n_clusters):
                    original_score = silhouette_score(X, self.labels_)

                    # 尝试合并簇 i 和 j
                    labels_merged = self.labels_.copy()
                    labels_merged[labels_merged == j] = i

                    # 重新映射标签为连续整数
                    unique_labels = np.unique(labels_merged)
                    label_map = {old: new for new, old in enumerate(unique_labels)}
                    labels_merged = np.array([label_map[l] for l in labels_merged])

                    try:
                        new_score = silhouette_score(X, labels_merged)
                    except:
                        continue  # 有时数据太少或簇数=1会出错

                    if new_score >= original_score - threshold:
                        print(f"合并簇 {i} 和 {j}，轮廓系数从 {original_score:.4f} → {new_score:.4f}")
                        self.labels_ = labels_merged
                        self.cluster_centers_ = np.array([
                            X[self.labels_ == l].mean(axis=0)
                            for l in range(len(np.unique(self.labels_)))
                        ])
                        self.n_clusters = len(self.cluster_centers_)
                        # plot_clusters(X, self.labels_, self.cluster_centers_)
                        merged = True
                        break  # 跳出 j 循环
                if merged:
                    break  # 跳出 i 循环并从头开始
            if not merged:
                break  # 所有组合都尝试了，没有合并，退出

    def _smart_init(self, X):
        n_samples = X.shape[0]
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)
        sum_dists = dist_matrix.sum(axis=1)
        first_center = X[np.argmin(sum_dists)]
        second_center = X[np.argmax(sum_dists)]
        return np.array([first_center, second_center])

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)

    def _compute_bounds(self, X, labels):
        bounds = []
        for i in range(self.n_clusters):
            points = X[labels == i]
            if len(points) == 0:
                bounds.append(None)
            else:
                bounds.append((points.min(axis=0), points.max(axis=0)))
        return bounds

    def _compute_overlap_volume(self, bounds1, bounds2):
        if bounds1 is None or bounds2 is None:
            return 0.0
        min1, max1 = bounds1
        min2, max2 = bounds2
        overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
        overlap = np.maximum(overlap, 0)  # 负值变 0
        return np.prod(overlap)

    def _compute_total_overlap(self, bounds):
        max_overlap = 0
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                vol = self._compute_overlap_volume(bounds[i], bounds[j])
                max_overlap = max(max_overlap, vol)
        return max_overlap

    def fit(self, X, K, assign_k):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.cluster_centers_ = self._smart_init(X)
        self.n_clusters = 2
        self.K = K
        self.assign_k = assign_k

        for iter_num in range(self.max_iter):
            for i in range(self.max_iter):
                # 标准 K-means 步骤
                distances = self._compute_distances(X)
                labels = np.argmin(distances, axis=1)
                # 更新中心
                new_centers = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])
                shift = np.linalg.norm(self.cluster_centers_ - new_centers)
                if shift < self.tol_kmeans:
                    break
                self.cluster_centers_ = new_centers
                # plot_clusters(X, labels, self.cluster_centers_)
            self.labels_ = labels


            # plot_clusters(X, labels, self.cluster_centers_)


            bounds = self._compute_bounds(X, self.labels_)
            S_all = 0
            S_max = 0
            for i in range(self.n_clusters):
                for j in range(i + 1, self.n_clusters):
                    vol = self._compute_overlap_volume(bounds[i], bounds[j])
                    S_all += vol
                    S_max = max(S_max, vol)

            print(f"迭代 {iter_num + 1}: 最大重叠体积 = {S_max:.5f}, 总重叠体积 = {S_all:.5f}")

            # 停止条件判断
            if S_all == 0:
                print("满足理想停止条件（重叠空间为0），终止迭代。")
                break
            if S_max <= self.tol_overlap:
                print("满足非理想停止条件（最大重叠体积小于阈值），终止迭代。")
                break
            if iter_num > 0 and S_max == prev_S_max:
                print("满足非理想停止条件（最大重叠体积不再变化），终止迭代。")
                break

            prev_S_max = S_max

            densities = []
            for i in range(self.n_clusters):
                subx = X[self.labels_ == i]
                if len(subx) == 0 or bounds[i] is None:
                    densities.append(np.inf)
                    continue
                length = bounds[i][1] - bounds[i][0]  # max - min
                volume = np.prod(length + 1e-8)  # 避免乘积为 0
                density = len(subx) / volume
                densities.append(density)
            # 2. 找出密度最低的簇
            low_density_idx = np.argmin(densities)
            subx = X[self.labels_ == low_density_idx]

            # 3. 在该簇中选出与所有点距离和最大的点

            dist_matrix = np.linalg.norm(subx[:, np.newaxis] - subx[np.newaxis, :], axis=2)
            sum_dists = dist_matrix.sum(axis=1)
            new_center = subx[np.argmax(sum_dists)]

            # 4. 加入新的中心点
            self.cluster_centers_ = np.vstack([self.cluster_centers_, new_center])
            self.n_clusters = self.cluster_centers_.shape[0]

            # 计算本轮最大重叠体积和总重叠体积



        # --- 后处理阶段：CSK-means 的重叠点归属逻辑 ---

        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                pts_i = X[self.labels_ == i]
                pts_j = X[self.labels_ == j]
                pts = np.vstack([pts_i, pts_j])

                # 计算交集边界
                if bounds[i] is None or bounds[j] is None:
                    continue
                min_bound = np.maximum(bounds[i][0], bounds[j][0])
                max_bound = np.minimum(bounds[i][1], bounds[j][1])

                mask = np.all((pts >= min_bound) & (pts <= max_bound), axis=1)
                overlap_pts = pts[mask]

                for pt in overlap_pts:
                    # 模拟将 pt 加入 i 后的密度
                    pts_i_plus = np.vstack([pts_i, pt])
                    vol_i = np.prod(bounds[i][1] - bounds[i][0] + 1e-8)
                    rho_i_new = len(pts_i_plus) / vol_i

                    # 模拟将 pt 加入 j 后的密度
                    pts_j_plus = np.vstack([pts_j, pt])
                    vol_j = np.prod(bounds[j][1] - bounds[j][0] + 1e-8)
                    rho_j_new = len(pts_j_plus) / vol_j

                    # 归属到密度更大的那个簇
                    self.labels_[np.all(X == pt, axis=1)] = i if rho_i_new > rho_j_new else j
        self.labels_, self.cluster_centers_ = greedy_merge_by_overlap_distance(
            X, self.labels_, K=self.K, assign_k=self.assign_k, margin=0.1, dist_threshold=0.8
        )
        self.n_clusters = len(np.unique(self.labels_))
        # merged = True
        # while merged:
        #     merged = False
        #     for i in range(self.n_clusters):
        #         for j in range(i + 1, self.n_clusters):
        #             merged_once, self.labels_, self.cluster_centers_ = try_merge_clusters(
        #                 i, j, X, self.labels_, self.cluster_centers_, margin=0.1, dist_threshold=0.8
        #             )
        #             if merged_once:
        #                 self.n_clusters = len(self.cluster_centers_)
        #                 merged = True
        #                 break
        #         if merged:
        #             break


# 生成双月数据集
# n_samples 控制样本数量，noise 控制噪声水平
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
# X, y_true = make_circles(n_samples=3000, noise=0.05, factor=0.5)
# X, y_true = make_swiss_roll(n_samples=300)

# 初始化并运行 K-means 聚类
# 我们设置簇数为2，符合双月数据的数量
use_k_or_non_k = False
k = 2
kmeans = NSKMeans(n_clusters=k, random_state=42)
kmeans.fit(X, k, use_k_or_non_k)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plot_clusters(X, labels, centroids)
# 可视化聚类结果
