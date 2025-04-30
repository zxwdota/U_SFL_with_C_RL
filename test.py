from sklearn.datasets import make_moons
import numpy as np
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
import matplotlib.pyplot as plt
plt.plot(X[:,0],X[:,1],'b.')
plt.show()
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

# dbscan2 = DBSCAN(eps=0.2, min_samples=5)
# dbscan2.fit(X)

# 查看前10个的标签，注意-1在dbscan中是离群点的意思
dbscan.labels_[:10]
# 输出：array([ 0,  2, -1, -1,  1,  0,  0,  0,  2,  5], dtype=int64)

# 查看核心样本应的索引，可以用来获取核心对象
dbscan.core_sample_indices_[:10]
# 输出：array([ 0,  4,  5,  6,  7,  8, 10, 11, 12, 13], dtype=int64)

# 还可以看一下，该算法运行后聚类了几簇，需要去掉-1，这里还是代表离群的那些数据
np.unique(dbscan.labels_)


# 输出：array([-1,  0,  1,  2,  3,  4,  5,  6], dtype=int64)

# 封装绘制函数，都和之前绘制图形的原理类似
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True  # 核心对象标记
    anomalies_mask = dbscan.labels_ == -1  # 离群数据
    non_core_mask = ~(core_mask | anomalies_mask)  # 不是核心数据的

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


# 调用函数绘图
plt.figure(figsize=(9, 3.2))
plt.subplot(121)
plot_dbscan(dbscan, X, size=100)

# plt.subplot(122)
# plot_dbscan(dbscan2, X, size=600, show_ylabels=False)

plt.show()
