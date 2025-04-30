import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score



def PCA2_anasys(features,F):
    pca = PCA()
    features_2d = pca.fit_transform(features)
    PC_components = np.arange(pca.n_components_) + 1
    _ = sns.set(style='whitegrid', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(4.5,4), dpi=300)
    _ = sns.barplot(x=PC_components, y=pca.explained_variance_ratio_, color='b')
    _ = sns.lineplot(x=PC_components - 1, y=np.cumsum(pca.explained_variance_ratio_), color='black', linestyle='-',
                     linewidth=2, marker='o', markersize=8)
    for i, val in enumerate(np.cumsum(pca.explained_variance_ratio_)):
        ax.text(i, val + 0.03, f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    ax.set_position([0.17, 0.12, 0.7, 0.55])
    # plt.title('Scree Plot')
    plt.xlabel('N-th Principal Component')
    plt.ylabel('Variance Explained')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(figsize=(4.5,4), dpi=400)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    _ = sns.heatmap(pca.components_ ** 2,
                    yticklabels=["PCA" + str(x) for x in range(1, pca.n_components_ + 1)],
                    xticklabels=list(F),
                    annot=True,
                    fmt='.2f',
                    square=True,
                    linewidths=0.05,
                    # cbar_kws={"orientation": "horizontal","shrink": 0.5})
                    cbar_kws = {"shrink": 0.85},
                    ax = ax)
    ax.set_position([0, 0.12, 0.8, 0.8])

    cbar = fig.axes[1]
    cbar.set_position([0.85, 0.12, 0.5, 0.8])
    plt.show()

    # 以上分析表面components_的值为2即可：

def PCA2_plot(features, y_label, F):

    if len(F)>2:
        F = ['PCA1', 'PCA2']
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # 创建 PCA 降维 + 标签 DataFrame
        pca_df = pd.DataFrame(features_2d, columns=F)
    else:
        pca_df = pd.DataFrame(features, columns=F)
    pca_df['cluster'] = y_label

    plt.figure(figsize=(4.5, 4), dpi=300)
    plt.axes([0.15, 0.12, 0.8, 0.8])

    # 先处理非 -1 的标签（正常聚类）
    clusters = sorted([c for c in pca_df['cluster'].unique() if c != -1])
    for cluster in clusters:
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(cluster_data[F[0]], cluster_data[F[1]], label=f'Cluster {cluster}', s=200, alpha=0.6)

    # 最后绘制噪声点（label = -1），用灰色
    if -1 in pca_df['cluster'].unique():
        noise_data = pca_df[pca_df['cluster'] == -1]
        plt.scatter(noise_data[F[0]], noise_data[F[1]], label='Noise', s=200, alpha=0.4, c='gray')

    plt.xlabel(F[0])
    plt.ylabel(F[1])
    plt.legend(fontsize=10, loc='best')
    plt.show()


def safe_scores(X, labels):
    # 有效簇的数量（去除 label=-1）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters <= 1:
        return None, None, None
    else:
        mask = labels != -1
        return (
            silhouette_score(X[mask], labels[mask]),           # 自动忽略噪声
            calinski_harabasz_score(X[mask], labels[mask]),   # 手动去除噪声
            davies_bouldin_score(X[mask], labels[mask])       # 手动去除噪声
        )

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def knn_preservation_score(X_high, X_low, k=5):
    """
    计算高维数据 X_high 和其降维结果 X_low 在 k 近邻结构上的一致性。
    参数：
        X_high: ndarray, shape (n_samples, n_features)
        X_low: ndarray, shape (n_samples, n_components)
        k: int, 最近邻个数
    返回：
        preservation_score: float, kNN 保留度得分
    """
    knn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    knn_low = NearestNeighbors(n_neighbors=k).fit(X_low)
    _, idx_high = knn_high.kneighbors(X_high)
    _, idx_low = knn_low.kneighbors(X_low)
    ratios = [len(set(h).intersection(set(l))) / k for h, l in zip(idx_high, idx_low)]
    return sum(ratios) / len(ratios)


def visualize_tsne(
    X,
    labels=None,
    perplexity=30,
    random_state=0,
    k=5
):
    """
    将高维数据 X 使用 t-SNE 降到 2 维并按簇绘制，同时计算 kNN 结构保留得分。

    返回:
        X_tsne, score
    """
    # 1) t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)


    # 3) 计算 kNN 保留度
    score = knn_preservation_score(X, X_tsne, k=k)
    print(f'kNN Preservation Score: {score:.2f}')

    # 2) 可视化
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)

    if labels is None:
        ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            s=200, edgecolor='k', alpha=0.6, color='dodgerblue'
        )
    else:
        # 按簇绘制
        clusters = np.unique(labels)
        cmap = plt.get_cmap('tab10', len(clusters))
        for idx, cluster in enumerate(clusters):
            pts = X_tsne[labels == cluster]
            if cluster == -1:
                col = 'gray'
                lbl = 'Noise'
            else:
                col = cmap(idx)
                lbl = f'Cluster {cluster}'
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=200, edgecolor='k', alpha=0.6, label=lbl
            )
        ax.legend(fontsize=10, loc='best')
    ax.set_position([0.15, 0.12, 0.8, 0.8])
    ax.set_title(f't-SNE with kNN score {score:.2f}')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    plt.show()


    return X_tsne, score

from sklearn.neighbors import NearestNeighbors

def knn_preservation_score(X_high, X_low, k=5):
    """
    计算高维和降维后最近邻的一致性
    """
    knn_high = NearestNeighbors(n_neighbors=k).fit(X_high)
    knn_low = NearestNeighbors(n_neighbors=k).fit(X_low)

    _, idx_high = knn_high.kneighbors(X_high)
    _, idx_low = knn_low.kneighbors(X_low)

    score = np.mean([len(set(h).intersection(set(l)))/k for h, l in zip(idx_high, idx_low)])
    return score