

from pythonProject.utils.k_means_non_spherical_client import NSKMeans
import config
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.PCA import PCA2_plot, safe_scores, visualize_tsne
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def plot_comparison_bars_with_labels(
    values1,
    values2,
    labels,
    ylabel,
    leloc='best',
    legend_labels=('Method 1', 'Method 2'),
    colors=('orange', 'blue'),
    title=None,
    figsize=(4.5, 4),
    dpi=300,
    fontsize=10,
    label_fontsize=9,
    rotation=0,
    decimal=2
):
    """
    绘制双组柱状图并在每个柱子上添加数值标签。

    参数：
        values1, values2: 两组对比数据（列表或 ndarray）
        labels: x轴标签（类别名）
        ylabel: y轴标签名称
        legend_labels: 图例名称，元组
        colors: 两组的颜色，元组
        title: 图表标题（可选）
        figsize: 图尺寸
        dpi: 分辨率
        fontsize: 轴字体大小
        label_fontsize: 柱子上数字字体大小
        rotation: x轴标签旋转角度
        decimal: 小数精度
    """
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bars1 = ax.bar(x, values1, width=0.2, label=legend_labels[0],edgecolor='black', color=colors[0])
    bars2 = ax.bar(x + 0.2, values2, width=0.2, label=legend_labels[1],edgecolor='black', color=colors[1])

    # 添加数值标签
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2-0.08, bar.get_height(),
                f'{bar.get_height():.{decimal}f}', ha='center', va='bottom', fontsize=label_fontsize)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2+0.08, bar.get_height(),
                f'{bar.get_height():.{decimal}f}', ha='center', va='bottom', fontsize=label_fontsize)

    ax.set_position([0.15, 0.12, 0.8, 0.8])
    ax.set_xticks(x + 0.1)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    if title:
        ax.set_title(title, fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize, loc=leloc)
    # plt.tight_layout()
    plt.show()
def assign_noise_points_to_clusters(X, labels):
    """
    将 DBSCAN 中的噪声点（label = -1）分配到最近的已有簇中。

    参数:
        X: ndarray of shape (n_samples, n_features)
           原始数据特征。
        labels: ndarray of shape (n_samples,)
           DBSCAN 聚类结果标签，噪声点的标签为 -1。

    返回:
        labels_reassigned: ndarray
           所有点都被分配到簇后的新标签。
    """
    # 找出核心点和噪声点
    core_mask = labels != -1
    noise_mask = labels == -1

    # 如果没有噪声点，直接返回
    if not np.any(noise_mask):
        return labels.copy()

    # 提取核心点及其标签
    X_core = X[core_mask]
    y_core = labels[core_mask]
    X_noise = X[noise_mask]

    # 使用最近邻把噪声点分配到最近核心点的簇
    nn = NearestNeighbors(n_neighbors=1).fit(X_core)
    distances, indices = nn.kneighbors(X_noise)
    reassigned_labels = y_core[indices.flatten()]

    # 合并标签
    labels_reassigned = labels.copy()
    labels_reassigned[noise_mask] = reassigned_labels

    return labels_reassigned


np.random.seed(config.seed)
split_mod = 4
client_number = config.num_clients
server_number = config.num_servers
KLD = np.load(f'npydata/KLD_client{client_number}.npy')
JSD = np.load(f'npydata/JSD_client{client_number}.npy')
q_acc = np.load(f'npydata/Q_acc_client{client_number}.npy')
q_loss = np.load(f'npydata/Q_loss_client{client_number}.npy')
data_num = np.load(f'npydata/client_num_client{client_number}.npy')
print('q_acc', q_acc, 'q_loss', q_loss, 'client_data_num', data_num)

# def cluster_data():
#     num_clients = config.num_clients
f = np.random.uniform(1, 2, size=client_number)
#     q = q_loss
client_data = {
    'location_x': np.random.rand(client_number),  # 客户端的X坐标
    'location_x_stan': StandardScaler().fit_transform(np.random.rand(client_number).reshape(-1, 1)).ravel(),
    'location_y': np.random.rand(client_number),  # 客户端的Y坐标
    'location_y_stan': StandardScaler().fit_transform(np.random.rand(client_number).reshape(-1, 1)).ravel(),
    'data_quality': 1 / q_loss,  # 数据质量，0到1之间的浮点数
    'data_quality_stan': StandardScaler().fit_transform(q_loss.reshape(-1, 1)).ravel(),
    'data_num': data_num,  # 数据量，100到1000之间的整数
    'data_num_stan': StandardScaler().fit_transform(data_num.reshape(-1, 1)).ravel(),
    'data_KLD': KLD,  # 数据和测试集的KL度量 0到1之间的浮点数
    'data_JSD': JSD,  # 数据和测试集的JS度量 0到1之间的浮点数
    'data_KLD_stan': StandardScaler().fit_transform(KLD.reshape(-1, 1)).ravel(),
    'computing_power': f,  # 计算能力，1到2之间的整数
    'computing_power_stan': StandardScaler().fit_transform(f.reshape(-1, 1)).ravel(),
    'bandwidth_requirement': np.random.randint(5, 20, size=client_number),  # 带宽需求，5到20之间的整数
    # 'rayleigh': np.random.rayleigh(1, client_number)  # 雷利分布
}

# SPLIT SPOT===================================================================================================

clients_df = pd.DataFrame(client_data)
df_split = clients_df[['computing_power_stan', 'data_num_stan']]
new_df_split = np.array(df_split)
n_clusters = split_mod
myk_means = NSKMeans(n_clusters,random_state=config.seed)
myk_means.fit(new_df_split,split_mod, False)
labels = myk_means.labels_
centroids = myk_means.cluster_centers_

PCA2_plot(new_df_split, labels, ['Client CPU', 'Data Quantity'])


dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(df_split)
dbscan_label = dbscan.labels_
new_labels_sp = assign_noise_points_to_clusters(new_df_split, dbscan_label)



silhouette_avg_dbscan_sp, calinski_harabasz_dbscan_sp, davies_bouldin_dbscan_sp = safe_scores(df_split, new_labels_sp)

print("Silhouette Score dbscan:", silhouette_avg_dbscan_sp)
print("Calinski-Harabasz Score dbscan:", calinski_harabasz_dbscan_sp)
print("Davies-Bouldin Score dbscan:", davies_bouldin_dbscan_sp)

silhouette_avg_kmeans_sp, calinski_harabasz_kmeans_sp, davies_bouldin_kmeans_sp = safe_scores(df_split, labels)

print("Silhouette Score:", silhouette_avg_kmeans_sp)
print("Calinski-Harabasz Score:", calinski_harabasz_kmeans_sp)
print("Davies-Bouldin Score:", davies_bouldin_kmeans_sp)


df_INF = pd.DataFrame(clients_df, columns=['data_num_stan', 'computing_power_stan','location_x_stan', 'location_y_stan'])
np_df_INF = np.array(df_INF)
n_clusters= server_number
myk_means = NSKMeans(n_clusters,random_state=config.seed)
myk_means.fit(np_df_INF,server_number,False)
labels = myk_means.labels_
centroids = myk_means.cluster_centers_
# PCA2_anasys(df_INF,['Data\nVolume','computing_power_stan', 'Client\nx-position', 'Client\ny-position'])
# PCA2_plot(np_df_INF, labels, ['Data\nVolume','computing_power_stan', 'Client\nx-position', 'Client\ny-position'])
X_tsne, score = visualize_tsne(np_df_INF, labels=labels, perplexity=20, random_state=2, k=5)

dbscan = DBSCAN(eps=0.6, min_samples=2)
dbscan.fit(np_df_INF)
dbscan_label = dbscan.labels_
new_labels_inf = assign_noise_points_to_clusters(np_df_INF, dbscan_label)
silhouette_avg_dbscan_inf, calinski_harabasz_dbscan_inf, davies_bouldin_dbscan_inf = safe_scores(np_df_INF, new_labels_inf)

print("Silhouette Score dbscan:", silhouette_avg_dbscan_inf)
print("Calinski-Harabasz Score dbscan:", calinski_harabasz_dbscan_inf)
print("Davies-Bouldin Score dbscan:", davies_bouldin_dbscan_inf)

silhouette_avg_kmeans_inf, calinski_harabasz_kmeans_inf, davies_bouldin_kmeans_inf = safe_scores(np_df_INF, labels)

print("Silhouette Score:", silhouette_avg_kmeans_inf)
print("Calinski-Harabasz Score:", calinski_harabasz_kmeans_inf)
print("Davies-Bouldin Score:", davies_bouldin_kmeans_inf)



df_CHO = pd.DataFrame(clients_df, columns=['data_num_stan', 'computing_power_stan','location_x_stan', 'location_y_stan', 'data_quality_stan'])
np_df_CHO = np.array(df_CHO)
n_clusters = 3
myk_means = NSKMeans(n_clusters, random_state=config.seed)
myk_means.fit(np_df_CHO, 3, False)
labels = myk_means.labels_
centroids = myk_means.cluster_centers_
# PCA2_anasys(np_df_CHO,['data_num_stan', 'computing_power_stan','location_x_stan', 'location_y_stan', 'data_quality_stan'])
# PCA2_plot(np_df_CHO, labels,['data_num_stan', 'computing_power_stan','location_x_stan', 'location_y_stan', 'data_quality_stan'])
X_tsne, score = visualize_tsne(np_df_CHO, labels=labels, perplexity=20, random_state=2, k=5)

dbscan = DBSCAN(eps=0.8, min_samples=2)
dbscan.fit(df_CHO)
dbscan_label = dbscan.labels_
new_labels_cho = assign_noise_points_to_clusters(df_CHO, dbscan_label)

# plot_kclusters(np_df_CHO,labels,centroids)
# plot_dbclusters(np_df_CHO,dbscan_label)
# plot_dbclusters(np_df_CHO,new_labels_cho)

silhouette_avg_dbscan_CHO, calinski_harabasz_dbscan_CHO, davies_bouldin_dbscan_CHO = safe_scores(df_CHO, new_labels_cho)
print("Silhouette Score dbscan:", silhouette_avg_dbscan_CHO)
print("Calinski-Harabasz Score dbscan:", calinski_harabasz_dbscan_CHO)
print("Davies-Bouldin Score dbscan:", davies_bouldin_dbscan_CHO)

silhouette_avg_kmeans_CHO, calinski_harabasz_kmeans_CHO, davies_bouldin_kmeans_CHO = safe_scores(df_CHO, labels)
print("Silhouette Score:", silhouette_avg_kmeans_CHO)
print("Calinski-Harabasz Score:", calinski_harabasz_kmeans_CHO)
print("Davies-Bouldin Score:", davies_bouldin_kmeans_CHO)



df_AGG = pd.DataFrame(clients_df, columns=['location_x_stan','location_y_stan'])
np_df_AGG = np.array(df_AGG)

myk_means = NSKMeans(n_clusters=server_number,random_state=config.seed)
myk_means.fit(np_df_AGG,server_number,False)
n_clusters = server_number
labels = myk_means.labels_
centroids = myk_means.cluster_centers_
# PCA2_anasys(df_AGG,['Client\nx-position', 'Client\ny-position', 'Data\nQuality'])
PCA2_plot(np_df_AGG, labels, ['Location x', 'Location y'])

dbscan = DBSCAN(eps=0.5, min_samples=2)
dbscan.fit(df_AGG)
dbscan_label = dbscan.labels_
new_labels_agg = assign_noise_points_to_clusters(df_AGG, dbscan_label)

silhouette_avg_dbscan_AGG, calinski_harabasz_dbscan_AGG, davies_bouldin_dbscan_AGG = safe_scores(df_AGG, new_labels_agg)

print("Silhouette Score dbscan:", silhouette_avg_dbscan_AGG)
print("Calinski-Harabasz Score dbscan:", calinski_harabasz_dbscan_AGG)
print("Davies-Bouldin Score dbscan:", davies_bouldin_dbscan_AGG)

silhouette_avg_kmeans_AGG, calinski_harabasz_kmeans_AGG, davies_bouldin_kmeans_AGG = safe_scores(df_AGG, labels)

print("Silhouette Score:", silhouette_avg_kmeans_AGG)
print("Calinski-Harabasz Score:", calinski_harabasz_kmeans_AGG)
print("Davies-Bouldin Score:", davies_bouldin_kmeans_AGG)


silhouette_avg_dbscan_list = [silhouette_avg_dbscan_sp, silhouette_avg_dbscan_inf, silhouette_avg_dbscan_CHO, silhouette_avg_dbscan_AGG]
silhouette_avg_kmeans_list = [silhouette_avg_kmeans_sp, silhouette_avg_kmeans_inf, silhouette_avg_kmeans_CHO, silhouette_avg_kmeans_AGG]
calinski_harabasz_dbscan_list = [calinski_harabasz_dbscan_sp, calinski_harabasz_dbscan_inf, calinski_harabasz_dbscan_CHO, calinski_harabasz_dbscan_AGG]
calinski_harabasz_kmeans_list = [calinski_harabasz_kmeans_sp, calinski_harabasz_kmeans_inf, calinski_harabasz_kmeans_CHO, calinski_harabasz_kmeans_AGG]
davies_bouldin_dbscan_list = [davies_bouldin_dbscan_sp, davies_bouldin_dbscan_inf, davies_bouldin_dbscan_CHO, davies_bouldin_dbscan_AGG]
davies_bouldin_kmeans_list = [davies_bouldin_kmeans_sp, davies_bouldin_kmeans_inf, davies_bouldin_kmeans_CHO, davies_bouldin_kmeans_AGG]
#
#
# plt.figure(figsize=(4.5, 4), dpi=300)
# plt.bar(np.arange(len(silhouette_avg_dbscan_list)), silhouette_avg_dbscan_list, width=0.2, label='DBSCAN', color='orange')
# plt.bar(np.arange(len(silhouette_avg_kmeans_list)) + 0.2, silhouette_avg_kmeans_list, width=0.2, label='KMeans', color='blue')
# plt.xticks(np.arange(len(silhouette_avg_kmeans_list)) + 0.1, ['Split Spot', 'Info Spot', 'Choice Spot', 'Aggre Spot'])
# plt.ylabel('Silhouette Score')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(4.5, 4), dpi=300)
# plt.bar(np.arange(len(calinski_harabasz_dbscan_list)), calinski_harabasz_dbscan_list, width=0.2, label='DBSCAN', color='orange')
# plt.bar(np.arange(len(calinski_harabasz_kmeans_list)) + 0.2, calinski_harabasz_kmeans_list, width=0.2, label='KMeans', color='blue')
# plt.xticks(np.arange(len(calinski_harabasz_kmeans_list)) + 0.1, ['Split Spot', 'Info Spot', 'Choice Spot', 'Aggre Spot'])
# plt.ylabel('Calinski-Harabasz Score')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(4.5, 4), dpi=300)
# plt.bar(np.arange(len(davies_bouldin_dbscan_list)), davies_bouldin_dbscan_list, width=0.2, label='DBSCAN', color='orange')
# plt.bar(np.arange(len(davies_bouldin_kmeans_list)) + 0.2, davies_bouldin_kmeans_list, width=0.2, label='KMeans', color='blue')
# plt.xticks(np.arange(len(davies_bouldin_kmeans_list)) + 0.1, ['Split Spot', 'Info Spot', 'Choice Spot', 'Aggre Spot'])
# plt.ylabel('Davies-Bouldin Score')
# plt.legend()
# plt.tight_layout()
# plt.show()

oringecolors = '#7880b4'
bluecolors = '#b5d6bb'

plot_comparison_bars_with_labels(
    values1=calinski_harabasz_dbscan_list,
    values2=calinski_harabasz_kmeans_list,
    labels=['Split\nSpot', 'Inference', 'Choose\nClient', 'Aggregation'],
    ylabel='Calinski-Harabasz Score',
    colors=(oringecolors, bluecolors),
    legend_labels=('DBSCAN', 'HMKC'),
    leloc='upper left',
    # title='Calinski-Harabasz Score Comparison'
)



plot_comparison_bars_with_labels(
    values1=silhouette_avg_dbscan_list,
    values2=silhouette_avg_kmeans_list,
    labels=['Split\nSpot', 'Inference', 'Choose\nClient', 'Aggregation'],
    ylabel='Silhouette Score',
    colors=(oringecolors, bluecolors),
    legend_labels=('DBSCAN', 'HMKC'),
    leloc='upper right',
    # title='Silhouette Score Comparison'
)

plot_comparison_bars_with_labels(
    values1=davies_bouldin_dbscan_list,
    values2=davies_bouldin_kmeans_list,
    labels=['Split\nSpot', 'Inference', 'Choose\nClient', 'Aggregation'],
    ylabel='Davies-Bouldin Score',
    colors=(oringecolors, bluecolors),
    legend_labels=('DBSCAN', 'HMKC'),
    leloc='upper right',
    # title='Davies-Bouldin Score Comparison'
)