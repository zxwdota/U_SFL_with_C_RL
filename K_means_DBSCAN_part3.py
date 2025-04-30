import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from splitdata_to_client_and_get_q import read_data_non_iid, get_quality
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score

from mydata_util import random_get_dict
import config
import torch
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
num_clients = config.num_clients



client_number = config.num_clients
KLD = np.load(f'npydata/KLD_client{client_number}.npy')
JSD = np.load(f'npydata/JSD_client{client_number}.npy')
q_acc = np.load(f'npydata/Q_acc_client{client_number}.npy')
q_loss = np.load(f'npydata/Q_loss_client{client_number}.npy')
client_data_num = np.load(f'npydata/client_num_client{client_number}.npy')
print('q_acc', q_acc, 'q_loss', q_loss, 'client_data_num', client_data_num)
# K_means(q_loss, client_data_num, KLD, JSD)

# DBSCAN

f = np.random.uniform(1, 2, size=num_clients)

client_data = {
        'location_x': np.random.rand(num_clients),  # 客户端的X坐标
        'location_y': np.random.rand(num_clients),  # 客户端的Y坐标
        'data_quality': 1 / q_loss,  # 数据质量，0到1之间的浮点数
        'data_quality_stan': StandardScaler().fit_transform(q_loss.reshape(-1, 1)).ravel(),
        'data_quantity': client_data_num,  # 数据量，100到1000之间的整数
        'data_quantity_stan': StandardScaler().fit_transform(client_data_num.reshape(-1, 1)).ravel(),
        'computing_power': f,  # 计算能力，1到2之间的整数
        'computing_power_stan': StandardScaler().fit_transform(f.reshape(-1, 1)).ravel(),
        'bandwidth_requirement': np.random.randint(5, 20, size=num_clients),  # 带宽需求，5到20之间的整数
        'rayleigh': np.random.rayleigh(1, num_clients)  # 雷利分布
    }
clients_df = pd.DataFrame(client_data)

# 生成服务器数据
num_servers = config.num_servers
server_data = {
    'location_x': np.random.rand(num_servers),  # 服务器的X坐标
    'location_y': np.random.rand(num_servers),  # 服务器的Y坐标
    'computing_power': np.random.uniform(2, 4, size=num_servers),  # 计算能力，
    'bandwidth': np.random.randint(30, 50, size=num_servers),  # 最大带宽，
}
servers_df = pd.DataFrame(server_data)

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np



def PCA2_anasys(features):
    pca = PCA()
    features_2d = pca.fit_transform(features)
    PC_components = np.arange(pca.n_components_) + 1
    _ = sns.set(style='whitegrid', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    _ = sns.barplot(x=PC_components, y=pca.explained_variance_ratio_, color='b')
    _ = sns.lineplot(x=PC_components - 1, y=np.cumsum(pca.explained_variance_ratio_), color='black', linestyle='-',
                     linewidth=2, marker='o', markersize=8)

    plt.title('Scree Plot')
    plt.xlabel('N-th Principal Component')
    plt.ylabel('Variance Explained')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

    _ = sns.heatmap(pca.components_ ** 2,
                    yticklabels=["PCA" + str(x) for x in range(1, pca.n_components_ + 1)],
                    xticklabels=list(label_xlist),
                    annot=True,
                    fmt='.2f',
                    square=True,
                    linewidths=0.05,
                    cbar_kws={"orientation": "horizontal"})
    plt.show()

    # 以上分析表面components_的值为2即可：

def PCA2_plot(features, y_label):
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # 创建 PCA 降维 + 标签 DataFrame
    pca_df = pd.DataFrame(features_2d, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = y_label

    plt.figure(figsize=(5, 4.5), dpi=400)

    # 先处理非 -1 的标签（正常聚类）
    clusters = sorted([c for c in pca_df['cluster'].unique() if c != -1])
    for cluster in clusters:
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', s=200, alpha=0.6)

    # 最后绘制噪声点（label = -1），用灰色
    if -1 in pca_df['cluster'].unique():
        noise_data = pca_df[pca_df['cluster'] == -1]
        plt.scatter(noise_data['PCA1'], noise_data['PCA2'], label='Noise', s=200, alpha=0.4, c='gray')

    plt.xlabel('PCA 1', fontsize=16)
    plt.ylabel('PCA 2', fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
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

with open('response_data/dbscan_cluster_0.pkl', 'rb') as f:
    dbscan_cluster_0 = pickle.load(f)
with open('response_data/dbscan_cluster_1.pkl', 'rb') as f:
    dbscan_cluster_1 = pickle.load(f)
with open('response_data/dbscan_cluster_2.pkl', 'rb') as f:
    dbscan_cluster_2 = pickle.load(f)
with open('response_data/dbscan_cluster_noise.pkl', 'rb') as f:
    dbscan_cluster_noise = pickle.load(f)

q_0 = np.average(dbscan_cluster_0['data_quality_stan'])
q_1 = np.average(dbscan_cluster_1['data_quality_stan'])
q_2 = np.average(dbscan_cluster_2['data_quality_stan'])
q_n = np.average(dbscan_cluster_noise['data_quality_stan'])
q_list = [q_0, q_1, q_2, q_n]
a = np.argmax(q_list)


clients_choosed_df = dbscan_cluster_1
label_list = ['location_x', 'location_y', 'data_quantity_stan']
label_xlist = ['x1','x2','x3']
X = clients_choosed_df[label_list]

PCA2_anasys(X)

dbscan = DBSCAN(eps=0.3,min_samples=2)
dbscan.fit(X)
y_label = dbscan.labels_

PCA2_plot(X, y_label)

silhouette_avg_dbscan, calinski_harabasz_dbscan, davies_bouldin_dbscan = safe_scores(X, y_label)
print("Silhouette Score:", silhouette_avg_dbscan)
print("Calinski-Harabasz Score:", calinski_harabasz_dbscan)
print("Davies-Bouldin Score:", davies_bouldin_dbscan)






with open('response_data/kmeans_cluster_0.pkl', 'rb') as f:
    kmeans_cluster_0 = pickle.load(f)
with open('response_data/kmeans_cluster_1.pkl', 'rb') as f:
    kmeans_cluster_1 = pickle.load(f)
with open('response_data/kmeans_cluster_2.pkl', 'rb') as f:
    kmeans_cluster_2 = pickle.load(f)


q_0 = np.average(kmeans_cluster_0['data_quality_stan'])
q_1 = np.average(kmeans_cluster_1['data_quality_stan'])
q_2 = np.average(kmeans_cluster_2['data_quality_stan'])
q_list = [q_0, q_1, q_2]
a = np.argmax(q_list)


clients_choosed_df = dbscan_cluster_1
label_list = ['location_x', 'location_y', 'data_quantity_stan']
label_xlist = ['x1','x2','x3']
X = clients_choosed_df[label_list]


kmeans = KMeans(n_clusters=config.num_servers, random_state=config.seed).fit(X)
y_kmeans = kmeans.labels_

silhouette_avg_kmeans, calinski_harabasz_kmeans, davies_bouldin_kmeans = safe_scores(X, y_kmeans)
PCA2_plot(X, y_kmeans)
print("Silhouette Score:", silhouette_avg_kmeans)
print("Calinski-Harabasz Score:", calinski_harabasz_kmeans)
print("Davies-Bouldin Score:", davies_bouldin_kmeans)