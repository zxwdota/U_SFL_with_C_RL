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

def K_means(q, data_num, KLD, JSD):
    # 生成客户端数据
    num_clients = config.num_clients
    np.random.seed(config.seed)
    f = np.random.uniform(1, 2, size=num_clients)
    client_data = {
        'location_x': np.random.rand(num_clients),  # 客户端的X坐标
        'location_y': np.random.rand(num_clients),  # 客户端的Y坐标
        'data_quality': 1 / q,  # 数据质量，0到1之间的浮点数
        'data_quality_stan': StandardScaler().fit_transform(q.reshape(-1, 1)).ravel(),
        'data_quantity': data_num,  # 数据量，100到1000之间的整数
        'data_quantity_stan': StandardScaler().fit_transform(data_num.reshape(-1, 1)).ravel(),
        'data_KLD': KLD,  # 数据和测试集的KL度量 0到1之间的浮点数
        'data_JSD': JSD,  # 数据和测试集的JS度量 0到1之间的浮点数
        'data_KLD_stan': StandardScaler().fit_transform(KLD.reshape(-1, 1)).ravel(),
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

    K_means_fxy = KMeans(n_clusters=config.num_servers, random_state=config.seed)

    K_means_fxy.fit(clients_df[['computing_power_stan', 'location_x', 'location_y', 'data_quantity_stan']])

    clients_df['fxy_cluster'] = K_means_fxy.labels_

    #######################                 PCA________________________####################
    features = clients_df[['computing_power_stan', 'location_x', 'location_y', 'data_quantity_stan']]

    # 使用 PCA 将数据降到 2 维
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # 创建一个新的 DataFrame，包含降维后的数据和聚类标签
    pca_df = pd.DataFrame(features_2d, columns=['PCA1', 'PCA2'])
    pca_df['cluster'] = clients_df['fxy_cluster']

    # 绘制聚类结果
    plt.figure(figsize=(5, 4.5),dpi=400)

    # 不同的簇使用不同的颜色进行绘制
    for cluster in sorted(pca_df['cluster'].unique()):
        cluster_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', s=200, alpha=0.6)
    plt.xlabel('PCA Component 1',fontsize=16)
    plt.ylabel('PCA Component 2',fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()

    Kmeans_q = KMeans(n_clusters=3, random_state=config.seed)

    Kmeans_q.fit(clients_df[['data_quality_stan', 'data_quantity_stan']])
    clients_df['final_cluster'] = Kmeans_q.labels_

    fig = plt.figure(figsize=(5,4.5),dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(clients_df['data_quality'], clients_df['data_quantity'],
                         clients_df['data_JSD'], c=clients_df['final_cluster'], s=100, cmap='viridis', alpha=0.6)
    ax.set_xlabel("quality")
    ax.set_ylabel("quantity")
    ax.set_zlabel("JSD")

    ax.tick_params(axis='both', which='major')
    ax.invert_xaxis()
    # # 设置每个轴的格线数目相等

    num_ticks = 6  # 你可以根据需要调整刻度的数量

    # 设置网格背景颜色为白色
    ax.xaxis.pane.set_facecolor('white')  # 设置 X-Y 平面的背景颜色为白色
    ax.yaxis.pane.set_facecolor('white')  # 设置 Y-Z 平面的背景颜色为白色
    ax.zaxis.pane.set_facecolor('white')  # 设置 Z-X 平面的背景颜色为白色
    # 如果你想让背景透明，可以使用以下方式：
    ax.xaxis.pane.set_alpha(0.0)  # 设置 X-Y 平面的透明度
    ax.yaxis.pane.set_alpha(0.0)  # 设置 Y-Z 平面的透明度
    ax.zaxis.pane.set_alpha(0.0)  # 设置 Z-X 平面的透明度

    ax.zaxis.set_label_position('lower')
    ax.zaxis.set_ticks_position('lower')

    x_min, x_max = 0, 0.5
    y_min, y_max = 0, 1000
    z_min, z_max = 0, 0.6

    # # 使用 np.linspace 生成合理数量的整数刻度
    ax.set_xticks(np.linspace(x_min, x_max, num_ticks))
    ax.set_yticks(np.linspace(y_min, y_max, num_ticks))
    ax.set_zticks(np.linspace(z_min, z_max, num_ticks))
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.5, 1]))  # 第二个参数为纵轴缩放比例
    ax.set_proj_type('ortho')

    unique_clusters = np.unique(clients_df['final_cluster'])
    colors = [plt.cm.viridis(i / (len(unique_clusters) - 1)) for i in unique_clusters]
    patches = [mpatches.Patch(color=colors[i], label=f'Cluster {unique_clusters[i]}') for i in
               range(len(unique_clusters))]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(0.95, 0.9))

    plt.tight_layout()
    ax.view_init(elev=10, azim=45)
    fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.savefig("3d_kmeans_plot.png", bbox_inches='tight', dpi=400)
    plt.show()

    # # 输出生成的初始数据
    # print("Clients Data:")
    # print(clients_df.head())
    #
    # print("\nServers Data:")
    # print(servers_df.head())
    #
    # plt.figure(dpi=200)
    # scatter1 = plt.scatter(clients_df['data_quantity'], clients_df['data_quality'], c=clients_df['final_cluster'],
    #                        s=200, cmap='viridis', alpha=0.6)
    # plt.title("Selected Clients Clustering (K=3)")
    # plt.xlabel("quantity")
    # plt.ylabel("quality")
    # plt.colorbar(scatter1, label='Cluster Label')
    # plt.show()
    plt.figure(figsize=(5,4.5),dpi=400)

    for cluster in sorted(clients_df['final_cluster'].unique()):
        cluster_data = clients_df[clients_df['final_cluster'] == cluster]
        plt.scatter(cluster_data['data_quantity'], cluster_data['data_quality'], label=f'Cluster {cluster}', s=200,
                    alpha=0.6)
    plt.xlabel("quantity",fontsize=16)
    plt.ylabel("quality",fontsize=16)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.show()

    #############################################p3d##############################p3d##############################p3d##############################p3d####################

    clustered_dataframes = [clients_df[clients_df['final_cluster'] == i] for i in range(3)]
    cluster_list = [clustered_dataframes[i].copy() for i in range(3)]
    avg = np.array([cluster_list[0]['data_quality'].mean(), cluster_list[1]['data_quality'].mean(),
                    cluster_list[2]['data_quality'].mean()])
    sort = np.argsort(avg)[::-1]
    if len(cluster_list[sort[0]]) < config.num_servers:
        choose_cluster = pd.concat([cluster_list[sort[0]], cluster_list[sort[1]]],axis=0)
        np.save('sort',sort[:2])
    else:
        choose_cluster = cluster_list[sort[0]]
        np.save('sort',[sort[0]])

    K_means_xy = KMeans(n_clusters=config.num_servers, random_state=config.seed)

    K_means_xy.fit(choose_cluster[['location_x', 'location_y']])

    choose_cluster['xy_cluster'] = K_means_xy.labels_

    # features_AGG = choose_cluster[['location_x', 'location_y', 'data_quantity_stan']]
    # ################################################################################################
    # # 使用 PCA 将数据降到 2 维
    # pca = PCA(n_components=2)
    # features_AGG_2d = pca.fit_transform(features_AGG)
    #
    # # 创建一个新的 DataFrame，包含降维后的数据和聚类标签
    # pca_df = pd.DataFrame(features_AGG_2d, columns=['PCA1', 'PCA2'])
    # pca_df['cluster'] = choose_cluster['xy_cluster'].values
    #
    # # 绘制聚类结果
    # plt.figure(dpi=200)
    #
    # # 不同的簇使用不同的颜色进行绘制
    # for cluster in pca_df['cluster'].unique():
    #     cluster_data = pca_df[pca_df['cluster'] == cluster]
    #     plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster}', s=200, alpha=0.6)
    #
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.title('K-means Clustering Visualization with PCA')
    # plt.legend()
    # plt.show()

    #   #########p3d##############################p3d##############################p3d##############################p3d####################


    plt.figure(figsize=(5,4.5),dpi=400)
    for cluster in sorted(choose_cluster['xy_cluster'].unique()):
        cluster_data = choose_cluster[choose_cluster['xy_cluster'] == cluster]
        plt.scatter(cluster_data['location_x'], cluster_data['location_y'],label=f'Cluster  {cluster}', s=200, alpha=0.6)
    plt.xlabel("x",fontsize=16)
    plt.ylabel("y",fontsize=16)
    plt.legend(fontsize=10,loc='best')
    plt.tight_layout()
    plt.show()

    with open('KLdata.pkl', 'wb') as f:
        pickle.dump(K_means_fxy, f)
        pickle.dump(Kmeans_q, f)
        pickle.dump(K_means_xy, f)
        pickle.dump(client_data, f)
        pickle.dump(server_data, f)
    return K_means_fxy, Kmeans_q, K_means_xy, client_data, server_data


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
                    xticklabels=list(['x1', 'x2', 'x3', 'x4']),
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


X = clients_df[['computing_power_stan', 'location_x', 'location_y', 'data_quantity_stan']]

PCA2_anasys(X)

dbscan = DBSCAN(eps=0.7,min_samples=2)
dbscan.fit(X)
y_label = dbscan.labels_

PCA2_plot(X, y_label)

silhouette_avg_dbscan, calinski_harabasz_dbscan, davies_bouldin_dbscan = safe_scores(X, y_label)
print("Silhouette Score:", silhouette_avg_dbscan)
print("Calinski-Harabasz Score:", calinski_harabasz_dbscan)
print("Davies-Bouldin Score:", davies_bouldin_dbscan)



kmeans = KMeans(n_clusters=config.num_servers, random_state=0).fit(X)
y_kmeans = kmeans.labels_

silhouette_avg_kmeans, calinski_harabasz_kmeans, davies_bouldin_kmeans = safe_scores(X, y_kmeans)
PCA2_plot(X, y_kmeans)
print("Silhouette Score:", silhouette_avg_kmeans)
print("Calinski-Harabasz Score:", calinski_harabasz_kmeans)
print("Davies-Bouldin Score:", davies_bouldin_kmeans)