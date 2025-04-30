import numpy as np
import config
import pickle
import pandas as pd
from scipy.optimize import linear_sum_assignment
from utils.k_means_non_spherical_client import NSKMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


# np.random.seed(config.seed)
# split_mod = 4
# client_number = config.num_clients
# server_number = config.num_servers
# KLD = np.load(f'npydata/KLD_client{client_number}.npy')
# JSD = np.load(f'npydata/JSD_client{client_number}.npy')
# q_acc = np.load(f'npydata/Q_acc_client{client_number}.npy')
# q_loss = np.load(f'npydata/Q_loss_client{client_number}.npy')
# quality = 1/q_loss
# quality_stan = StandardScaler().fit_transform(quality.reshape(-1, 1)).ravel()
# data_num = np.load(f'npydata/client_num_client{client_number}.npy')
# data_num_stan = StandardScaler().fit_transform(data_num.reshape(-1, 1)).ravel()
# f = np.random.uniform(1, 2, size=client_number)
# f_stan = StandardScaler().fit_transform(f.reshape(-1, 1)).ravel()
# loc_x = np.random.rand(client_number)  # 原始 X 坐标
# loc_x_stan = StandardScaler().fit_transform(loc_x.reshape(-1, 1)).ravel()
# loc_y = np.random.rand(client_number)  # 原始 Y 坐标
# loc_y_stan = StandardScaler().fit_transform(loc_y.reshape(-1, 1)).ravel()
# client_data = {
#     'location_x': loc_x,  # 客户端的X坐标
#     'location_x_stan': loc_x_stan,
#     'location_y': loc_y,  # 客户端的Y坐标
#     'location_y_stan': loc_y_stan,
#     'data_quality': 1 / q_loss,  # 数据质量，0到1之间的浮点数
#     'data_quality_stan': StandardScaler().fit_transform(quality.reshape(-1, 1)).ravel(),
#     'data_num': data_num,  # 数据量，100到1000之间的整数
#     'data_num_stan': StandardScaler().fit_transform(data_num.reshape(-1, 1)).ravel(),
#     'computing_power': f,  # 计算能力，1到2之间的整数
#     'computing_power_stan': f_stan,
# }
# num_servers = config.num_servers
# server_data = {
#     'location_x': np.random.rand(num_servers),  # 服务器的X坐标
#     'location_y': np.random.rand(num_servers),  # 服务器的Y坐标
#     'computing_power': np.random.uniform(2, 4, size=num_servers),  # 计算能力，
#     'bandwidth': np.random.randint(30, 50, size=num_servers),  # 最大带宽，
# }
#
#






class Env:
    def __init__(self):
        # np.random.seed(config.seed)
        split_mod = 3
        client_number = config.num_clients
        q_loss = np.load(f'npydata/Q_loss_client{client_number}.npy')
        quality = 1 / q_loss
        data_num = np.load(f'npydata/client_num_client{client_number}.npy')
        f = np.random.uniform(1, 2, size=client_number)
        loc_x = np.random.rand(client_number) * 500  # 原始 X 坐标
        loc_y = np.random.rand(client_number) * 500  # 原始 Y 坐标
        self.time_per_pic = 0.005  # per pic
        self.sd = 2 / 64

        self.client_data = {

            'location_x': loc_x,  # 客户端的X坐标
            'location_y': loc_y,  # 客户端的Y坐标
            'data_quality': quality,  # 数据质量，0到1之间的浮点数
            'data_num': data_num,  # 数据量，100到1000之间的整数
            'computing_power': f,  # 计算能力，1到2之间的整数
        }
        num_servers = config.num_servers
        self.server_data = {
            'location_x': np.random.rand(num_servers) * 500,  # 服务器的X坐标
            'location_y': np.random.rand(num_servers) * 500,  # 服务器的Y坐标
            'computing_power': np.random.uniform(2, 4, size=num_servers),  # 计算能力，
            'bandwidth': np.random.randint(30, 50, size=num_servers),  # 最大带宽，
        }
        # with open('KLdata.pkl', 'rb') as f:
        #     self.K_means_fxy = pickle.load(f)
        #     self.K_means_q = pickle.load(f)
        #     self.K_means_xy = pickle.load(f)
        self.cluster()
        self.calculate_distances()
        self.s = self.init_inf_obs(False,False)

    def client_quit(self, client_data, num_drop=20):
        """
        从 client_data 中随机删除 num_drop 个客户端的数据
        """
        # 1) 随机选择多个客户端索引，注意不能重复！
        drop_indices = np.random.choice(config.num_clients, size=num_drop, replace=False)
        drop_indices = np.sort(drop_indices)[::-1]  # 先降序排序，方便后面删除不会出错
        print(f"删除客户端索引：{drop_indices}")

        # 2) 对每个字段删除对应位置的数据
        for key, arr in client_data.items():
            for idx in drop_indices:
                client_data[key] = np.delete(client_data[key], idx, axis=0)

        return client_data

    def server_quit(self,server_data):
        # 1) 随机选择一个服务器索引
        num_servers = config.num_servers
        drop_idx = np.random.randint(0, num_servers)
        print(f"删除服务器索引：{drop_idx}")

        # 2) 对每个字段删除对应位置的数据
        for key, arr in server_data.items():
            server_data[key] = np.delete(arr, drop_idx, axis=0)

    def cluster(self):
        split_mod = 3
        num_servers = len(self.server_data['location_x'])
        num_clients = len(self.client_data['location_x'])
        self.scaler_quality = StandardScaler().fit(self.client_data['data_quality'].reshape(-1, 1))
        quality_stan = self.scaler_quality.transform(self.client_data['data_quality'].reshape(-1, 1)).ravel()
        self.scaler_data_num = StandardScaler().fit(self.client_data['data_num'].reshape(-1, 1))
        data_num_stan = self.scaler_data_num.transform(self.client_data['data_num'].reshape(-1, 1)).ravel()
        self.scaler_f = StandardScaler().fit(self.client_data['computing_power'].reshape(-1, 1))
        f_stan = self.scaler_f.transform(self.client_data['computing_power'].reshape(-1, 1)).ravel()
        self.scaler_x = StandardScaler().fit(self.client_data['location_x'].reshape(-1, 1))
        loc_x_stan = self.scaler_x.transform(self.client_data['location_x'].reshape(-1, 1)).ravel()
        self.scaler_y = StandardScaler().fit(self.client_data['location_y'].reshape(-1, 1))
        loc_y_stan = self.scaler_y.transform(self.client_data['location_y'].reshape(-1, 1)).ravel()

        self.client_info_stan = {
            'location_x': loc_x_stan,  # 客户端的X坐标
            'location_y': loc_y_stan,  # 客户端的Y坐标
            'data_quality': quality_stan,  # 数据质量，0到1之间的浮点数
            'data_num': data_num_stan,  # 数据量，100到1000之间的整数
            'computing_power': f_stan,  # 计算能力，1到2之间的整数
        }

        # 聚类sp
        sp_df = pd.DataFrame(self.client_info_stan, columns=['computing_power', 'data_num'])
        np_sp = np.array(sp_df)
        self.SPLIT_cluster = NSKMeans(n_clusters=split_mod, random_state=config.seed)
        self.SPLIT_cluster.fit(np_sp, split_mod, True)
        SP_labels = self.SPLIT_cluster.labels_
        SP_centroids = self.SPLIT_cluster.cluster_centers_
        # 对每列单独反变换
        SP_centroids_f_orig = self.scaler_f.inverse_transform(SP_centroids[:, 0].reshape(-1, 1)).ravel()
        SP_centroids_data_num_orig = self.scaler_data_num.inverse_transform(SP_centroids[:, 1].reshape(-1, 1)).ravel()

        # 合并成原始坐标空间下的聚类中心
        SP_centroids_orig = np.column_stack([SP_centroids_f_orig, SP_centroids_data_num_orig])
        cluster_sizes = np.array([np.sum(SP_labels == k) for k in range(split_mod)])
        # print(SP_centroids_orig.shape)
        # print(cluster_sizes.shape)
        if SP_centroids_orig.shape[0] == 2:
            time = 0
        rho_heavy = SP_centroids_orig[:, 1] / SP_centroids_orig[:, 0] * cluster_sizes

        list = np.argsort(rho_heavy)[::-1]
        if config.split_freedom == True:
            sp_client_model_computing = np.array([0.07, 0.32, 0.55])
            sp_client_model_size = np.array([0.23, 0.28, 0.66])
        else:
            sp_client_model_computing = np.array([0.32, 0.32, 0.32])
            sp_client_model_size = np.array([0.28, 0.28, 0.28])
        sp_server_model_compputing = 1 - sp_client_model_computing
        sp_server_model_size = 1 - sp_client_model_size
        sp_result = np.vstack([list, sp_client_model_computing])
        self.client_side_data_num = []
        self.server_side_data_num = []
        self.client_side_model_size = []
        for idex, cluster in enumerate(SP_labels):
            for idex_sp, sp_i in enumerate(list):
                if sp_i == cluster:
                    client_side = sp_client_model_computing[idex_sp] * self.client_data['data_num'][idex]
                    server_side = sp_server_model_compputing[idex_sp] * self.client_data['data_num'][idex]
                    self.client_side_data_num.append(client_side)
                    self.server_side_data_num.append(server_side)
                    self.client_side_model_size.append(sp_client_model_size[idex_sp])

        client_side_data_num = np.array(self.client_side_data_num)
        server_side_data_num = np.array(self.server_side_data_num)
        client_side_model_size = np.array(self.client_side_model_size)
        self.scaler_client_side = StandardScaler().fit(client_side_data_num.reshape(-1, 1))
        client_side_data_num_stan = self.scaler_client_side.transform(client_side_data_num.reshape(-1, 1)).ravel()
        self.scaler_server_side = StandardScaler().fit(server_side_data_num.reshape(-1, 1))
        server_side_data_num_stan = self.scaler_server_side.transform(server_side_data_num.reshape(-1, 1)).ravel()
        self.scaler_client_side_model_size = StandardScaler().fit(client_side_model_size.reshape(-1, 1))
        client_side_model_size_stan = self.scaler_client_side_model_size.transform(
            client_side_model_size.reshape(-1, 1)).ravel()

        self.client_info_stan['client_side_data_num'] = client_side_data_num_stan
        self.client_info_stan['server_side_data_num'] = server_side_data_num_stan
        self.client_info_stan['client_side_model_size'] = client_side_model_size_stan

        # 聚类INF
        self.inf_df = pd.DataFrame(self.client_info_stan,
                                   columns=['client_side_data_num', 'server_side_data_num', 'computing_power',
                                            'location_x', 'location_y', 'client_side_model_size'])
        np_inf = np.array(self.inf_df)
        self.INF_cluster = NSKMeans(n_clusters=num_servers, random_state=config.seed)
        self.INF_cluster.fit(np_inf, num_servers, True)
        self.INF_labels = self.INF_cluster.labels_
        self.INF_cluster_result = []
        for l in range(num_servers):
            self.INF_cluster_result.append(np.where(self.INF_labels == l)[0])

        INF_centroids = self.INF_cluster.cluster_centers_
        # 对每列单独反变换
        INF_centroids_client_side_data_num_orig = self.scaler_client_side.inverse_transform(
            INF_centroids[:, 0].reshape(-1, 1)).ravel()
        INF_centroids_server_side_data_num_orig = self.scaler_server_side.inverse_transform(
            INF_centroids[:, 1].reshape(-1, 1)).ravel()
        INF_centroids_f_orig = self.scaler_f.inverse_transform(INF_centroids[:, 2].reshape(-1, 1)).ravel()
        INF_centroids_x_orig = self.scaler_x.inverse_transform(INF_centroids[:, 3].reshape(-1, 1)).ravel()
        INF_centroids_y_orig = self.scaler_y.inverse_transform(INF_centroids[:, 4].reshape(-1, 1)).ravel()
        INF_centroids_client_side_model_size_orig = self.scaler_client_side_model_size.inverse_transform(
            INF_centroids[:, 5].reshape(-1, 1)).ravel()

        # 合并成原始坐标空间下的聚类中心
        INF_centroids_orig = np.column_stack(
            [INF_centroids_client_side_data_num_orig, INF_centroids_server_side_data_num_orig, INF_centroids_f_orig,
             INF_centroids_x_orig, INF_centroids_y_orig, INF_centroids_client_side_model_size_orig])

        self.distances_inf = np.zeros((len(INF_centroids_orig), num_servers))

        # 计算每个客户端与每个服务器之间的欧几里得距离
        for i in range(len(INF_centroids_orig)):
            for j in range(num_servers):
                # 计算欧几里得距离
                dist = np.sqrt((INF_centroids_orig[i, 3] - self.server_data['location_x'][j]) ** 2 +
                               (INF_centroids_orig[i, 4] - self.server_data['location_y'][j]) ** 2)
                self.distances_inf[i, j] = dist

        INF_rate = np.zeros((len(INF_centroids_orig), num_servers))
        pho_error = np.zeros((len(INF_centroids_orig), num_servers))
        for i in range(len(INF_centroids_orig)):
            for j in range(num_servers):
                B = self.server_data['bandwidth'][j]
                # 计算 SINR
                avg_SINR = self.sinr(self.distances_inf[i, j], self.server_data['bandwidth'][j] * 1e6)
                INF_rate[i, j] = B * 1e6 * np.log2(1 + avg_SINR) / 1e6  # mbps
                m = 10 ** (0.023 / 10)
                pho_error[i, j] = 1 - np.exp(-m / avg_SINR)
        time_list_inf = np.zeros((len(INF_centroids_orig), num_servers))
        for i in range(len(INF_centroids_orig)):
            for j in range(num_servers):
                t1 = self.time_per_pic * INF_centroids_orig[i, 0] / INF_centroids_orig[i, 2]
                t2 = self.time_per_pic * INF_centroids_orig[i, 1] / self.server_data['computing_power'][j]
                t_up = INF_centroids_orig[i, 5] / INF_rate[i, j]
                time = t1 + t2  # +t_up
                time_list_inf[i, j] = time * 1000  # ms

        row_ind, col_ind = linear_sum_assignment(time_list_inf)
        total_cost = time_list_inf[row_ind, col_ind].sum()

        self.inf_ass_clo_ind = col_ind

        self.inf_ass = [
            self.INF_cluster_result[j]
            for i in range(5)
            for j in np.where(self.inf_ass_clo_ind == i)[0]
        ]

        # CHOOSE
        self.cho_df = pd.DataFrame(self.client_info_stan,
                                   columns=['client_side_data_num', 'server_side_data_num', 'computing_power',
                                            'location_x', 'location_y', 'client_side_model_size', 'data_quality'])
        np_cho = np.array(self.cho_df)
        self.CHO_cluster = NSKMeans(n_clusters=3, random_state=config.seed)
        self.CHO_cluster.fit(np_cho, 3, True)
        CHO_labels = self.CHO_cluster.labels_
        CHO_centroids = self.CHO_cluster.cluster_centers_
        # 对每列单独反变换
        CHO_centroids_client_side_data_num_orig = self.scaler_client_side.inverse_transform(
            CHO_centroids[:, 0].reshape(-1, 1)).ravel()
        CHO_centroids_server_side_data_num_orig = self.scaler_server_side.inverse_transform(
            CHO_centroids[:, 1].reshape(-1, 1)).ravel()
        CHO_centroids_f_orig = self.scaler_f.inverse_transform(CHO_centroids[:, 2].reshape(-1, 1)).ravel()
        CHO_centroids_x_orig = self.scaler_x.inverse_transform(CHO_centroids[:, 3].reshape(-1, 1)).ravel()
        CHO_centroids_y_orig = self.scaler_y.inverse_transform(CHO_centroids[:, 4].reshape(-1, 1)).ravel()
        CHO_centroids_client_side_model_size_orig = self.scaler_client_side_model_size.inverse_transform(
            CHO_centroids[:, 5].reshape(-1, 1)).ravel()
        CHO_centroids_quality_orig = self.scaler_quality.inverse_transform(CHO_centroids[:, 6].reshape(-1, 1)).ravel()
        # 合并成原始坐标空间下的聚类中心
        CHO_centroids_orig = np.column_stack(
            [CHO_centroids_client_side_data_num_orig, CHO_centroids_server_side_data_num_orig, CHO_centroids_f_orig,
             CHO_centroids_x_orig, CHO_centroids_y_orig, CHO_centroids_client_side_model_size_orig,
             CHO_centroids_quality_orig])

        self.distances_cho = np.zeros((len(CHO_centroids_orig), num_servers))

        for i in range(len(CHO_centroids_orig)):
            for j in range(num_servers):
                # 计算欧几里得距离
                dist = np.sqrt((CHO_centroids_orig[i, 3] - self.server_data['location_x'][j]) ** 2 +
                               (CHO_centroids_orig[i, 4] - self.server_data['location_y'][j]) ** 2)
                self.distances_cho[i, j] = dist

        CHO_rate = np.zeros((len(CHO_centroids_orig), num_servers))
        pho_error = np.zeros((len(CHO_centroids_orig), num_servers))
        for i in range(len(CHO_centroids_orig)):
            for j in range(num_servers):
                B = self.server_data['bandwidth'][j]
                # 计算 SINR
                avg_SINR = self.sinr(self.distances_cho[i, j], self.server_data['bandwidth'][j] * 1e6)
                CHO_rate[i, j] = B * 1e6 * np.log2(1 + avg_SINR) / 1e6  # mbps
                m = 10 ** (0.023 / 10)
                pho_error[i, j] = 1 - np.exp(-m / avg_SINR)
        time_list = np.zeros((len(CHO_centroids_orig), num_servers))
        for i in range(len(CHO_centroids_orig)):
            for j in range(num_servers):
                t1 = self.time_per_pic * CHO_centroids_orig[i, 0] / CHO_centroids_orig[i, 2]
                t2 = self.time_per_pic * CHO_centroids_orig[i, 1] / self.server_data['computing_power'][j]
                t_up = CHO_centroids_orig[i, 5] / CHO_rate[i, j]
                time = t1 + t2 + t_up
                time_list[i, j] = time * 1000
        choose = np.argsort(CHO_centroids_orig[:, 6])[::-1]

        # Aggregation

        self.agg_df = pd.DataFrame(self.client_info_stan,
                                   columns=['client_side_model_size', 'location_x', 'location_y'])
        idex = np.where(CHO_labels == choose[0])[0]
        i = 1  # 从 choose[1] 开始
        while len(idex) < (num_clients * 0.3) and i < len(choose):
            tempidex = np.where(CHO_labels == choose[i])[0]
            idex = np.hstack([idex, tempidex])
            i += 1
        self.agg_df_cho = self.agg_df.iloc[idex]
        np_agg = np.array(self.agg_df_cho)
        self.AGG_cluster = NSKMeans(n_clusters=num_servers, random_state=config.seed)
        self.AGG_cluster.fit(np_agg, num_servers, True)
        AGG_labels = self.AGG_cluster.labels_
        self.AGG_cluster_result = []
        for l in range(num_servers):
            self.AGG_cluster_result.append(np.where(AGG_labels == l)[0])
        self.agg_global_clusters = [idex[local_ids] for local_ids in self.AGG_cluster_result]
        AGG_centroids = self.AGG_cluster.cluster_centers_
        # 对每列单独反变换
        AGG_centroids_client_side_model_size_orig = self.scaler_client_side.inverse_transform(
            AGG_centroids[:, 0].reshape(-1, 1)).ravel()
        AGG_centroids_x_orig = self.scaler_x.inverse_transform(AGG_centroids[:, 0].reshape(-1, 1)).ravel()
        AGG_centroids_y_orig = self.scaler_y.inverse_transform(AGG_centroids[:, 1].reshape(-1, 1)).ravel()
        # 合并成原始坐标空间下的聚类中心
        AGG_centroids_orig = np.column_stack(
            [AGG_centroids_client_side_model_size_orig, AGG_centroids_x_orig, AGG_centroids_y_orig])
        self.distances_agg = np.zeros((len(AGG_centroids_orig), num_servers))
        for i in range(len(AGG_centroids_orig)):
            for j in range(num_servers):
                # 计算欧几里得距离
                dist = np.sqrt((AGG_centroids_orig[i, 1] - self.server_data['location_x'][j]) ** 2 +
                               (AGG_centroids_orig[i, 2] - self.server_data['location_y'][j]) ** 2)
                self.distances_agg[i, j] = dist
        AGG_rate = np.zeros((len(AGG_centroids_orig), num_servers))
        for i in range(len(AGG_centroids_orig)):
            for j in range(num_servers):
                B = self.server_data['bandwidth'][j]
                # 计算 SINR
                avg_SINR = self.sinr(self.distances_agg[i, j], self.server_data['bandwidth'][j] * 1e6)
                AGG_rate[i, j] = B * 1e6 * np.log2(1 + avg_SINR) / 1e6
        time_list_agg = np.zeros((len(AGG_centroids_orig), num_servers))
        for i in range(len(AGG_centroids_orig)):
            for j in range(num_servers):
                t_up = AGG_centroids_orig[i, 0] / AGG_rate[i, j]
                time = t_up
                time_list_agg[i, j] = time * 1000
        row_ind, col_ind = linear_sum_assignment(time_list_agg)
        total_cost = time_list_agg[row_ind, col_ind].sum()

        self.agg_ass_clo_ind = col_ind
        self.agg_ass = [
            self.agg_global_clusters[j]
            for i in range(5)
            for j in np.where(self.agg_ass_clo_ind == i)[0]
        ]

    def sinr(self, d, b):

        N = 1000  # 样本数量

        Pt = 1.0  # W
        Gt = 10.0  # dBi（用线性值就是 10 倍增益）
        Gr = 1.0  # 用户设备天线增益
        fc = 3.5e9  # 5G Sub-6GHz
        lambda_c = 3e8 / fc

        N0 = 4e-21  # 更接近物理热噪声（W/Hz）
        # N0 = 1e-9  # 噪声功率谱密度 (W/Hz)

        I = 1e-9  # 邻区干扰功率
        sigma_list = [0.5, 1, 2, 3]  # 瑞利衰落的标准差
        colors_simulation = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']  # 仿真曲线的颜色
        colors_theory = ['tab:cyan', 'tab:pink', 'tab:red', 'tab:olive']  # 理论曲线的颜色
        # np.random.seed(config.seed)
        sigma = 1
        # ---------- 客户端与服务器之间的距离 ----------
        # 假设距离为 100 米
        # ---------- 大尺度路径损耗（Free Space Path Loss） ----------
        FSPL = Pt * Gt * Gr * (lambda_c / (4 * np.pi)) ** 2  # 衰减随 d^2

        # ---------- 小尺度瑞利衰落（复高斯） ----------
        h = (np.random.normal(0, sigma / np.sqrt(2), N) + 1j * np.random.normal(0, sigma / np.sqrt(2), N))
        Rayleigh_gain = np.abs(h) ** 2  # |h|^2 ~ Exp(sigma^2)

        # ---------- 总信道增益 ----------
        channel_gain = FSPL * Rayleigh_gain / (d ** 2)

        # ---------- SINR ----------
        SINR = channel_gain / (I + N0 * b)
        avg_SINR = np.mean(SINR)
        return avg_SINR

    def calculate_distances(self):
        # 初始化一个距离矩阵，存储每个客户端和服务器之间的距离
        self.distances = np.zeros((len(self.inf_df['location_x']), len(self.server_data['location_x'])))

        # 计算每个客户端与每个服务器之间的欧几里得距离
        for i in range(len(self.inf_df['location_x'])):
            for j in range(len(self.server_data['location_x'])):
                # 计算欧几里得距离
                dist = np.sqrt((self.inf_df['location_x'][i] - self.server_data['location_x'][j]) ** 2 +
                               (self.inf_df['location_y'][i] - self.server_data['location_y'][j]) ** 2)
                self.distances[i, j] = dist

    def init_inf_obs(self,client_quit_bool,server_quit_bool):
        if config.dynamic_env == True:
            client_number = len(self.client_data['location_x'])
            q_loss = np.load(f'npydata/Q_loss_client{client_number}.npy')
            quality = 1 / q_loss
            data_num = np.load(f'npydata/client_num_client{client_number}.npy')
            f = np.random.uniform(1, 2, size=client_number)
            loc_x = np.random.rand(client_number) * 500  # 原始 X 坐标
            loc_y = np.random.rand(client_number) * 500  # 原始 Y 坐标
            self.time_per_pic = 0.005  # per pic
            self.sd = 2 / 64

            self.client_data = {

                'location_x': loc_x,  # 客户端的X坐标
                'location_y': loc_y,  # 客户端的Y坐标
                'data_quality': quality,  # 数据质量，0到1之间的浮点数
                'data_num': data_num,  # 数据量，100到1000之间的整数
                'computing_power': f,  # 计算能力，1到2之间的整数
            }
            num_servers = len(self.server_data['location_x'])
            self.server_data = {
                'location_x': np.random.rand(num_servers) * 500,  # 服务器的X坐标
                'location_y': np.random.rand(num_servers) * 500,  # 服务器的Y坐标
                'computing_power': np.random.uniform(2, 4, size=num_servers),  # 计算能力，
                'bandwidth': np.random.randint(30, 50, size=num_servers),  # 最大带宽，
            }
            self.cluster()
            self.calculate_distances()
        if client_quit_bool:
            self.client_quit(self.client_data)
            self.cluster()
            self.calculate_distances()

        if server_quit_bool:
            self.server_quit(self.server_data)
            self.cluster()
            self.calculate_distances()

        s = []
        self.relize_fj = np.copy(self.server_data['computing_power'])
        self.relize_bandwidth = np.copy(self.server_data['bandwidth'])
        for j in range(len(self.server_data['location_x'])):
            for i in self.inf_ass[j]:
                s.append(np.array(
                    [self.client_data['data_num'][i], self.distances[i, j], self.client_data['computing_power'][i],
                     self.server_data['computing_power'][j], self.server_data['bandwidth'][j]]))
        return s[0]

    def step(self, action, stepi, stepj):
        done = False
        if stepj < len(self.inf_ass):
            if stepi < len(self.inf_ass[stepj]):
                idx = self.inf_ass[stepj][stepi]
                client_data_num = self.scaler_client_side.inverse_transform(
                    self.inf_df['client_side_data_num'][idx].reshape(-1, 1)).ravel().item()
                f = self.scaler_f.inverse_transform(self.inf_df['computing_power'][idx].reshape(-1, 1)).ravel().item()
                data_quantity = client_data_num
                distance = self.distances[idx, stepj]
                f_i = f
                self.relize_fj[stepj] = self.relize_fj[stepj] - action[1]
                f_j = self.relize_fj[stepj]
                self.relize_bandwidth[stepj] = self.relize_bandwidth[stepj] - action[0]
                bandwidth = self.relize_bandwidth[stepj]
                new_state = np.array([client_data_num, distance, f_i, f_j, bandwidth])
                stepi = stepi + 1
            else:
                stepj = stepj + 1
                stepi = 0
                if stepj < len(self.inf_ass):
                    idx = self.inf_ass[stepj][stepi]
                    client_data_num = self.scaler_client_side.inverse_transform(
                        self.inf_df['client_side_data_num'][idx].reshape(-1, 1)).ravel().item()
                    f = self.scaler_f.inverse_transform(
                        self.inf_df['computing_power'][idx].reshape(-1, 1)).ravel().item()

                    data_quantity = client_data_num
                    distance = self.distances[idx, stepj]
                    f_i = f
                    self.relize_fj[stepj] = self.relize_fj[stepj] - action[1]
                    f_j = self.relize_fj[stepj]
                    self.relize_bandwidth[stepj] = self.relize_bandwidth[stepj] - action[0]
                    bandwidth = self.relize_bandwidth[stepj]
                    new_state = np.array([client_data_num, distance, f_i, f_j, bandwidth])
                    stepi = stepi + 1
        if done == False:
            client_data_num = self.scaler_client_side.inverse_transform(
                self.inf_df['client_side_data_num'][idx].reshape(-1, 1)).ravel().item()
            f = self.scaler_f.inverse_transform(self.inf_df['computing_power'][idx].reshape(-1, 1)).ravel().item()
            server_data_num = self.scaler_server_side.inverse_transform(
                self.inf_df['server_side_data_num'][idx].reshape(-1, 1)).ravel().item()
            b = (action[0] + 1) * config.b
            B = np.array(config.B)
            m = 10 ** (0.023 / 10)
            # 计算 SINR
            avg_SINR_U = self.sinr(distance, b)
            r_uij = b * np.log2(1 + avg_SINR_U) / 1e6  # mbps

            # avg_SINR_D = self.sinr(distance, B)
            r_dij = 3.5 * r_uij  # mbps
            pho_error = 1 - np.exp(-m / avg_SINR_U)
            Error = 1 / (1 - pho_error)

            t_1 = (np.array(client_data_num) * self.time_per_pic) / f
            t_2 = (np.array(server_data_num) * self.time_per_pic) / np.array(
                (action[1] * 10 + 0.1))

            t_3 = (np.array(client_data_num) * self.sd) / r_uij
            t_4 = (np.array(client_data_num) * self.sd) / r_dij
            t = t_1 + t_2 + t_3 + t_4
            if stepi in self.agg_global_clusters[stepj]:
                # train_time = t + (np.array(client_data_num) * self.sd) / r_dij
                agg_time = (np.array(self.client_side_model_size[stepi]) * 11) / r_dij

            reward = (-(5 * t) - (5 * pho_error))  # 1e6，1e4
            # reward = action[0]+action[1]
        if stepj == (len(self.inf_ass) - 1) and stepi == len(self.inf_ass[stepj]):
            done = True

        return new_state, reward, done, stepi, stepj, t, pho_error


