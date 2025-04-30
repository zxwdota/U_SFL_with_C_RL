import numpy as np
import config
import pickle
import pandas as pd
from scipy.optimize import linear_sum_assignment

class Env:
    def __init__(self):
        with open('KLdata.pkl', 'rb') as f:
            self.K_means_fxy = pickle.load(f)
            self.K_means_q = pickle.load(f)
            self.K_means_xy = pickle.load(f)
            self.client_data = pickle.load(f)
            self.server_data = pickle.load(f)
        self.client_rayleigh = self.client_data['rayleigh']
        self.client_num = config.num_clients
        self.server_num = config.num_servers
        self.client_quality = np.load(f'npydata/Q_loss_client{config.num_clients}.npy')
        self.client_x = 1000*self.client_data['location_x']
        self.client_y = 1000*self.client_data['location_y']
        self.clients_location = np.array([self.client_x, self.client_y]).T
        self.server_x = 1000*self.server_data['location_x']
        self.server_y = 1000*self.server_data['location_y']
        self.servers_location = np.array([self.server_x, self.server_y]).T
        self.bandwidth = self.server_data['bandwidth']
        self.client_f =self.client_data['computing_power']
        self.server_f = self.server_data['computing_power']
        self.relize_fj = self.server_f
        self.relize_bandwidth = self.bandwidth
        self.calculate_distances()
        self.r_iu,self.r_id = self.KL_INF()
        self.r_au,self.r_ad = self.KL_AGG()
        self.client_data_num = np.load(f'npydata/client_num_client{config.num_clients}.npy')

        self.cluster_inf_idx = [np.where(self.K_means_fxy.labels_ == i)[0] for i in range(self.server_num)]
        self.cluster_q_idx = [np.where(self.K_means_q.labels_ == i)[0] for i in range(3)]
        self.cluster_agg_idx = [np.where(self.K_means_xy.labels_ == i)[0] for i in range(self.server_num)]
        a = np.load('sort.npy')
        if len(a) > 1:
            b = np.concatenate([self.cluster_q_idx[a[0]], self.cluster_q_idx[a[1]]])
        else:
            b = self.cluster_q_idx[a[0]]
        self.cluster_agg_idx_orgin = [b[i] for i in self.cluster_agg_idx]
        self.cluster_data_num = [self.client_data_num[self.cluster_inf_idx[i]].sum() for i in range(self.server_num)]
        self.avg_data_num = [self.cluster_data_num[i]/len(self.cluster_inf_idx[i]) for i in range(self.server_num)]
        self.cluster_f = [self.client_f[self.cluster_inf_idx[i]].sum()/len(self.cluster_inf_idx[i]) for i in range(self.server_num)]
        self.it = self.inftime()
        _,self.inf_server_assign = linear_sum_assignment(self.it)
        self.inf_ass = [self.cluster_inf_idx[i] for i in self.inf_server_assign]
        self.at = self.aggtime()
        _,self.agg_server_assign = linear_sum_assignment(self.at)
        self.agg_ass = [self.cluster_agg_idx_orgin[i] for i in self.agg_server_assign]
        self.s = self.init_obs()

    def inftime(self):
        self.time_per_pic = 0.00175
        self.time_client_side = 0.1765
        self.time_server_side = 0.8235
        self.sd = 64/256
        t_1 = (np.array(self.avg_data_num)*self.time_per_pic*self.time_client_side)/np.array(self.cluster_f)
        t_2 = (np.array(self.avg_data_num)*self.time_per_pic*self.time_server_side)/np.array(self.server_f)
        t_3 = (np.array(self.avg_data_num)*self.sd)/self.r_iu
        t_4 = (np.array(self.avg_data_num)*self.sd)/self.r_id
        t = t_1 + t_2 + t_3 + t_4
        return t
    def aggtime(self):
        self.client_model_size = 346
        self.server_model_size = 45568
        a = np.array([len(self.cluster_agg_idx_orgin[i]) for i in range(config.num_servers)])
        t = (a*self.client_model_size)/self.r_au
        return t
    def calculate_distances(self):
        # 初始化一个距离矩阵，存储每个客户端和服务器之间的距离
        self.distances = np.zeros((self.client_num, self.server_num))

        # 计算每个客户端与每个服务器之间的欧几里得距离
        for i in range(self.client_num):
            for j in range(self.server_num):
                # 计算欧几里得距离
                dist = np.sqrt((self.clients_location[i, 0] - self.servers_location[j, 0]) ** 2 +
                               (self.clients_location[i, 1] - self.servers_location[j, 1]) ** 2)
                self.distances[i, j] = dist
    def KL_INF(self):
        distance = np.zeros((self.K_means_fxy.cluster_centers_.shape[0],self.server_num))
        for i in range(self.K_means_fxy.cluster_centers_.shape[0]):
            for j in range(self.server_num):
                distance[i,j] = np.sqrt((self.K_means_fxy.cluster_centers_[i,1]*1000-self.servers_location[j,0])**2 + (self.K_means_fxy.cluster_centers_[i,2]*1000-self.servers_location[j,1])**2)
        h = distance**-2*config.sigma*config.beta
        labels = self.K_means_fxy.labels_
        label_counts = pd.Series(labels).value_counts()
        count = np.array(label_counts)
        b = (self.bandwidth/count[:,np.newaxis])*config.b
        SNR_ij = (config.PU*h)/(b*config.N0)
        r_u = (b*np.log2(1+SNR_ij))
        B = np.array(config.B)
        r_d = (B*np.log2(1+config.PD*h/(B*config.N0)))
        return r_u,r_d
    def KL_AGG(self):
        distance = np.zeros((self.K_means_xy.cluster_centers_.shape[0],self.server_num))
        for i in range(self.K_means_xy.cluster_centers_.shape[0]):
            for j in range(self.server_num):
                distance[i,j] = np.sqrt((self.K_means_xy.cluster_centers_[i,0]*1000-self.servers_location[j,0])**2 + (self.K_means_xy.cluster_centers_[i,1]*1000-self.servers_location[j,1])**2)
        h = distance**-2*config.sigma*config.beta
        labels = self.K_means_xy.labels_
        label_counts = pd.Series(labels).value_counts()
        count = np.array(label_counts)
        b = (self.bandwidth/count[:,np.newaxis])*config.b
        SNR_ij = (config.PU*h)/(b*config.N0)
        r_u = (b*np.log2(1+SNR_ij))
        B = np.array(config.B)
        r_d = (B*np.log2(1+config.PD*h/(B*config.N0)))
        return r_u, r_d
    def init_obs(self):

        s = []
        self.relize_fj = np.copy(self.server_f)
        self.relize_bandwidth = np.copy(self.bandwidth)
        for j in range(self.server_num):
            for i in self.inf_ass[j]:
                s.append(np.array([self.client_data_num[i],self.distances[i,j],self.client_f[i],self.server_f[j],self.bandwidth[j]]))
        return s[0]
    def step(self,action,stepi,stepj):
        done = False
        if stepj < len(self.inf_ass):
            if stepi < len(self.inf_ass[stepj]):
                idx = self.inf_ass[stepj][stepi]
                data_quantity = self.client_data_num[idx]
                distance = self.distances[idx, stepj]
                f_i = self.client_f[idx]
                self.relize_fj[stepj] = self.relize_fj[stepj] - action[1]
                f_j = self.relize_fj[stepj]
                self.relize_bandwidth[stepj] = self.relize_bandwidth[stepj] - action[0]
                bandwidth = self.relize_bandwidth[stepj]
                new_state = np.array([data_quantity, distance, f_i, f_j, bandwidth])
                stepi = stepi+1
            else:
                stepj = stepj+1
                stepi = 0
                if stepj<len(self.inf_ass):
                    idx = self.inf_ass[stepj][stepi]
                    data_quantity = self.client_data_num[idx]
                    distance = self.distances[idx, stepj]
                    f_i = self.client_f[idx]
                    self.relize_fj[stepj] = self.relize_fj[stepj] - action[1]
                    f_j = self.relize_fj[stepj]
                    self.relize_bandwidth[stepj] = self.relize_bandwidth[stepj] - action[0]
                    bandwidth = self.relize_bandwidth[stepj]
                    new_state = np.array([data_quantity, distance, f_i, f_j, bandwidth])
                    stepi = stepi + 1
        if done==False:
            h = (distance**-2)*config.sigma*config.beta
            b = (action[0]+1)*config.b
            r_uij = (b*np.log2(1+config.PU*h/(b*config.N0)))
            B = np.array(config.B)
            r_dij = (B * np.log2(1 + config.PD * h / (B * config.N0)))

            SNRP = config.PU*h
            SNRN = config.N0*b
            SNR = 10*np.log10(SNRP/SNRN)
            pho = 1-np.exp((-config.m/SNR))
            Error = 1/(1-pho)

            t_1 = (np.array(self.client_data_num[idx]) * self.time_per_pic * self.time_client_side) / np.array(
                self.client_f[idx]*1e6)
            t_2 = (np.array(self.client_data_num[idx]) * self.time_per_pic * self.time_server_side) / np.array(
                (action[1]+0.1)*1e6)
            t_3 = (np.array(self.client_data_num[idx]) * self.sd) / r_uij
            t_4 = (np.array(self.client_data_num[idx]) * self.sd) / r_dij
            t = t_1 + t_2 + t_3 + t_4

            reward = -((0.40*t*1e6) +(0.60*pho*1e5))  # 1e6，1e4
            # reward = action[0]+action[1]
        if stepj==(len(self.inf_ass)-1) and stepi==len(self.inf_ass[stepj]):
            done = True

        return new_state,reward,done,stepi,stepj,t,pho

    def reward(self,action):

        for j in len(self.inf_ass):
            for i in len(self.inf_ass[j]):
                idx = self.inf_ass[j][i]
                data_quantity = self.client_data_num[idx]
                distance = self.distances[idx,j]
                f_i = self.client_f[idx]
                f_j = self.relize_fj[j]
                bandwidth = self.relize_bandwidth[j]
                obs = np.array(idx,data_quantity,distance,f_i,f_j,bandwidth)





