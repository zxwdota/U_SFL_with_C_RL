import numpy as np
import random
import torch
from mydata_util import read_data, getdata
from mysfl_noacc import ResNet18_client_side, ResNet18_server_side, Baseblock, run, DatasetSplit, DataLoader, \
    global_model_evaluate


# state: [datasize,quality,f_n,f_e,r_ne,r_ec] [10,10,10,5,50,5] 这里fn单位用的是 unit/s
# 超参数：get quality的batchsample
class Env:
    def __init__(self, device, num_clients, num_servers, frac, test_size, per, sample_train_p):
        self.num_clients = num_clients
        self.num_servers = num_servers
        self.min_datasize = sample_train_p
        self.frac = frac
        self.test_size = test_size
        self.per = per
        self.choose_number = frac * self.num_clients
        self.dataset_train, self.dataset_test, self.dict_users, self.dict_users_test = read_data(self.num_clients,
                                                                                                 self.test_size,
                                                                                                 self.per)
        self.dataset_train_random = getdata(self.dict_users, self.min_datasize)
        self.dataset_test_random = getdata(self.dict_users_test, self.min_datasize)
        self.data_size = np.array([len(self.dataset_train_random[i]) for i in self.dataset_train_random]).reshape(1, -1)
        self.device = device
        self.net_glob_client = ResNet18_client_side()
        self.net_glob_client.to(self.device)
        self.net_glob_server = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
        self.net_glob_server.to(self.device)
        self.net_model_server = [self.net_glob_server for i in range(int(self.num_clients))]
        self.data_quality = self.get_quality()
        self.state_init()
        self.train_time = []
        self.fed_time = []
        self.train_energy = []
        self.acc_list = []

        self.cyclepunit = 0.00175
        self.cyclepunit_of_client = self.cyclepunit * 0.1765
        self.cyclepunit_of_server = self.cyclepunit * 0.8235
        self.perdatasize = 64/256
        self.idex=0
    def state_init(self):
        self.f_n = np.random.uniform(1, 2, size=(1, self.num_clients))  # 1-2Ghz
        # 1秒钟内 1.4Ghz处理800张 1GHz 处理571张图片，每个图片是0.00175cycle，本地需要17.65%cycle 也就是1ghz可以处理3235张图片
        self.f_e = np.random.uniform(2, 5, size=(1, self.num_servers))  # 2ghz-5ghz 需要8235%cycle，也就是1ghz可以处理693张图片
        self.r_n_e = np.random.randint(2049, 4096, size=(self.num_clients, self.num_servers))
        self.r_j_cloud = np.random.randint(4097, 8192, size=(1, self.num_servers))

    def reset(self):
        self.dataset_train_random = getdata(self.dict_users, self.min_datasize)  # 100-800
        self.data_size = np.array([len(self.dataset_train_random[i]) for i in self.dataset_train_random]).reshape(1, -1)
        self.data_quality = self.get_quality()
        self.pre_f_n = self.f_n
        self.pre_f_e = self.f_e
        self.pre_r_n_e = self.r_n_e
        self.pre_r_j_cloud = self.r_j_cloud

        self.f_n = np.random.uniform(1, 2, size=(1, self.num_clients))  # 1-2Ghz
        # 1秒钟内 1.4Ghz处理800张 1GHz 处理571张图片，每个图片是0.00175cycle，本地需要17.65%cycle 也就是1ghz可以处理3235张图片
        self.f_e = np.random.uniform(2, 5, size=(1, self.num_servers))  # 2ghz-5ghz 需要8235%cycle，也就是1ghz可以处理693张图片
        self.r_n_e = np.random.randint(2049, 4096, size=(self.num_clients, self.num_servers))
        self.r_j_cloud = np.random.randint(4097, 8192, size=(1, self.num_servers))
        state = np.concatenate(
            [(self.data_size - self.min_datasize + 1) / (len(self.dict_users[0]) - self.min_datasize),
             self.data_quality / 100,
             (self.pre_f_n - 1) / (2 - 1), (self.pre_f_e - 2) / (5 - 2),
             ((self.pre_r_n_e - 2048) / (4096 - 2048)).reshape(1, -1),
             (self.pre_r_j_cloud - 4096) / (8192 - 4096)], axis=1).reshape(-1)


        return state
        # self.net_glob_client = ResNet18_client_side()
        # self.net_glob_server = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
        # self.net_model_server = [self.net_glob_server for i in range(int(self.num_clients))]

    def step(self, action, select_num):
        action_i = action[0:select_num].astype(int)
        action_j = action[select_num:select_num * 2].astype(int)
        action_k = action[select_num * 2:select_num * 3].astype(int)
        action_f_i = action[select_num * 3:select_num * 4]
        action_f_j = action[select_num * 4:select_num * 5]
        train_dict = {}
        aggrate_dict = {}
        f_i = np.zeros(shape=[self.num_clients, self.num_servers])
        f_j = np.zeros(shape=[self.num_clients, self.num_servers])

        fidx = 0
        for i, j in zip(action_i, action_j):
            if j not in train_dict:
                train_dict[j] = []
            train_dict[j].append(i)
            f_i[i, j] = action_f_i[fidx]
            f_j[i, j] = action_f_j[fidx]
            fidx += 1

        for i, k in zip(action_i, action_k):
            if k not in aggrate_dict:
                aggrate_dict[k] = []
            aggrate_dict[k].append(i)
        total_train_list_time = []
        total_train_list_energy = []
        # 首先是训练时间
        for j in set(action_j):
            # 本地训练时间
            the_i = np.array(train_dict[j])
            datasize = self.data_size[:, the_i]
            train_t_client = (self.cyclepunit_of_client * datasize) / (
                    self.f_n[:, the_i] * f_i[the_i, j])
            # 上传smashed时间 ,经res-18和HAM10000测试， sd = 64 kb/256pic
            e_client = np.sum(
                np.square(self.f_n[:, the_i] * f_i[the_i, j]) * datasize * self.cyclepunit_of_client,
                axis=1)
            trans_up_sd = (self.perdatasize * datasize) / self.r_n_e[the_i, j]
            t_1_list = train_t_client + trans_up_sd
            # 注意，忽略了下载gradient的时间。
            t_1 = np.max(t_1_list, axis=1)
            # 服务器训练时间，串行？并行？注意这里的0.15和0.85大概是运算量的不同。暂时用串行
            train_t_list_server = (self.cyclepunit_of_server * self.data_size[:, the_i]) / (
                    self.f_n[:, j] * f_j[the_i, j])
            e_server = np.sum(np.square(self.f_n[:, j] * f_j[the_i, j]) * (
                    self.cyclepunit_of_server * datasize), axis=1)

            t_2 = np.sum(train_t_list_server, axis=1)
            total_time_client_and_server = t_1 + t_2
            e_total = e_client + e_server
            total_train_list_energy.append(e_total)
            total_train_list_time.append(total_time_client_and_server)
        sum_train_energy = np.sum(total_train_list_energy)
        max_train_time = np.max(total_train_list_time)
        self.train_energy.append(sum_train_energy)
        self.train_time.append(max_train_time)
        # 然后是聚合时间；
        # client_model的聚合：clienti上传client_model到k，k上传到cloud t = t_1+max(t_2),T = max(t)
        #               ，server模型上传，
        cmodel_size = 346  # kb
        smodel_size = 45568  # kb 44.5Mb
        fed_model_k = []
        for k in set(action_k):
            fed_t_1 = cmodel_size / self.r_n_e[aggrate_dict[k], k]
            fed_t_2 = smodel_size / self.r_j_cloud[:, k]
            t = np.max(fed_t_1) + fed_t_2
            fed_model_k.append(np.max(t))
        max_fed_time = max(fed_model_k)
        self.fed_time.append(max_fed_time)
        total_time = max_train_time + max_fed_time

        print("begin take action")
        batch_acc_list, batch_loss_list = self.take_FL_train(action_i)
        # batch_acc_list = [100.0]
        global_acc = np.mean(np.array(batch_acc_list), keepdims=True)
        print("end take action")
        print("get acc of global = {} and the time = {}, energy = {}", global_acc, total_time, sum_train_energy)
        # acc = batch_acc_list[len(batch_acc_list)-1]
        self.acc_list.append(float(global_acc))
        reward = (global_acc - total_time - sum_train_energy)  # 注意参数
        self.idex += 1
        # if global_acc >= 0:  # if acc > 70 finish
        if self.idex % 10 == 0:
            done = True
        else:
            done = False

        # 接下来进行 state trans state' 变化范围：data 和 q
        next_state = self.reset()
        return next_state, reward, done

    def take_FL_train(self, action_i):
        self.net_glob_client.train()

        selected_client_idx = action_i

        w_locals_client_list = []
        need_fed_check = False
        idx_collect = []
        w_locals_server_list = []  # net_glob_server.state_dict()
        batch_acc_test = []
        batch_loss_test = []
        print("begin FL train")
        self.net_glob_client, self.net_glob_server = run(self.choose_number, self.device, self.dataset_train,
                                                         self.dataset_test,
                                                         self.dataset_train_random, self.dataset_test_random,
                                                         idx_collect, self.net_glob_client, w_locals_client_list,
                                                         w_locals_server_list,
                                                         self.net_model_server, self.net_glob_server
                                                         , need_fed_check, selected_client_idx)
        print("end FL train")
        print("begin test global model")
        ldr_test = DataLoader(self.dataset_test, batch_size=256, shuffle=True)
        batch_acc_test, batch_loss_test = global_model_evaluate(ldr_test, self.net_glob_client, self.net_glob_server,
                                                                self.device, batch_acc_test, batch_loss_test,
                                                                criterion=torch.nn.CrossEntropyLoss())
        print("end test global model")
        return batch_acc_test, batch_loss_test

    def get_quality(self):
        sampled_dict = {}
        batch_acc_test = []
        batch_loss_test = []
        sampled_num = int(int(len(self.dataset_test) / self.num_clients) * 0.2)
        for key, dict_set in self.dict_users_test.items():
            # 随机采样
            sampled_ints = set(random.sample(self.dict_users_test[key], sampled_num))
            # 将采样结果添加到新字典中
            sampled_dict[key] = sampled_ints
        print("begin get data quality")
        for idx in range(self.num_clients):
            quality = DataLoader(DatasetSplit(self.dataset_test, sampled_dict[idx]), batch_size=sampled_num, shuffle=True)
            batch_acc_test, batch_loss_test = global_model_evaluate(quality, self.net_glob_client, self.net_glob_server,
                                                   self.device, batch_acc_test, batch_loss_test,
                                                   criterion=torch.nn.CrossEntropyLoss())
        # for idx in range(self.num_clients):
        #     batch_acc_test.append(1.0)
        print("end got data quality")
        return np.array(batch_acc_test).reshape(1, -1)  # np.array(batch_loss_test).reshape(1,-1)

# e = Env()
# q = e.get_quality()
# action = np.array([1.,         5.,         9.,         3.,         4.,         0.,
#  4.,         1.,         0.,         0.54198116, 1.14510202, 1.0263958,
#  1.06700456, 1.55844092, 1.12455952])
# ns,re,d = e.step(action, 3)
#
# j = 1
