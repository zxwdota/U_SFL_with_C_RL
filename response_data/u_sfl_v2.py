import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import ConcatDataset
import config
from mydata_util import read_data
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader
from mydata_util import DatasetSplit, random_get_dict
from mysfl_noacc import run,ResNet18_client_side,Baseblock,ResNet18_server_side
import copy
from env_splitpoint_with_FL import *
from scipy.spatial.distance import jensenshannon
from matplotlib.ticker import FuncFormatter
import pickle


# 每次从轮初始的服务器模型开始训练
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

# https://www.cnblogs.com/orion-orion/p/15897853.html
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    np.random.seed(config.seed)
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def dirichlet_soft_balance_split(train_labels, alpha, n_clients, imbalance_ratio=0.2):
    '''
    Dirichlet划分，且样本数量差异限制在允许比例内（soft balance）

    imbalance_ratio: 允许的最大样本数浮动比例，比如0.2就是±20%
    '''
    n_classes = train_labels.max() + 1
    n_samples = len(train_labels)
    avg_samples_per_client = n_samples / n_clients
    min_samples = int(avg_samples_per_client * (1 - imbalance_ratio))
    max_samples = int(avg_samples_per_client * (1 + imbalance_ratio))

    # 每类有哪些样本
    class_idcs = [np.where(train_labels == y)[0].tolist() for y in range(n_classes)]
    for idcs in class_idcs:
        np.random.shuffle(idcs)

    client_idcs = [[] for _ in range(n_clients)]
    client_samples = [0 for _ in range(n_clients)]

    for cls in range(n_classes):
        idcs = class_idcs[cls]
        np.random.shuffle(idcs)

        proportions = np.random.dirichlet([alpha] * n_clients)

        # 考虑客户端当前已有样本数，动态调整
        proportions = proportions / proportions.sum()

        split_points = (np.cumsum(proportions) * len(idcs)).astype(int)[:-1]
        split_idcs = np.split(idcs, split_points)

        for client, idcs_part in enumerate(split_idcs):
            if client_samples[client] + len(idcs_part) <= max_samples:
                client_idcs[client].extend(idcs_part)
                client_samples[client] += len(idcs_part)
            else:
                allowed = max(0, max_samples - client_samples[client])
                if allowed > 0:
                    client_idcs[client].extend(idcs_part[:allowed])
                    client_samples[client] += allowed
                # 剩余部分重新分配给别人
                leftover = idcs_part[allowed:]
                if len(leftover) > 0:
                    # 随机分配剩下的
                    for sample in leftover:
                        candidates = [i for i in range(n_clients) if client_samples[i] < max_samples]
                        if not candidates:
                            break
                        target = np.random.choice(candidates)
                        client_idcs[target].append(sample)
                        client_samples[target] += 1

    # 确保所有客户端都达到至少 min_samples
    for client in range(n_clients):
        while client_samples[client] < min_samples:
            candidates = [i for i in range(n_clients) if client_samples[i] > min_samples]
            if not candidates:
                break
            donor = np.random.choice(candidates)
            if len(client_idcs[donor]) > 0:
                moved_sample = client_idcs[donor].pop()
                client_idcs[client].append(moved_sample)
                client_samples[client] += 1
                client_samples[donor] -= 1

    return client_idcs

def dirichlet_hard_balance_split(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    n_samples = len(train_labels)
    samples_per_client = n_samples // n_clients

    # 预处理：每个类别有哪些样本
    class_idcs = [np.where(train_labels == y)[0].tolist() for y in range(n_classes)]
    for idcs in class_idcs:
        np.random.shuffle(idcs)

    client_idcs = [[] for _ in range(n_clients)]

    for client in range(n_clients):
        client_sampled = []

        # Dirichlet 采样当前客户端的类别分布
        class_proportions = np.random.dirichlet([alpha] * n_classes)

        # 目标：每类分配多少个样本
        class_sample_nums = (class_proportions * samples_per_client).astype(int)

        # 调整：防止总数对不上
        diff = samples_per_client - class_sample_nums.sum()
        for _ in range(abs(diff)):
            idx = np.argmax(class_proportions) if diff > 0 else np.argmin(class_sample_nums)
            class_sample_nums[idx] += np.sign(diff)

        # 开始取样
        for cls, n_samples_cls in enumerate(class_sample_nums):
            available = len(class_idcs[cls])
            n_take = min(available, n_samples_cls)
            if n_take > 0:
                take_idx = class_idcs[cls][:n_take]
                client_sampled.extend(take_idx)
                class_idcs[cls] = class_idcs[cls][n_take:]

        # 如果总数还是不够，从剩余类别补齐
        while len(client_sampled) < samples_per_client:
            for cls in range(n_classes):
                if len(class_idcs[cls]) > 0:
                    client_sampled.append(class_idcs[cls].pop(0))
                    if len(client_sampled) == samples_per_client:
                        break

        client_idcs[client] = np.array(client_sampled)

    return client_idcs
def scientific_notation(x, pos):
    return f'{int(x / 1000)}e3'
def scientific_notation2(x, pos):
    return f'{int(x / 100)}e3'
def read_data_non_iid():
    dataset_train, dataset_test, dict_users_iid, dict_users_test = read_data()

    np.random.seed(config.seed)
    # train_data = datasets.EMNIST(
    #     root="C:/Users/zxw/Desktop/pythonProject/EMNIST", split="byclass", download=False, train=True)
    # test_data = datasets.EMNIST(
    #     root="C:/Users/zxw/Desktop/pythonProject/EMNIST", split="byclass", download=False, train=False)
    train_data = dataset_train
    test_data = dataset_test
    classes = np.array([0,1,2,3,4,5,6])
    n_classes = len(classes)

    #labels = np.concatenate(
    #    [np.array(train_data.df['target']), np.array(test_data.df['target'])], axis=0)
    labels = np.array(train_data.df['target'])
    dataset = ConcatDataset([train_data, test_data])

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    if config.unbalence_dir==True:
        client_idcs = dirichlet_hard_balance_split(
            labels, alpha=config.dirichlet_alpha, n_clients=config.num_clients)
    else:
        client_idcs = dirichlet_split_noniid(labels, alpha=config.dirichlet_alpha, n_clients=config.num_clients)
    dict_users_non_iid = {}
    for i, ndarray in enumerate(client_idcs):
        dict_users_non_iid[i] = set(ndarray)


    # 展示不同label划分到不同client的情况
    plt.figure(figsize=(4.5, 4),dpi=400)
    plt.hist([labels[idc]for idc in client_idcs], stacked=True,
             bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
             label=["{}".format(i) for i in range(config.num_clients)],
             rwidth=0.5)
    plt.xticks(np.arange(n_classes), train_data.classes,)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))  # 设置Y轴科学计数法
    plt.xlabel("Label type")
    plt.ylabel("Number of samples")
    plt.legend(fontsize=6)
    plt.title("Display Label Distribution on Different Clients")
    plt.tight_layout()
    plt.show()

    # 展示不同client上的label分布
    plt.figure(figsize=(4.5, 4),dpi=400)
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, config.num_clients + 1.5, 1),
             label=classes, rwidth=0.5)
    plt.xticks(np.arange(config.num_clients), ["%d" %
                                      c_id for c_id in range(config.num_clients)])
    plt.yticks()
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation2))  # 设置Y轴科学计数法
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend(fontsize=6)
    plt.title("Display Label Distribution on Different Clients")
    plt.tight_layout()
    plt.show()
    return dataset_train, dataset_test, dict_users_non_iid, dict_users_iid, dict_users_test


def get_quality(dataset_train, dataset_test, Qdict_users,Q_dict_users_test):
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    test_idxs = Q_dict_users_test[0]
    test_data = DataLoader(DatasetSplit(dataset_test, test_idxs), batch_size=len(test_idxs), shuffle=False)
    global_model_C = ResNet18_client_side()
    global_model_S = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
    global_model_C.apply(initialize_weights)
    global_model_S.apply(initialize_weights)
    acc_list = []
    loss_list = []
    for i in range(0, config.num_clients):
        every_range_acc = []
        every_range_loss = []
        global_model_1 = copy.deepcopy(global_model_C).to(config.device_fl)
        global_model_2 = copy.deepcopy(global_model_S).to(config.device_fl)
        for epoch in range(5):
            global_model_1.train()
            global_model_2.train()
            idxs = Qdict_users[i]
            data = DataLoader(DatasetSplit(dataset_train,idxs), batch_size=256, shuffle=False)
            optimizer_client = torch.optim.Adam(global_model_1.parameters(), lr=config.lr)
            optimizer_server = torch.optim.Adam(global_model_2.parameters(), lr=config.lr)
            for batch_idx, (images, labels) in enumerate(data):
                images, labels = images.to(config.device_fl), labels.to(config.device_fl)
                client_output = global_model_1(images)
                client_output_c = client_output.clone().detach().requires_grad_(True)
                optimizer_client.zero_grad()
                optimizer_server.zero_grad()
                y = labels.clone().detach().to(config.device_fl)
                server_fx = global_model_2(client_output_c)
                loss_server = torch.nn.CrossEntropyLoss()(server_fx, y)
                acc_server = calculate_accuracy(server_fx, y)
                loss_server.backward()
                client_grad = client_output_c.grad
                print("client", i, "epoch", epoch, "batch_idx:", batch_idx, "loss:", loss_server)
                client_output.backward(client_grad)
                optimizer_server.step()
                optimizer_client.step()
            every_range_acc, every_range_loss = global_model_evaluate(test_data, global_model_1, global_model_2, every_range_acc, every_range_loss)
        acc_list.append(sum(every_range_acc)/len(every_range_acc))
        loss_list.append(sum(every_range_loss)/len(every_range_loss))
        print(acc_list, loss_list)
    return np.array(acc_list).reshape(-1), np.array(loss_list).reshape(-1)

def global_model_evaluate(ldr_test, net_glob_client, net_global_server,
                batch_acc_test, batch_loss_test, criterion=torch.nn.CrossEntropyLoss()):

    # torch.manual_seed(config.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    device = config.device_fl
    net_glob_client.to(device)
    net_global_server.to(device)
    net_glob_client.eval()
    net_global_server.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(ldr_test):
            images, labels = images.to(device), labels.to(device)
            fx_client = net_glob_client(images).to(device)
            fx_server = net_global_server(fx_client).to(device)
            loss = criterion(fx_server, labels)
            acc = calculate_accuracy(fx_server, labels)
            batch_acc_test.append(acc.item())
            batch_loss_test.append(loss.item())

    return batch_acc_test, batch_loss_test


class Server(object):
    def __init__(self, id, global_model_S):
        self.id = id
        self.model_S = copy.deepcopy(global_model_S).to(config.device_fl)
        self.device = config.device_fl
        self.lr = config.lr
        self.model_S.to(self.device)
        self.optimizer = torch.optim.Adam(self.model_S.parameters(), lr=self.lr)
        self.model_list = []

    def train_server(self, client_output_c, y, choose_j_client_len):

        self.temp = copy.deepcopy(self.model_S).to(config.device_fl)
        self.temp.train()
        self.optimizer.zero_grad()
        server_fx = self.temp(client_output_c)
        loss_server = torch.nn.CrossEntropyLoss()(server_fx, y)

        # ✅ 用 autograd.grad 显式求梯度
        client_output_grad = torch.autograd.grad(loss_server, client_output_c, retain_graph=True)[0]

        loss_server.backward(retain_graph=True)
        self.optimizer.step()

        self.model_list.append(copy.deepcopy(self.temp))
        if len(self.model_list) == choose_j_client_len:
            # 进行模型聚合
            global_model = federated_server_averaging_no_weight(self.model_S, self.model_list)
            self.model_S.load_state_dict(global_model.state_dict())
            self.model_list.clear()



        return client_output_grad

    def update_model(self,global_model_S):
        self.model_S = copy.deepcopy(global_model_S).to(config.device_fl)

class Client(object):
    def __init__(self, id, dataset, dict_users, global_model_C):
        self.id = id
        self.dataset = dataset
        self.dict_users = dict_users
        self.model_C = copy.deepcopy(global_model_C).to(config.device_fl)
        self.device = config.device_fl
        self.lr = config.lr
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, self.dict_users[self.id]), batch_size=64, shuffle=True)

    def train_client(self, server, choose_j_client_len):
        self.model_C.train()
        # self.model_C.to(self.device) # mps
        self.optimizer_client = torch.optim.Adam(self.model_C.parameters(), lr=self.lr)
        for local_train_num in range(1):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer_client.zero_grad()
                client_output = self.model_C(images)
                client_output.retain_grad()
                client_output_c = client_output
                client_output_grad = server.train_server(client_output_c, labels, choose_j_client_len)
                client_output.backward(client_output_grad)
                self.optimizer_client.step()
    def update_model(self,globel_model_C):
        self.model_C = copy.deepcopy(globel_model_C).to(config.device_fl)


# def federated_Client_averaging(global_model, clients, client_weights=None):
#     """
#     进行FedAvg聚合，更新全局模型。
#     :param global_model: 全局模型
#     :param clients: 客户端模型列表（每个客户端的模型应该是相同结构）
#     :param client_weights: 每个客户端的权重（例如，基于数据量的大小）
#     :return: 更新后的全局模型
#     """
#     # # 确保有client_weights，如果没有则默认为1（所有客户端权重相同）
#     # if client_weights is None:
#     #     client_weights = [1] * len(clients)
#
#     # 初始化全局模型参数
#     global_dict = global_model.state_dict()
#
#     # 获取所有客户端模型的参数
#     client_state_dicts = []
#     for client in clients:
#         client_state_dicts.append(copy.deepcopy(client.model_C.state_dict()))
#
#     # 进行加权平均聚合
#     for key in global_dict.keys():
#         # 计算每个参数的加权平均
#         weighted_sum = torch.zeros_like(global_dict[key])
#         for i, client_dict in enumerate(client_state_dicts):
#             weight = client_weights[i]
#             weighted_sum += weight * client_dict[key]
#
#         # 更新全局模型的参数
#         global_dict[key] = weighted_sum / sum(client_weights)
#
#     # 将聚合后的参数更新到全局模型中
#     global_model.load_state_dict(global_dict)
#
#     return global_model
# def federated_Server_averaging(global_model, servers, server_weights=None):
#     """
#     进行FedAvg聚合，更新全局模型。
#     :param global_model: 全局模型
#     :param clients: 客户端模型列表（每个客户端的模型应该是相同结构）
#     :param client_weights: 每个客户端的权重（例如，基于数据量的大小）
#     :return: 更新后的全局模型
#     """
#     # 确保有client_weights，如果没有则默认为1（所有客户端权重相同）
#     # if server_weights is None:
#     #     server_weights = [1] * len(servers)
#
#     # 初始化全局模型参数
#     global_dict = global_model.state_dict()
#
#     # 获取所有客户端模型的参数
#     server_state_dicts = []
#     for server in servers:
#         server_state_dicts.append(copy.deepcopy(server.model_S.state_dict()))
#
#     # 进行加权平均聚合
#     for key in global_dict.keys():
#         # 计算每个参数的加权平均
#         weighted_sum = torch.zeros_like(global_dict[key])
#         for i, server_dict in enumerate(server_state_dicts):
#             weight = client_weights[i]
#             weighted_sum += weight * server_dict[key]
#
#         # 更新全局模型的参数
#         global_dict[key] = weighted_sum / sum(server_weights)
#
#     # 将聚合后的参数更新到全局模型中
#     global_model.load_state_dict(global_dict)
#     return global_model

def federated_client_averaging_no_weight(global_model, clients):
    """
    进行FedAvg聚合，更新全局模型（不加权重，直接平均）。
    :param global_model: 全局模型
    :param clients: 客户端模型列表（每个客户端的模型应该是相同结构）
    :return: 更新后的全局模型
    """
    # 初始化全局模型参数
    global_dict = global_model.state_dict()

    # 获取所有客户端模型的参数
    client_state_dicts = []
    for client_set in clients:
        for client in client_set:
            client_state_dicts.append(client.model_C.state_dict())

    # 进行直接平均聚合
    for key in global_dict.keys():
        # 计算每个参数的平均
        total_sum = torch.zeros_like(global_dict[key]).to(config.device_fl)
        for client_dict in client_state_dicts:
            total_sum += client_dict[key]

        # 更新全局模型的参数
        global_dict[key] = total_sum / sum([len(clients[i]) for i in range(len(clients))])

    # 将聚合后的参数更新到全局模型中
    global_model.load_state_dict(global_dict)

    return global_model

def federated_server_averaging_no_weight(global_model, clients):
    """
    进行FedAvg聚合，更新全局模型（不加权重，直接平均）。
    :param global_model: 全局模型
    :param clients: 客户端模型列表（每个客户端的模型应该是相同结构）
    :return: 更新后的全局模型
    """
    # 初始化全局模型参数
    global_dict = global_model.state_dict()

    # 获取所有客户端模型的参数
    client_state_dicts = []
    for client in clients:
        client_state_dicts.append(client.model_S.state_dict())

    # 进行直接平均聚合
    for key in global_dict.keys():
        # 计算每个参数的平均
        total_sum = torch.zeros_like(global_dict[key]).to(config.device_fl)
        for client_dict in client_state_dicts:
            total_sum += client_dict[key]

        # 更新全局模型的参数
        global_dict[key] = total_sum / len(clients)

    # 将聚合后的参数更新到全局模型中
    global_model.load_state_dict(global_dict)

    return global_model

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc

from scipy.stats import entropy



if __name__ == '__main__':
    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))



    dataset_train, dataset_test, dict_users_non_iid, dict_users_iid, dict_users_test = read_data_non_iid()

    # iid 的 dict
    Q_dict_users_iid = random_get_dict(dict_users_iid, config.Q_sample_train_p_iid)

    # non_iid 的 idex
    Q_dict_users_test = random_get_dict(dict_users_test, config.Q_sample_train_p_iid)

    Q_dict_users_non_iid = random_get_dict(dict_users_non_iid, config.Q_sample_train_p_non_iid)


    q_acc, q_loss = get_quality(dataset_train, dataset_test, Q_dict_users_non_iid, Q_dict_users_test)
    client_data_num = np.array([len(dict_users_non_iid[idex]) for idex in range(len(dict_users_iid))])

    np.save(f'response_data/FL_data/50client_non_iid_v2/Q_acc_client{config.num_clients}.npy', np.array(q_acc))

    Dt = DataLoader(DatasetSplit(dataset_test, dict_users_test[0]), batch_size=206, shuffle=False)

    # q_loss = np.load(f'npydata/Q_loss_client{config.num_clients}.npy')

    env = Env(q_loss,client_data_num)
    choose_client_index = env.agg_ass
    server_num = len(env.server_data['location_x'])
    choose_server_index = range(server_num)

    # choose_client_index = [[0,1,2,3,4]]
    # choose_server_index = [0]

    global_model_C = ResNet18_client_side()
    global_model_S = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
    global_model_C.apply(initialize_weights)
    global_model_S.apply(initialize_weights)

    client = {}
    server = {}
    # 初始化所有客户端模型
    for j in choose_server_index:
        for i in choose_client_index[j]:
            # client[i] = Client(i, dataset_train, dict_users_iid, global_model_C)
            client[i] = Client(i, dataset_train, dict_users_non_iid, global_model_C)
        server[j] = Server(j, global_model_S)


    acc=[]
    loss=[]
    clients = []
    client_weights = []
    servers = []
    # cho_client_idx = np.array(range(50))
    for j in choose_server_index:
        clients.append([client[i] for i in choose_client_index[j]])  # 所有客户端的模型
        client_weights.append([len(client[i].ldr_train.dataset) for i in choose_client_index[j]])  # 基于数据量的权重
        servers.append(server[j])  # 所有客户端的模型


    # 训练500轮联邦
    for r in tqdm(range(500), desc='Federated Rounds'):
        tqdm.write(f'range:{r}')
        for j in choose_server_index:
            for i in choose_client_index[j]:
                client[i].train_client(server[j], len(choose_client_index[j]))

        global_model_C = federated_client_averaging_no_weight(global_model_C, clients)
        global_model_S = federated_server_averaging_no_weight(global_model_S, servers)

        for j in choose_server_index:
            for i in choose_client_index[j]:
                client[i].update_model(global_model_C)
            server[j].update_model(global_model_S)

        turn_acc = []
        turn_loss = []

        global_model_evaluate(Dt, global_model_C, global_model_S, turn_acc, turn_loss)

        round_acc = sum(turn_acc) / len(turn_acc)
        tqdm.write(f"acc: {turn_acc}, {turn_loss}")
        tqdm.write(f"avgacc: {round_acc}")
        acc.append(round_acc)

        # 可选：更新进度条后缀显示当前平均准确率
        tqdm.write(f'Round {r + 1} avg accuracy: {round_acc:.4f}')

        del Dt
        torch.cuda.empty_cache()

        with open('response_data/FL_data/50client_non_iid_v2/non_iid_acc.pkl', 'wb') as f:
            pickle.dump(acc, f)
        with open('response_data/FL_data/50client_non_iid_v2/non_iid_model_C.pkl', 'wb') as f:
            pickle.dump(global_model_C, f)
        with open('response_data/FL_data/50client_non_iid_v2/non_model_S.pkl', 'wb') as f:
            pickle.dump(global_model_S, f)

