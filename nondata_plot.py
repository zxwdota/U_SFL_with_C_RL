import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import ConcatDataset
import config
from mydata_util import read_data
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from mydata_util import DatasetSplit, random_get_dict
from mysfl_noacc import run,ResNet18_client_side,Baseblock,ResNet18_server_side
import copy
from scipy.spatial.distance import jensenshannon
from matplotlib.ticker import FuncFormatter
import pickle
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
    client_idcs = dirichlet_split_noniid(
        labels, alpha=config.dirichlet_alpha, n_clients=config.num_clients)

    dict_users_non_iid = {}
    for i, ndarray in enumerate(client_idcs):
        dict_users_non_iid[i] = set(ndarray)
    # 选择柔和的颜色调色板，比如 Pastel1 或 Set3
    colors = plt.cm.Set3.colors[:config.num_clients]
    # 如果需要更多颜色，可以用重复的方法扩展颜色数组：
    colors = colors * (config.num_clients // len(colors)) + colors[:config.num_clients % len(colors)]

    # 展示不同label划分到不同client的情况
    plt.figure(figsize=(4.5, 4),dpi=400)
    plt.hist([labels[idc]for idc in client_idcs], stacked=True,
             bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
             label=["{}".format(i) for i in range(config.num_clients)],
             rwidth=0.5,alpha=1, color=colors)
    plt.xticks(np.arange(n_classes), train_data.classes,)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))  # 设置Y轴科学计数法
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xlabel("Label type", fontsize=14)
    plt.ylabel("Number of samples", fontsize=14)
    plt.legend(fontsize=10,ncol=2)
    # plt.title("Display Label Distribution on Different Clients")
    plt.tight_layout()
    plt.show()
    x_positions = np.arange(0, 20, 1)
    x_labels = [str(i) for i in range(20)]
    colors = plt.cm.Set3.colors[:7]
    # 如果需要更多颜色，可以用重复的方法扩展颜色数组：
    colors = colors * (7 // len(colors)) + colors[:7 % len(colors)]

    # 展示不同client上的label分布
    plt.figure(figsize=(6.5, 4),dpi=400)
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, config.num_clients + 1.5, 1),
             label=classes, rwidth=0.5, alpha=1, color=colors)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    # plt.xticks(np.arange(config.num_clients), ["%d" %
    #                                   c_id for c_id in range(config.num_clients)])
    # 设置 x 轴刻度间隔
    plt.xticks(x_positions, [""] * len(x_positions))  # 隐藏默认刻度标签

    # 设置单数刻度标签在第一行，复数刻度标签在第二行
    for i, x in enumerate(x_positions):
        if i % 2 == 0:
            # 复数（偶数）刻度在第二行
            plt.text(x, -30.5, x_labels[i], ha='center', va='top', fontsize=14)
        else:
            # 单数刻度在第一行
            plt.text(x, -30.5, x_labels[i], ha='center', va='top', fontsize=14)

    plt.yticks()
    # plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation2))  # 设置Y轴科学计数法
    plt.xlabel("Client ID", fontsize=14,labelpad=20)
    plt.ylabel("Number of samples", fontsize=14)
    plt.legend(fontsize=12)
    # plt.title("Display Label Distribution on Different Clients")
    plt.tight_layout()
    plt.show()
    return dataset_train, dataset_test, dict_users_non_iid, dict_users_iid, dict_users_test


def get_quality(dataset_train, dataset_test,Qdict_users,Q_dict_users_test):
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

    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def train_server(self, client_output_c, y):
        self.model_S.train()
        # self.model_S.to(self.device)
        self.optimizer.zero_grad()
        server_fx = self.model_S(client_output_c)
        loss_server = torch.nn.CrossEntropyLoss()(server_fx, y)
        loss_server.backward()
        client_output_grad = client_output_c.grad
        self.optimizer.step()
        # print("loss:", loss_server)
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

    def train_client(self, server):
        self.model_C.train()
        # self.model_C.to(self.device) # mps
        self.optimizer_client = torch.optim.Adam(self.model_C.parameters(), lr=self.lr)
        for local_train_num in range(1):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer_client.zero_grad()
                client_output = self.model_C(images)
                client_output_c = client_output.clone().detach().requires_grad_(True)
                client_output_grad = server.train_server(client_output_c, labels)
                client_output.backward(client_output_grad)
                self.optimizer_client.step()
    def update_model(self,globel_model_C):
        self.model_C = copy.deepcopy(globel_model_C).to(config.device_fl)


def federated_Client_averaging(global_model, clients, client_weights=None):
    """
    进行FedAvg聚合，更新全局模型。
    :param global_model: 全局模型
    :param clients: 客户端模型列表（每个客户端的模型应该是相同结构）
    :param client_weights: 每个客户端的权重（例如，基于数据量的大小）
    :return: 更新后的全局模型
    """
    # # 确保有client_weights，如果没有则默认为1（所有客户端权重相同）
    # if client_weights is None:
    #     client_weights = [1] * len(clients)

    # 初始化全局模型参数
    global_dict = global_model.state_dict()

    # 获取所有客户端模型的参数
    client_state_dicts = []
    for client in clients:
        client_state_dicts.append(copy.deepcopy(client.model_C.state_dict()))

    # 进行加权平均聚合
    for key in global_dict.keys():
        # 计算每个参数的加权平均
        weighted_sum = torch.zeros_like(global_dict[key])
        for i, client_dict in enumerate(client_state_dicts):
            weight = client_weights[i]
            weighted_sum += weight * client_dict[key]

        # 更新全局模型的参数
        global_dict[key] = weighted_sum / sum(client_weights)

    # 将聚合后的参数更新到全局模型中
    global_model.load_state_dict(global_dict)

    return global_model
def federated_Server_averaging(global_model, servers, server_weights=None):
    """
    进行FedAvg聚合，更新全局模型。
    :param global_model: 全局模型
    :param clients: 客户端模型列表（每个客户端的模型应该是相同结构）
    :param client_weights: 每个客户端的权重（例如，基于数据量的大小）
    :return: 更新后的全局模型
    """
    # 确保有client_weights，如果没有则默认为1（所有客户端权重相同）
    # if server_weights is None:
    #     server_weights = [1] * len(servers)

    # 初始化全局模型参数
    global_dict = global_model.state_dict()

    # 获取所有客户端模型的参数
    server_state_dicts = []
    for server in servers:
        server_state_dicts.append(copy.deepcopy(server.model_S.state_dict()))

    # 进行加权平均聚合
    for key in global_dict.keys():
        # 计算每个参数的加权平均
        weighted_sum = torch.zeros_like(global_dict[key])
        for i, server_dict in enumerate(server_state_dicts):
            weight = client_weights[i]
            weighted_sum += weight * server_dict[key]

        # 更新全局模型的参数
        global_dict[key] = weighted_sum / sum(server_weights)

    # 将聚合后的参数更新到全局模型中
    global_model.load_state_dict(global_dict)
    return global_model

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
    for client in clients:
        client_state_dicts.append(client.model_C.state_dict())

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

    torch.manual_seed(config.seed)

    n_clients = config.num_clients
    dirichlet_alpha = 1.0
    seed = 42
    global_model_C = ResNet18_client_side()
    global_model_S = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
    global_model_C.apply(initialize_weights)
    global_model_S.apply(initialize_weights)
    dataset_train, dataset_test, dict_users_non_iid, dict_users_iid, dict_users_test = read_data_non_iid()

    # iid 的 dict
    Q_dict_users_iid = random_get_dict(dict_users_iid, config.Q_sample_train_p_iid)

    Q_dict_users_test = random_get_dict(dict_users_test, config.Q_sample_train_p_iid)

    Q_dict_users_non_iid = random_get_dict(dict_users_non_iid, config.Q_sample_train_p_non_iid)

    all_categories = dataset_train.df['target'].unique()

    client_data = [dataset_train.df.iloc[list(dict_users_non_iid[i])] for i in range(n_clients)]
    client_counts = [client_data[i]['target'].value_counts().reindex(all_categories, fill_value=0) for i in range(n_clients)]
    client_data_dist = [client_counts[i] / client_counts[i].sum() for i in range(n_clients)]
    client_num = np.array([len(dict_users_non_iid[i]) for i in range(n_clients)])
    test_dist = dataset_test.df['target'].value_counts() / dataset_test.df['target'].value_counts().sum()

    # KLD = [entropy(client_data_dist[i], test_dist) for i in range(n_clients)]  # KLD 越小表示越相似
    # JSD = [jensenshannon(client_data_dist[i], test_dist) for i in range(n_clients)]  # JSD 越小表示越相似

    # q_acc, q_loss = get_quality(dataset_train, dataset_test, Q_dict_users_non_iid, Q_dict_users_test)

    # np.save(f'client_num_client{config.num_clients}',client_num)
    # np.save(f'KLD_client{config.num_clients}.npy', np.array(KLD))
    # np.save(f'JSD_client{config.num_clients}.npy', np.array(JSD))
    # np.save(f'Q_acc_client{config.num_clients}.npy', np.array(q_acc))
    # np.save(f'Q_loss_client{config.num_clients}.npy', np.array(q_loss))
