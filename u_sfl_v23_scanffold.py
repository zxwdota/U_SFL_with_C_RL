import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from mydata_util import read_data
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from mydata_util import DatasetSplit, random_get_dict
from mysfl_noacc import run, ResNet18_client_side, Baseblock, ResNet18_server_side
import copy
from env_splitpoint_with_FL import *
from matplotlib.ticker import FuncFormatter

num_rounds = 20
K = 1


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
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1] * len(k_idcs)).
                                                  astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

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


def average_weighted_dict(dicts, weights):
    """
    dicts   : List[Dict[str, Tensor]]   Δw 或 Δc 的列表
    weights : List[float] 或 np.ndarray  对应的样本量／数据量权重
    返回    : Dict[str, Tensor]          权重平均后的字典
    """
    # 1) 归一化权重
    import numpy as _np
    ws = _np.array(weights, dtype=_np.float32)
    ws = ws / ws.sum()

    # 2) 首先深拷贝一份模板，并把所有张量置零
    avg = {k: dicts[0][k].clone().detach().zero_()
           for k in dicts[0].keys()}

    # 3) 按权重累加
    for w, d in zip(ws, dicts):
        for k in avg:
            # 保证在同一 device 上
            avg[k] = avg[k] + d[k].to(avg[k].device) * w

    return avg


def read_data_non_iid(dirichlet_alpha):
    dataset_train, dataset_test, dict_users_iid, dict_users_test = read_data()

    np.random.seed(config.seed)
    # train_data = datasets.EMNIST(
    #     root="C:/Users/zxw/Desktop/pythonProject/EMNIST", split="byclass", download=False, train=True)
    # test_data = datasets.EMNIST(
    #     root="C:/Users/zxw/Desktop/pythonProject/EMNIST", split="byclass", download=False, train=False)
    train_data = dataset_train
    test_data = dataset_test
    classes = np.array([0, 1, 2, 3, 4, 5, 6])
    n_classes = len(classes)

    # labels = np.concatenate(
    #    [np.array(train_data.df['target']), np.array(test_data.df['target'])], axis=0)
    labels = np.array(train_data.df['target'])
    dataset = ConcatDataset([train_data, test_data])

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    if config.unbalence_dir == True:
        client_idcs = dirichlet_hard_balance_split(
            labels, alpha=config.dirichlet_alpha, n_clients=config.num_clients)
    else:
        client_idcs = dirichlet_split_noniid(labels, alpha=dirichlet_alpha, n_clients=config.num_clients)
    dict_users_non_iid = {}
    for i, ndarray in enumerate(client_idcs):
        dict_users_non_iid[i] = set(ndarray)

    # 展示不同label划分到不同client的情况
    plt.figure(figsize=(4.5, 4), dpi=400)
    plt.hist([labels[idc] for idc in client_idcs], stacked=True,
             bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
             label=["{}".format(i) for i in range(config.num_clients)],
             rwidth=0.5)
    plt.xticks(np.arange(n_classes), train_data.classes, )
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))  # 设置Y轴科学计数法
    plt.xlabel("Label type")
    plt.ylabel("Number of samples")
    plt.legend(fontsize=6)
    plt.title("Display Label Distribution on Different Clients")
    plt.tight_layout()
    plt.show()

    # 展示不同client上的label分布
    plt.figure(figsize=(4.5, 4), dpi=400)
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


def evaluate(net_client, net_server, data_loader,
             device=config.device_fl,
             criterion=torch.nn.functional.cross_entropy):
    """
    Parameters
    ----------
    net_client : nn.Module
        头部模型（客户端侧）
    net_server : nn.Module
        尾部模型（服务器侧）
    data_loader : DataLoader
        用于评估的 DataLoader（整张测试集即可，不必拆分到客户端）
    device : torch.device
        推理设备
    criterion : callable
        损失函数，默认 cross-entropy

    Returns
    -------
    avg_acc : float
        平均 Top-1 准确率（%）
    avg_loss : float
        平均交叉熵 loss
    """
    net_client.eval()
    net_server.eval()
    net_client.to(device)
    net_server.to(device)

    total_correct, total_samples = 0, 0
    total_loss = 0.0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        feat = net_client(images)
        logits = net_server(feat)

        loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)

        # 统计
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_acc = 100. * total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_acc, avg_loss


import copy
import torch
from torch.utils.data import DataLoader
from mydata_util import DatasetSplit
import config

import copy
import torch
from torch.utils.data import DataLoader
from mydata_util import DatasetSplit
import config


class Client:
    def __init__(self, cid, dataset, idxs,
                 global_client_model, c_clients):
        self.cid = cid
        self.data = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=64, shuffle=True
        )
        # 本地模型 & 优化器
        self.model_C = copy.deepcopy(global_client_model) \
            .to(config.device_fl)
        self.opt_C = torch.optim.Adam(
            self.model_C.parameters(),
            lr=config.lr
        )
        # c_i 保证在同一设备
        self.c_i = {
            k: v.clone().to(config.device_fl)
            for k, v in c_clients[self.cid].items()
        }

    def train_one_round(self,
                        server_model,
                        c_server,
                        w_global,
                        local_steps):
        device = config.device_fl

        self.model_C.train()
        server_model.train()
        opt_S = torch.optim.Adam(
            server_model.parameters(),
            lr=config.lr
        )

        # 本地多步训练
        for _ in range(local_steps):
            for images, labels in self.data:
                images, labels = images.to(device), labels.to(device)

                # 客户端前向
                feat_c = self.model_C(images)
                feat_c_det = feat_c.detach().requires_grad_()

                # 服务器前向+反向
                opt_S.zero_grad()
                out = server_model(feat_c_det)
                loss_S = torch.nn.functional.cross_entropy(
                    out, labels, weight=class_weights
                )
                loss_S.backward()
                opt_S.step()

                # 客户端反传 + SCAFFOLD 校正
                self.opt_C.zero_grad()
                feat_c.backward(feat_c_det.grad)

                for name, param in self.model_C.named_parameters():
                    param.grad.data.add_(
                        c_server[name].to(device)
                        - self.c_i[name].to(device)
                    )
                self.opt_C.step()

        # 1) 计算 Δw_i
        w_new = self.model_C.state_dict()
        delta_w = {}
        for k in w_new:
            # 先在 CPU 上算差值，再搬到 device
            dw = (w_new[k].cpu() - w_global[k].cpu()) \
                .to(device)
            delta_w[k] = dw

        # 2) 计算 c_i_new 及 Δc_i
        c_i_new = {}
        delta_c = {}
        for k in self.c_i:
            # 保证所有项都在同一 device
            term = delta_w[k] / (local_steps * config.lr)
            c_i_new[k] = (
                    self.c_i[k]
                    - c_server[k].to(device)
                    + term
            )
            delta_c[k] = c_i_new[k] - self.c_i[k]

        # 更新本地 c_i
        self.c_i = c_i_new

        # 3) 收集更新后的 server state
        new_s_state = copy.deepcopy(
            server_model.state_dict()
        )

        return delta_w, delta_c, new_s_state

    def sync_with_global(self, new_client_state):
        self.model_C.load_state_dict(new_client_state)


class Server:
    def __init__(self, sid, global_server_model, clients, c_server_j):
        self.sid = sid
        self.global_S = copy.deepcopy(global_server_model).to(config.device_fl)
        self.clients = clients  # list[Client]
        # 本地 server control variate
        self.c_j = {k: v.clone() for k, v in c_server_j.items()}

    def train_one_round(self,
                        global_client_state,  # 客户端那一层的全局模型
                        c_global,  # 客户端那一层的全局 control variate
                        c_global_server,  # 服务器那一层的全局 control variate
                        local_steps,  # Server 侧本地步数 (比如 FedAvg 聚合步数1)
                        lr_server=config.lr  # Server 侧学习率
                        ):
        device = config.device_fl

        """
        global_client_state: 本轮下发给所有 client 的全局 client 模型 state_dict
        c_global:         服务器端的 control variate（字典）
        local_steps:      SCAFFOLD 本地步数 K
        """
        # 1) 清空累积列表
        delta_ws, delta_cs, temp_S_pool, data_sizes = [], [], [], []

        # 2) 遍历本服务器下的所有客户端
        for c in self.clients:
            # —— 深拷贝一次当轮初始的 server 模型 —— #
            temp_S = copy.deepcopy(self.global_S).to(config.device_fl)

            # —— 调用改过的 client.train_one_round，返回 Δw_i 和 Δc_i —— #
            dw_i, dc_i, updated_server_state = c.train_one_round(
                temp_S,  # server 模型
                c_global,  # 服务器 control variate
                global_client_state,  # 当轮全局 client 模型
                local_steps  # 本地步数 K
            )

            # —— 收集 SCAFFOLD 的增量和旧的 FedAvg 的 temp_S —— #
            delta_ws.append(dw_i)
            delta_cs.append(dc_i)
            temp_S_pool.append(updated_server_state)
            data_sizes.append(len(c.data.dataset))

        # 3) 聚合 Δw，得到 new_client_state
        mean_delta_w = average_weighted_dict(delta_ws, data_sizes)
        mean_delta_w_cpu = {k: v.cpu() for k, v in mean_delta_w.items()}

        # 先把 global_client_state 也搬到 CPU
        global_client_state_cpu = {
            k: v.cpu()
            for k, v in global_client_state.items()
        }

        # 2) 更新 client state（CPU）
        new_client_state = {
            k: global_client_state_cpu[k] + mean_delta_w_cpu[k]
            for k in global_client_state
        }

        # 4) 聚合 Δc，更新服务器 control variate c_global
        mean_delta_c = average_weighted_dict(delta_cs, data_sizes)
        for k in c_global:
            # 只更新浮点类型的 control variate
            if c_global[k].dtype.is_floating_point:
                # 确保 mean_delta_c[k] 在同一设备上
                delta = mean_delta_c[k].to(c_global[k].device)
                c_global[k] += (len(self.clients) / config.num_clients) * delta
        # 5) 原来的 intra-server FedAvg：聚合 temp_S_pool 更新 global_S
        new_server_state = average_weights(temp_S_pool, data_sizes)
        self.global_S.load_state_dict(new_server_state)

        # 6) 把 new_client_state 同步给本服务器所有 client
        for c in self.clients:
            c.sync_with_global(new_client_state)

        global_server_state = global_server_model.state_dict()
        new_S_state = new_server_state  # 从上面平均得到的 state_dict

        # 2.1 计算 Δw^(S)_j
        delta_w_S = {
            k: new_S_state[k].cpu() - global_server_state[k].cpu()
            for k in new_S_state
        }
        # 全搬回 device
        delta_w_S = {k: v.to(device) for k, v in delta_w_S.items()}

        # 2.2 更新本地 server 模型 c_j（Server 侧 SCAFFOLD）
        # c_j_new = c_j_old - c_global_server + Δw_S / (local_steps * lr_server)
        c_j_new = {}
        delta_c_S = {}
        for k in self.c_j:
            term = delta_w_S[k] / (local_steps * lr_server)
            c_j_new[k] = (
                    self.c_j[k]
                    - c_global_server[k]
                    + term
            )
            delta_c_S[k] = c_j_new[k] - self.c_j[k]

        # 2.3 更新本地 c_j
        self.c_j = c_j_new

        # 7) 返回：给外层第二阶段去聚合
        return new_client_state, new_server_state, sum(data_sizes), delta_c_S


def average_weights(state_dicts, weights=None):
    """
    state_dicts : List[OrderedDict]    # 各客户端/服务器模型参数
    weights     : List[float] | np.ndarray | None
                  # 对应样本量 n_i，若为 None → 等权
    返回        : OrderedDict          # 加权平均后的权重
    """
    if weights is None:
        weights = np.ones(len(state_dicts), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()  # 归一化到 1

    # —— 用第一份权重做模板，先深拷贝后清零 —— #
    avg = copy.deepcopy(state_dicts[0])

    for k in avg.keys():
        if torch.is_floating_point(avg[k]):  # ------- 浮点张量 -------
            avg[k].data.zero_()
            for w, sd in zip(weights, state_dicts):
                avg[k] += sd[k].to(avg[k].device) * w
            # 可选：强制与原 dtype 一致（尤其混用 fp16/bf16 时）
            avg[k] = avg[k].to(state_dicts[0][k].dtype)
        else:  # ------- 整型/布尔等 -------
            # 直接保留第一个客户端/服务器的值
            avg[k] = state_dicts[0][k].clone()

    for k in avg.keys():  # 只处理张量
        if torch.is_tensor(avg[k]):
            avg[k] = avg[k].cpu()

    return avg  # 仍返回 CPU 版（跟旧逻辑兼容）


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


from scipy.stats import entropy


########################################################################
# 在 read_data_non_iid() 之后、进入正式训练前执行一次即可
########################################################################
def compute_class_weights(dataset, num_classes):
    """
    dataset : torch.utils.data.Dataset（train 全集即可）
    num_classes : 类别数（如 7）
    返回:  shape=(num_classes,) 的 torch.FloatTensor
    """
    # 统计每个类别的样本数
    targets = np.array(dataset.df['target'])
    class_cnt = np.bincount(targets, minlength=num_classes)  # [n_0, …, n_{K-1}]

    # 计算加权系数：公式可按需要自定义
    # 例：w_k = N / (K * n_k)            —— “反频率”方案
    # 或：w_k = 1 / sqrt(n_k)            —— 稍缓和
    total = class_cnt.sum()
    weights = total / (len(class_cnt) * class_cnt + 1e-12)

    # 转为 tensor 并丢到 GPU
    return torch.tensor(weights, dtype=torch.float32, device=config.device_fl)


if __name__ == '__main__':
    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)

    dirichlet_alpha = 0.5

    dataset_train, dataset_test, dict_users_non_iid, dict_users_iid, dict_users_test = read_data_non_iid(
        dirichlet_alpha)

    # ──────────────────── 仅调用一次 ────────────────────
    NUM_CLASSES = 7
    class_weights = compute_class_weights(dataset_train, NUM_CLASSES)
    print("类别权重 =", class_weights)

    client_data_num = np.array([len(dict_users_non_iid[idex]) for idex in range(len(dict_users_iid))])

    eval_data = DataLoader(dataset_test, batch_size=206, shuffle=False)

    data_quality_loss = np.load(f'response_data/FL_data/50client_non_iid_v2/Q_loss_client{config.num_clients}.npy')
    env = Env(data_quality_loss, client_data_num)
    choose_client_index = env.agg_ass
    server_num = len(env.server_data['location_x'])
    choose_server_index = range(server_num)
    # 1. 全局模型 & 控制变量初始化
    global_client_model = ResNet18_client_side()
    global_server_model = ResNet18_server_side(Baseblock, [2, 2, 2], NUM_CLASSES)
    for m in (global_client_model, global_server_model):
        m.apply(initialize_weights)

    # 1 SCAFFOLD control variates for clients & server
    c_global_client = {
        k: v.to(config.device_fl)
        for k, v in global_client_model.state_dict().items()
    }
    c_clients = {
        i: {
            k: torch.zeros_like(v)
            for k, v in global_client_model.state_dict().items()
        }
        for i in choose_client_index
    }
    c_global_server = {k: torch.zeros_like(v).to(config.device_fl)
                       for k, v in global_server_model.state_dict().items()}
    c_servers = {
        sid: {k: torch.zeros_like(v).to(config.device_fl)
              for k, v in global_server_model.state_dict().items()}
        for sid in choose_server_index
    }

    # 2. 构造 Client + Server 实例
    client, server = {}, {}
    for j in choose_server_index:
        # 每个 server 底下的 client 列表
        clist = []
        for i in choose_client_index[j]:
            cl = Client(i,
                        dataset_train,
                        dict_users_non_iid[i],
                        global_client_model,
                        c_clients)
            client[i] = cl
            clist.append(cl)
        server[j] = Server(j,
                           global_server_model,
                           clist,
                           c_servers[j])
    server_list = list(server.values())

    # 3. 外层循环：Rounds
    for r in range(num_rounds):
        # 3.1 保存并下发本轮全局 client state + c_global
        global_client_state = global_client_model.state_dict()
        # （c_global, c_global_server 已在上面初始化并被逐轮更新）

        # 3.2 intra-server + 客户端 SCAFFOLD + 服务器端 SCAFFOLD 更新
        server_states_for_inter = []
        client_states_for_inter = []
        server_sample_nums = []
        server_delta_cs = []  # 新增，用于收集每台 Server 的 Δc_S

        for srv in server_list:
            # train_one_round 现在返回四个值：new_client_state, new_server_state, data_num, delta_c_S_j
            new_c_state, new_s_state, data_num, delta_c_S_j = srv.train_one_round(
                global_client_state,  # 下发给客户端层的全局模型
                c_global,  # 客户端层的全局 control variate
                c_global_server,  # 服务器层的全局 control variate
                local_steps=K  # 本地步数
            )

            client_states_for_inter.append(new_c_state)
            server_states_for_inter.append(new_s_state)
            server_sample_nums.append(data_num / len(srv.clients))
            server_delta_cs.append(delta_c_S_j)

        # 3.3 第②层跨服务器聚合（模型参数 & server-level SCAFFOLD）
        # —— 普通 FedAvg 跨服务器模型聚合 —— #
        inter_client_state = average_weights(
            client_states_for_inter,
            server_sample_nums
        )
        inter_server_state = average_weights(
            server_states_for_inter,
            server_sample_nums
        )

        # —— SCAFFOLD 聚合 Δc_S 更新全局 server control variate —— #
        mean_delta_c_S = average_weighted_dict(
            server_delta_cs,
            server_sample_nums
        )
        for k in c_global_server:
            if c_global_server[k].dtype.is_floating_point:
                c_global_server[k] += mean_delta_c_S[k].to(c_global_server[k].device)

        # 3.4 更新全局模型 & 同步
        global_client_model.load_state_dict(inter_client_state)
        global_server_model.load_state_dict(inter_server_state)

        for srv in server_list:
            srv.global_S.load_state_dict(inter_server_state)
            for cl in srv.clients:
                cl.sync_with_global(inter_client_state)

        # 3.5 评估、日志……
        test_acc, test_loss = evaluate(
            global_client_model,
            global_server_model,
            eval_data
        )
        print(f"[Round {r:02d}] acc={test_acc:.2f}% loss={test_loss:.4f}")