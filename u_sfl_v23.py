import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from mydata_util import read_data
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from mydata_util import DatasetSplit, random_get_dict
from mysfl_noacc import run,ResNet18_client_side,Baseblock,ResNet18_server_side
import copy
from env_splitpoint_with_FL import *
from matplotlib.ticker import FuncFormatter

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
def read_data_non_iid(dirichlet_alpha):
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
        client_idcs = dirichlet_split_noniid(labels, alpha=dirichlet_alpha, n_clients=config.num_clients)
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

        feat   = net_client(images)
        logits = net_server(feat)

        loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)

        # 统计
        preds  = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss    += loss.item() * labels.size(0)

    avg_acc  = 100. * total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_acc, avg_loss


class Client:
    def __init__(self, cid, dataset, idxs, global_client_model):
        self.cid = cid
        self.data = DataLoader(DatasetSplit(dataset, idxs),
                               batch_size=64, shuffle=True)
        self.model_C = copy.deepcopy(global_client_model).to(config.device_fl)
        self.opt_C   = torch.optim.Adam(self.model_C.parameters(), lr=config.lr)

    def train_one_round(self, server_model):
        """server_model 已是本轮‘初始化’的 deepcopy"""
        self.model_C.train()
        server_model.train()
        opt_S = torch.optim.Adam(server_model.parameters(), lr=config.lr)

        for images, labels in self.data:
            images, labels = images.to(config.device_fl), labels.to(config.device_fl)

            # ---- 客户端前向 ----
            f_c = self.model_C(images)
            f_c_det = f_c.detach().requires_grad_()

            # ---- 服务器端 ----
            opt_S.zero_grad()
            out = server_model(f_c_det)
            loss_S = torch.nn.functional.cross_entropy(out, labels, weight=class_weights)
            loss_S.backward()
            opt_S.step()

            # ---- 反梯度回传到客户端 ----
            self.opt_C.zero_grad()
            f_c.backward(f_c_det.grad)
            self.opt_C.step()

        return (copy.deepcopy(self.model_C.state_dict()),
                copy.deepcopy(server_model.state_dict()))

    def sync_with_global(self, new_client_state):
        self.model_C.load_state_dict(new_client_state)

class Server:
    def __init__(self, sid, global_server_model, clients):
        self.sid = sid
        self.global_S = copy.deepcopy(global_server_model).to(config.device_fl)
        self.clients  = clients          # list[Client]

    def train_one_round(self):
        """遍历本服务器下的所有客户端 -> intra-server 聚合"""
        client_pool, temp_S_pool, data_sizes = [], [], []

        for c in self.clients:
            # 为该客户端复制“当轮初始”服务器模型
            temp_S = copy.deepcopy(self.global_S).to(config.device_fl)
            w_c, w_s = c.train_one_round(temp_S)
            client_pool.append(w_c)
            temp_S_pool.append(w_s)

            # 记录样本数量
            data_sizes.append(len(c.data.dataset))

        # ---- intra-server FedAvg (带权)----
        new_client_state = average_weights(client_pool, data_sizes)
        new_server_state = average_weights(temp_S_pool, data_sizes)
        server_sample_nums
        # 更新本服务器持有的 server_model
        self.global_S.load_state_dict(new_server_state)

        # 把聚合后的客户端模型广播给本服务器的所有客户端
        for c in self.clients:
            c.sync_with_global(new_client_state)

        # 返回：聚合好后的 server_state，供第②层再聚合
        return new_client_state, new_server_state, sum(data_sizes)



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
    weights = weights / weights.sum()               # 归一化到 1

    # —— 用第一份权重做模板，先深拷贝后清零 —— #
    avg = copy.deepcopy(state_dicts[0])

    for k in avg.keys():
        if torch.is_floating_point(avg[k]):      # ------- 浮点张量 -------
            avg[k].data.zero_()
            for w, sd in zip(weights, state_dicts):
                avg[k] += sd[k].to(avg[k].device) * w
            # 可选：强制与原 dtype 一致（尤其混用 fp16/bf16 时）
            avg[k] = avg[k].to(state_dicts[0][k].dtype)
        else:                                    # ------- 整型/布尔等 -------
            # 直接保留第一个客户端/服务器的值
            avg[k] = state_dicts[0][k].clone()

    for k in avg.keys():  # 只处理张量
        if torch.is_tensor(avg[k]):
            avg[k] = avg[k].cpu()

    return avg                              # 仍返回 CPU 版（跟旧逻辑兼容）

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
    class_cnt = np.bincount(targets, minlength=num_classes)     # [n_0, …, n_{K-1}]

    # 计算加权系数：公式可按需要自定义
    # 例：w_k = N / (K * n_k)            —— “反频率”方案
    # 或：w_k = 1 / sqrt(n_k)            —— 稍缓和
    total = class_cnt.sum()
    weights = total / (len(class_cnt) * class_cnt + 1e-12)

    # 转为 tensor 并丢到 GPU
    return torch.tensor(weights, dtype=torch.float32, device=config.device_fl)



SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


dirichlet_alpha = 0.5


dataset_train, dataset_test, dict_users_non_iid, dict_users_iid, dict_users_test = read_data_non_iid(dirichlet_alpha)

# ──────────────────── 仅调用一次 ────────────────────
NUM_CLASSES = 7
class_weights = compute_class_weights(dataset_train, NUM_CLASSES)
print("类别权重 =", class_weights)


# iid 的 dict
Q_dict_users_iid = random_get_dict(dict_users_iid, config.Q_sample_train_p_iid)

# non_iid 的 idex
Q_dict_users_test = random_get_dict(dict_users_test, config.Q_sample_train_p_iid)

Q_dict_users_non_iid = random_get_dict(dict_users_non_iid, config.Q_sample_train_p_non_iid)


client_data_num = np.array([len(dict_users_non_iid[idex]) for idex in range(len(dict_users_iid))])

Dataset_test_loder = DataLoader(dataset_test, batch_size=206, shuffle=False)

q_loss = np.load(f'response_data/FL_data/50client_non_iid_v2/Q_loss_client{config.num_clients}.npy')

env = Env(q_loss,client_data_num)
choose_client_index = env.agg_ass
server_num = len(env.server_data['location_x'])
choose_server_index = range(server_num)

global_model_C = ResNet18_client_side()
global_model_S = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
global_model_C.apply(initialize_weights)
global_model_S.apply(initialize_weights)

client = {}
server = {}
client_list = []
server_list = []
# 初始化所有客户端模型
for j in choose_server_index:
    temp_list = []
    for i in choose_client_index[j]:
        # client[i] = Client(i, dataset_train, dict_users_iid, global_model_C)
        client[i] = Client(i, dataset_train, dict_users_non_iid[i], global_model_C)
        temp_list.append(client[i])
    server[j] = Server(j, global_model_S, temp_list)
    client_list.append(temp_list)
    server_list.append(server[j])


acc=[]
loss=[]
clients = []
client_weights = []
servers = []
# cho_client_idx = np.array(range(50))
for j in choose_server_index:
    clients.append([client[i] for i in choose_client_index[j]])  # 所有客户端的模型
    client_weights.append([len(client[i].data.dataset) for i in choose_client_index[j]])  # 基于数据量的权重
    servers.append(server[j])  # 所有客户端的模型

global_client_model = ResNet18_client_side()
global_server_model = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
for m in (global_client_model, global_server_model): m.apply(initialize_weights)

# server_list: List[Server]，每个 Server 内含若干 Client
server_states_for_inter = []   # 临时容器
client_states_for_inter = []
server_sample_nums = []

num_rounds = 20          # = len(range(20))
num_servers = len(server_list)

# ---------- 外层进度条 ----------
for r in tqdm(range(num_rounds),
              desc='Rounds',
              unit='round'):
    with torch.no_grad():
        wC_before = global_client_model.layer1[0].weight.flatten()[:5].cpu().clone()
        wS_before = global_server_model.layer3[0].weight.flatten()[:5].cpu().clone()
    # ---------- 内层进度条 ----------
    server_states_for_inter.clear()
    client_states_for_inter.clear()
    server_sample_nums.clear()

    for srv in tqdm(server_list,
                    desc=f'  Servers (round {r})',
                    leave=False,          # 内层结束后自动清除
                    unit='srv'):
        new_client_state, new_server_state, data_num = srv.train_one_round()
        server_states_for_inter.append(new_server_state)
        client_states_for_inter.append(new_client_state)
        server_sample_nums.append(data_num/len(srv.clients))

    # ----- 第②层跨服务器聚合 -----
    inter_server_state = average_weights(server_states_for_inter, server_sample_nums)
    inter_client_state = average_weights(client_states_for_inter, server_sample_nums)

    global_server_model.load_state_dict(inter_server_state)
    global_client_model.load_state_dict(inter_client_state)

    # 广播
    for srv in server_list:
        srv.global_S.load_state_dict(inter_server_state)
        for c in srv.clients:
            c.sync_with_global(inter_client_state)

    # -------- 评估并把指标写进外层后缀 --------
    test_acc, test_loss = evaluate(global_client_model,
                                   global_server_model,
                                   Dataset_test_loder)

    print(f'[Round {r:02d}]  acc={test_acc:.2f}%  loss={test_loss:.4f}')

    with torch.no_grad():
        ΔC = (global_client_model.layer1[0].weight.flatten()[:5].cpu() - wC_before).abs().mean()
        ΔS = (global_server_model.layer3[0].weight.flatten()[:5].cpu() - wS_before).abs().mean()
    print(f"ΔW_client={ΔC:.2e}  ΔW_server={ΔS:.2e}")
    # 也可以把结果挂在外层进度条后缀：
    tqdm._instances.clear()       # 避免 leave=False 条带来的一次性刷新问题