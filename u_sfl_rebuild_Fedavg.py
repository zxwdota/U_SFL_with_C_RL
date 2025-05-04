from copy import deepcopy

import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

from config import device_fl
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
import json
SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
device_fl= torch.device("mps")  # cpu cuda mps
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
_SAVE_DIR = 'rebuild/all_client_1turn_fedavg/'
_User_DIR = 'rebuild/'
# 每次从轮初始的服务器模型开始训练
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

def evaluate(net_client, net_server, data_loader,
             device=device_fl):
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
    def __init__(self, cid, idxs):
        global dataset_train, global_client_model
        self.cid = cid
        self.idxs = idxs

    def train_one_round(self, server,ep):
        global dataset_train, global_client_model, temp_model, global_server_model

        c_model = copy.deepcopy(global_client_model).to(device_fl)
        data = DataLoader(DatasetSplit(dataset_train, self.idxs),
                               batch_size=64, shuffle=True)
        temp_model = global_server_model
        for t in range(ep):
            for images, labels in data:
                images, labels = images.to(device_fl), labels.to(device_fl)

                c_model.train()
                opt_c = torch.optim.Adam(c_model.parameters(), lr=config.lr)

                opt_c.zero_grad()

                # 1) 前向取激活
                f_c = c_model(images)

                # 2) 送服务器，拿回 dL/df_c
                grad_f_c, s_model = server.train_oneround(f_c.detach(), labels, class_weights)

                # 3) 在本地把这段梯度“注入”进 f_c，更新 client 模型
                torch.autograd.backward(f_c, grad_tensors=grad_f_c)

                opt_c.step()

        return c_model.state_dict(), s_model.state_dict()

    def sync_with_global(self, new_client_state):
        self.global_C.load_state_dict(new_client_state)

class Server:
    def __init__(self, sid):
        self.sid = sid

    def train_oneround(self, f_c_detached, y, class_weights):
        """
        参数：
            f_c_detached: client_model(images).detach()，requires_grad=False
            y: labels
            class_weights: 分类权重
        返回：
            grad_f_c: tensor, shape same as f_c，用于客户端反向
        """
        global global_server_model, net_model_server, temp_model
        s_model = copy.deepcopy(temp_model).to(device_fl)
        s_model.train()
        opt_s = torch.optim.Adam(s_model.parameters(), lr=config.lr)
        opt_s.zero_grad()
        # 在服务器端继续构图，先让 f_c 可以求导
        f_c = f_c_detached.requires_grad_(True)
        y_hat = s_model(f_c)

        # 计算损失并反向——仅服务器参数和 f_c 会留下 grad
        loss = torch.nn.functional.cross_entropy(y_hat, y, weight=class_weights)
        loss.backward()
        opt_s.step()

        # 把 f_c 的梯度 detach 出来，传回客户端
        return f_c.grad.detach(), s_model



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

    # for k in avg.keys():  # 只处理张量
    #     if torch.is_tensor(avg[k]):
    #         avg[k] = avg[k].cpu()

    return avg                              # 仍返回 CPU 版（跟旧逻辑兼容）

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc

from scipy.stats import entropy

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
    return torch.tensor(weights, dtype=torch.float32, device=device_fl)


if __name__ == '__main__':


    with open(_User_DIR + 'data_dict.pkl', 'rb') as f:
        dataset_train = pickle.load(f)
        dataset_test = pickle.load(f)
        dict_users_non_iid = pickle.load(f)
        dict_users_iid = pickle.load(f)
        dict_users_test = pickle.load(f)


    #计算全局类别权重
    NUM_CLASSES = 7
    class_weights = compute_class_weights(dataset_train, NUM_CLASSES)


    client_data_num = np.array([len(dict_users_non_iid[idex]) for idex in range(len(dict_users_iid))])

    Dataset_test_loder = DataLoader(dataset_test, batch_size=206, shuffle=False)

    q_loss = np.load(f'response_data/FL_data/50client_non_iid_v2/Q_loss_client{config.num_clients}.npy')

    env = Env(q_loss,client_data_num)
    choose_client_index = env.agg_ass
    server_num = len(env.server_data['location_x'])
    choose_server_index = range(server_num)

    client_index = []
    client_list = []
    server_list = []
    client_weights = []

    acc=[]
    loss=[]


    global_client_model = ResNet18_client_side()
    global_server_model = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
    for m in (global_client_model, global_server_model): m.apply(initialize_weights)


    for arr in choose_client_index:
        client_index.extend(arr.tolist())

    json.dump(client_index, open(_SAVE_DIR + 'client_index.json', 'w'))

    client_index = [i for i in range(50)]

    for cid in client_index:
        # client[i] = Client(i, dataset_train, dict_users_iid, global_client_model)
        client_list.append(Client(cid, dict_users_non_iid[cid]))
        server_list.append(Server(cid))
        client_weights.append(len(dict_users_non_iid[cid]))





    # server_list: List[Server]，每个 Server 内含若干 Client
    server_states_for_inter = []   # 临时容器
    client_states_for_inter = []

    num_rounds = 500          # = len(range(20))
    num_servers = len(server_list)

    # ---------- 外层进度条 ----------
    for r in tqdm(range(num_rounds),desc='Rounds',unit='round'):
        server_states_for_inter.clear()
        client_states_for_inter.clear()
        local_ep = [1 for _ in client_index]
        for srv,cli,ep in zip(server_list,client_list,local_ep):
            c_model_dict,s_model_dict = cli.train_one_round(srv,ep)
            server_states_for_inter.append(s_model_dict)
            client_states_for_inter.append(c_model_dict)


        # ----- 聚合 -----
        # print('聚合')
        inter_server_state = average_weights(server_states_for_inter, client_weights)
        inter_client_state = average_weights(client_states_for_inter, client_weights)

        global_server_model.load_state_dict(inter_server_state)
        global_client_model.load_state_dict(inter_client_state)

        # 广播
        # for srv,cli in zip(server_list,client_list):
        #     srv.global_S = copy.deepcopy(global_server_model).to(device_fl)
        #     cli.global_C = copy.deepcopy(global_client_model).to(device_fl)

        # -------- 评估并把指标写进外层后缀 --------
        test_acc, test_loss = evaluate(global_client_model, global_server_model, Dataset_test_loder)
        print(f'[Round {r:02d}]  acc={test_acc:.2f}%  loss={test_loss:.4f}')
        # tqdm.write(f'[Round {r:02d}]  acc={test_acc:.2f}%  loss={test_loss:.4f}')

        acc.append(test_acc)
        loss.append(test_loss)

        torch.mps.empty_cache()
        # 也可以把结果挂在外层进度条后缀：

        json.dump(acc, open(_SAVE_DIR + 'test_acc.json', 'w'))
        json.dump(loss, open(_SAVE_DIR + 'test_loss.json', 'w'))
        pickle.dump(global_client_model, open(_SAVE_DIR + 'client_model.pkl', 'wb'))
        pickle.dump(global_server_model, open(_SAVE_DIR + 'server_model.pkl', 'wb'))

        tqdm._instances.clear()       # 避免 leave=False 条带来的一次性刷新问题
