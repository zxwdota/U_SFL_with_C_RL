from copy import deepcopy
import numpy as np
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
from typing import Dict, List, OrderedDict
SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
device_fl= torch.device("mps")  # cpu cuda mps
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
_SAVE_DIR = 'rebuild/local_5ep_scaffold/'
_User_DIR = 'rebuild/'
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
global_client_model = ResNet18_client_side()
global_server_model = ResNet18_server_side(Baseblock, [2, 2, 2], 7)
global_client_model.to(device_fl)
global_server_model.to(device_fl)

for m in (global_client_model, global_server_model): m.apply(initialize_weights)
c_global = [torch.zeros_like(param).to(device_fl)
            for param in global_client_model.parameters()]
s_global = [torch.zeros_like(param).to(device_fl)
            for param in global_server_model.parameters()]

# 每次从轮初始的服务器模型开始训练


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

        loss = torch.nn.functional.cross_entropy(logits, labels)

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
        global dataset_train, global_client_model, c_global
        self.cid = cid
        self.idxs = idxs
        self.c_local = copy.deepcopy(c_global)
        self.lr = 0.0001

    def train_one_round(self, server, ep):
        global dataset_train, global_client_model, s_model, global_server_model, c_global

        # copy 模型 和 加载data
        c_model = copy.deepcopy(global_client_model).to(device_fl)
        data = DataLoader(DatasetSplit(dataset_train, self.idxs),
                               batch_size=64, shuffle=True)
        s_model = copy.deepcopy(global_server_model).to(device_fl)


        # 训练ep轮
        for t in range(ep):
            for batch_idex, (images, labels) in enumerate(data):
                images, labels = images.to(device_fl), labels.to(device_fl)
                # 训练batchsize
                c_model.train()
                opt_c = torch.optim.Adam(c_model.parameters(), lr=self.lr)
                opt_c.zero_grad()
                f_c = c_model(images)
                if batch_idex == len(data)-1 and t==ep-1:
                    batch_finished = True
                else:
                    batch_finished = False

                grad_f_c, s_model = server.train_oneround(f_c.detach(), s_model, labels, ep, batch_finished)

                torch.autograd.backward(f_c, grad_tensors=grad_f_c)

                grads = []
                for p in c_model.parameters():
                    if p.requires_grad:
                        grads.append(p.grad.clone())
                # 4. 校正梯度
                corrected_grads = [
                    g - c_l + c_g
                    for g, c_l, c_g in zip(grads, self.c_local, c_global)
                ]
                # 5. 赋值校正梯度
                for p, cg in zip(c_model.parameters(), corrected_grads):
                    if p.requires_grad:
                        p.grad = cg
                opt_c.step()

        # 更新本地控制变量
        with torch.no_grad():
            c_plus = []
            for c_l, c_g, p_l, p_g in zip(self.c_local, c_global, c_model.parameters(), global_client_model.parameters()):
                c_plus.append(c_l - c_g - (1/(ep*self.lr)) * (p_g - p_l))
            self.c_local = c_plus



        return c_model.state_dict(), s_model.state_dict()

    def sync_with_global(self, new_client_state):
        self.global_C.load_state_dict(new_client_state)

class Server:
    def __init__(self, sid):
        self.sid = sid
        self.s_local = copy.deepcopy(s_global)
        self.lr = 0.0001



    def train_oneround(self, f_c_detached, s_model, y, ep, batch_finished):


        global global_server_model, net_model_server, temp_model, s_global
        s_model.train()
        opt_s = torch.optim.Adam(s_model.parameters(), lr=self.lr)
        opt_s.zero_grad()
        # 在服务器端继续构图，先让 f_c 可以求导
        f_c = f_c_detached.requires_grad_(True)
        y_hat = s_model(f_c)

        # 计算损失并反向——仅服务器参数和 f_c 会留下 grad
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        loss.backward()


        grads = []
        for p in s_model.parameters():
            if p.requires_grad:
                grads.append(p.grad.clone())
        # 4. 校正梯度
        corrected_grads = [
            g - s_l + s_g
            for g, s_l, s_g in zip(grads, self.s_local, s_global)
        ]
        # 5. 赋值校正梯度
        for p, cg in zip(s_model.parameters(), corrected_grads):
            if p.requires_grad:
                p.grad = cg
        opt_s.step()



        if batch_finished:
            with torch.no_grad():
                s_plus = []
                for s_l, s_g, p_l, p_g in zip(self.s_local, s_global, s_model.parameters(), global_server_model.parameters()):
                    s_plus.append(s_l - s_g - (1 / (ep * self.lr)) * (p_l - p_g))
                self.s_local = s_plus

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

def aggregate_client(res_cache, client_weights):
    global global_client_model, local_ep, client_index, c_global
    y_delta_cache = list(zip(*res_cache))[0]
    c_delta_cache = list(zip(*res_cache))[1]
    trainable_parameter = filter(
        lambda p: p.requires_grad,
        global_client_model.parameters()
    )
    # update global model
    avg_weight_1 = torch.tensor(
        [
            client_weights[i] / (len(client_index)*sum(client_weights))
            for i in range(len(client_index))
        ],
        device=device_fl,
    )

    avg_weight = torch.tensor(
        [
            1 / len(client_index)
            for _ in range(len(client_index))
        ],
        device=device_fl,
    )
    for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
        x_del = torch.sum(avg_weight_1 * torch.stack(y_del, dim=-1), dim=-1)
        param.data += x_del

    # update global control
    for c_g, c_del in zip(c_global, zip(*c_delta_cache)):
        c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
        c_g.data += (len(client_index) / config.num_clients) * c_del


def aggregate_server(res_cache, client_weights):
    # res_cache 是列表，每项是 (y_delta_list, s_delta_list)
    global global_server_model, local_ep, client_index, s_global

    y_delta_cache = list(zip(*res_cache))[0]
    s_delta_cache = list(zip(*res_cache))[1]
    # 更新全局 server 模型
    avg_weight_1 = torch.tensor(
        [
            client_weights[i] / (len(client_index)*sum(client_weights))
            for i in range(len(client_index))
        ],
        device=device_fl,
    )
    avg_weight = torch.tensor(
        [
            1 / len(client_index)
            for _ in range(len(client_index))
        ],
        device=device_fl,
    )
    trainable_parameter = filter(
        lambda p: p.requires_grad,
        global_server_model.parameters()
    )
    for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
        x_del = torch.sum(avg_weight_1 * torch.stack(y_del, dim=-1), dim=-1)
        param.data += x_del

    for s_g, s_del in zip(s_global, zip(*s_delta_cache)):
        s_del = torch.sum(avg_weight * torch.stack(s_del, dim=-1), dim=-1)
        s_g.data += (len(client_index) / config.num_clients) * s_del

def debug_norms(model, tag):
    for name, p in model.named_parameters():
        print(f"{tag} {name}: {p.data.norm().item():.6f}")
    print("-"*40)

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

    for arr in choose_client_index:
        client_index.extend(arr.tolist())

    json.dump(client_index, open(_SAVE_DIR + 'client_index.json', 'w'))

    client_index = range(50)


    for cid in client_index:
        # client[i] = Client(i, dataset_train, dict_users_iid, global_client_model)
        client_list.append(Client(cid, dict_users_non_iid[cid]))
        server_list.append(Server(cid))
        client_weights.append(len(dict_users_non_iid[cid]))




    # server_list: List[Server]，每个 Server 内含若干 Client
    server_states_for_inter = []   # 临时容器
    client_states_for_inter = []
    client_res_cache = []
    server_res_cache = []
    num_rounds = 100          # = len(range(20))
    num_servers = len(server_list)

    # ---------- 外层进度条 ----------
    for r in tqdm(range(num_rounds),desc='Rounds',unit='round'):
        server_states_for_inter.clear()
        client_states_for_inter.clear()
        client_res_cache.clear()
        server_res_cache.clear()
        local_ep = [min(int(800/client_weights[i]),5) for i in range(len(client_index))]
        # local_ep = [1 for i in range(len(client_index))]
        rc = random.sample(client_index, 5)
        for i in rc:
            c_model_dict,s_model_dict = client_list[i].train_one_round(server_list[i],local_ep[i])
            server_states_for_inter.append(s_model_dict)
            client_states_for_inter.append(c_model_dict)

        S = len(rc)
        N = len(client_list)
        with torch.no_grad():
            global_client_model.load_state_dict(average_weights(client_states_for_inter))
            global_server_model.load_state_dict(average_weights(server_states_for_inter))
            for srv, cli in zip(server_list,client_list):
                for cg, cl in zip(c_global, cli.c_local):
                    cg.data.copy_(cg + ((S/N)*(cl - cg)))
                for sg, sl in zip(s_global, srv.s_local):
                    sg.data.copy_(sg + ((S/N)*(sl - sg)))



        test_acc, test_loss = evaluate(global_client_model, global_server_model, Dataset_test_loder)
        print(f'[Round {r:02d}]  acc={test_acc:.2f}%  loss={test_loss:.4f}')
        # tqdm.write(f'[Round {r:02d}]  acc={test_acc:.2f}%  loss={test_loss:.4f}')

        acc.append(test_acc)
        loss.append(test_loss)

        # 也可以把结果挂在外层进度条后缀：

        json.dump(acc, open(_SAVE_DIR + 'test_acc.json', 'w'))
        json.dump(loss, open(_SAVE_DIR + 'test_loss.json', 'w'))
        pickle.dump(global_client_model, open(_SAVE_DIR + 'client_model.pkl', 'wb'))
        pickle.dump(global_server_model, open(_SAVE_DIR + 'server_model.pkl', 'wb'))

        tqdm._instances.clear()       # 避免 leave=False 条带来的一次性刷新问题
