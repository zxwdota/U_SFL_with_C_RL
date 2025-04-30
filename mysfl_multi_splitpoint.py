import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
from pandas import DataFrame
import random
import numpy as np
import matplotlib
import copy
from mydata_util import read_data
import config


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


def predictionmodel(net_model_server_new):
    global net_model_server
    net_model_server = net_model_server_new


def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


class ResNet18_head_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_head_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_planes = 64

        self.layer2 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer3 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer4 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer5 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x, cut_point_1, cut_point_2):
        input_point = 0
        exit_point = cut_point_1
        outputs = [None] * 8  # 0~7 层输出

        # 输入从哪层开始
        outputs[input_point] = x

        # 层执行逻辑（顺序执行 & 判断 early exit）
        if input_point <= 0:
            outputs[1] = F.relu(self.layer1(outputs[0]))
            if exit_point == 1:
                return outputs[1]

        if input_point <= 1:
            outputs[2] = self.layer2(self.maxpool(outputs[1]))
            if exit_point == 2:
                return outputs[2]

        if input_point <= 2:
            outputs[3] = self.layer3(outputs[2])
            if exit_point == 3:
                return outputs[3]

        if input_point <= 3:
            outputs[4] = self.layer4(outputs[3])
            if exit_point == 4:
                return outputs[4]

        if input_point <= 4:
            outputs[5] = self.layer5(outputs[4])
            if exit_point == 5:
                return outputs[5]


        if input_point <= 5:
            pooled = self.averagePool(outputs[5])
            flat = pooled.view(pooled.size(0), -1)
            outputs[6] = self.fc(flat)
            return outputs[6]

class ResNet18_mid_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_mid_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_planes = 64

        self.layer2 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer3 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer4 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer5 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x, cut_point_1, cut_point_2):
        input_point = cut_point_1
        exit_point = cut_point_2
        outputs = [None] * 8  # 0~7 层输出

        # 输入从哪层开始
        outputs[input_point] = x

        # 层执行逻辑（顺序执行 & 判断 early exit）
        if input_point <= 0:
            outputs[1] = F.relu(self.layer1(outputs[0]))
            if exit_point == 1:
                return outputs[1]

        if input_point <= 1:
            outputs[2] = self.layer2(self.maxpool(outputs[1]))
            if exit_point == 2:
                return outputs[2]

        if input_point <= 2:
            outputs[3] = self.layer3(outputs[2])
            if exit_point == 3:
                return outputs[3]

        if input_point <= 3:
            outputs[4] = self.layer4(outputs[3])
            if exit_point == 4:
                return outputs[4]

        if input_point <= 4:
            outputs[5] = self.layer5(outputs[4])
            if exit_point == 5:
                return outputs[5]


        if input_point <= 5:
            pooled = self.averagePool(outputs[5])
            flat = pooled.view(pooled.size(0), -1)
            outputs[6] = self.fc(flat)
            return outputs[6]

class ResNet18_tail_side(nn.Module):
    def __init__(self, block, num_layers, classes):
        super(ResNet18_tail_side, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.input_planes = 64

        self.layer2 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer3 = self._layer(block, 128, num_layers[0], stride=2)
        self.layer4 = self._layer(block, 256, num_layers[1], stride=2)
        self.layer5 = self._layer(block, 512, num_layers[2], stride=2)
        self.averagePool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x, cut_point_1, cut_point_2):
        input_point = cut_point_2
        exit_point = 7
        outputs = [None] * 8  # 0~7 层输出

        # 输入从哪层开始
        outputs[input_point] = x

        # 层执行逻辑（顺序执行 & 判断 early exit）
        if input_point <= 0:
            outputs[1] = F.relu(self.layer1(outputs[0]))
            if exit_point == 1:
                return outputs[1]

        if input_point <= 1:
            outputs[2] = self.layer2(self.maxpool(outputs[1]))
            if exit_point == 2:
                return outputs[2]

        if input_point <= 2:
            outputs[3] = self.layer3(outputs[2])
            if exit_point == 3:
                return outputs[3]

        if input_point <= 3:
            outputs[4] = self.layer4(outputs[3])
            if exit_point == 4:
                return outputs[4]

        if input_point <= 4:
            outputs[5] = self.layer5(outputs[4])
            if exit_point == 5:
                return outputs[5]


        if input_point <= 5:
            pooled = self.averagePool(outputs[5])
            flat = pooled.view(pooled.size(0), -1)
            outputs[6] = self.fc(flat)
            return outputs[6]

class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


# class ResNet18_mid_side(nn.Module):
#     def __init__(self, block, num_layers, classes):
#         super(ResNet18_mid_side, self).__init__()
#         self.input_planes = 64
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#         )
#         self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
#         self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
#         self.layer6 = self._layer(block, 512, num_layers[2], stride=2)
#         self.averagePool = nn.AvgPool2d(kernel_size=2, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _layer(self, block, planes, num_layers, stride):
#         dim_change = None
#         if stride != 1 or planes != self.input_planes * block.expansion:
#             dim_change = nn.Sequential(
#                 nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes * block.expansion))
#         netLayers = []
#         netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
#         self.input_planes = planes * block.expansion
#         for i in range(1, num_layers):
#             netLayers.append(block(self.input_planes, planes))
#             self.input_planes = planes * block.expansion
#
#         return nn.Sequential(*netLayers)
#
#     def forward(self, x):
#         out2 = self.layer3(x)
#         out2 = out2 + x
#         x3 = F.relu(out2)
#
#         x4 = self.layer4(x3)
#         x5 = self.layer5(x4)
#         x6 = self.layer6(x5)
#
#         # x7 = self.averagePool(x6)
#         # x8 = x7.view(x7.size(0), -1)
#         # y_hat = self.fc(x8)
#
#         return x6  # y_hat

# class ResNet18_tail_side(nn.Module):
#     def __init__(self, block, num_layers, classes):
#         super(ResNet18_tail_side, self).__init__()
#         self.input_planes = 64
#         self.averagePool = nn.AvgPool2d(kernel_size=2, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x7 = self.averagePool(x)
#         x8 = x7.view(x7.size(0), -1)
#         y_hat = self.fc(x8)
#
#         return y_hat

class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None):  # idx: client index
        self.idx = idx
        self.device = config.device_fl
        self.lr = lr
        self.local_ep = 1  # 本地训练次数
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256, shuffle=True)

    def train(self, net_model_server, client_finish_check, need_fed_check, num_client, idx_collect, w_locals_server_list, net_glob_server, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
        for l_epoch_count in range(self.local_ep):  # 本地训练轮次
            count1 = 0  # 表示训练到第几批次
            len_batch = len(self.ldr_train)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):  # 将这个client的ldr_train数据集分批次训练
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer_client.zero_grad()
                fx = net(images)
                client_fx = fx.clone().detach().requires_grad_(True)
                (dfx_client, count1, client_finish_check, need_fed_check, w_locals_server,
                 net_model_server, net_glob_server, idx_collect) = train_server(client_fx, labels, self.idx, num_client, net_model_server,
                                                                       w_locals_server_list, net_glob_server,
                                                                       idx_collect, client_finish_check, need_fed_check,
                                                                       count1, len_batch, l_epoch_count, self.local_ep,
                                                                       self.lr, self.device,
                                                                       criterion=nn.CrossEntropyLoss())  # dfx, net_model_server,
                fx.backward(dfx_client)
                optimizer_client.step()
        return (net.state_dict(), idx_collect, w_locals_server_list, net_model_server, net_glob_server,
                client_finish_check, need_fed_check)




def train_server(fx_client, y, idx, num_client, net_model_server,
                 w_locals_server, net_glob_server,
                 idx_collect, client_finish_check, need_fed_check,
                 count1, len_batch, l_epoch_count, l_epoch,
                 lr, device, criterion):
    net_server = copy.deepcopy(net_model_server[idx]).to(device)  # 创建一个server模型镜像

    net_server.train()
    optimizer_server = torch.optim.Adam(net_server.parameters(), lr=lr)
    optimizer_server.zero_grad()
    fx_client = fx_client.to(device)
    y = y.to(device)
    fx_server = net_server(fx_client)
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()
    net_model_server[idx] = copy.deepcopy(net_server)

    count1 += 1
    if count1 == len_batch:  # 表示设备结束了一次本地训练
        w_server = net_server.state_dict()  # 表示server在这轮聚合前最后一次训练的模型参数
        if l_epoch_count == l_epoch - 1:  # 表示设备完成了l_epoch次本地训练
            client_finish_check = True
            w_locals_server.append(copy.deepcopy(w_server))
            if idx not in idx_collect:
                idx_collect.append(idx)
        if len(idx_collect) == num_client:  # 表示所有设备训练结束，准备聚合
            need_fed_check = True
            print("fedrate server model")
            w_glob_server = FedAvg(w_locals_server)  # 聚合
            net_glob_server.load_state_dict(w_glob_server)
            net_model_server = [net_glob_server for i in range(int(num_client))]  # 更新所有server的模型
            w_locals_server = []
            idx_collect = []


    return (dfx_client, count1, client_finish_check, need_fed_check, w_locals_server,
            net_model_server, net_glob_server, idx_collect)


def global_model_evaluate(ldr_test, net_glob_client, net_global_server,
                batch_acc_test, batch_loss_test, criterion=nn.CrossEntropyLoss()):

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


def run(dataset_train,dataset_test,dict_users,dict_users_test, idx_collect, net_glob_client,w_locals_client_list, w_locals_server_list, net_model_server, net_glob_server,
         need_fed_check, selected_client_idx, lr=0.0001):
    num_client = config.num_clients
    device = config.device_fl
    for idx in selected_client_idx:
        client_finish_check = False
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])

        (w_client, idx_collect, w_locals_server_list, net_model_server, net_glob_server,
         client_finish_check, need_fed_check) = local.train(net_model_server,
                                       client_finish_check, need_fed_check, num_client, idx_collect,
                                       w_locals_server_list, net_glob_server,
                                       net=copy.deepcopy(net_glob_client).to(device))

        w_locals_client_list.append(copy.deepcopy(w_client))

        #local.evaluate(net_model_server, fed_iter, net=copy.deepcopy(net_glob_client).to(device))

    print(" Federate client model ")
    w_glob_client = FedAvg(w_locals_client_list)


    net_glob_client.load_state_dict(w_glob_client)
    return net_glob_client, net_glob_server

