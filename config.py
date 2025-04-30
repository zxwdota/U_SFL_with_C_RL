# config.py
import torch
import numpy as np
split_freedom = True
dynamic_env = True
client_quit = True
server_quit = True
unbalence_dir = False
seed = 42
device_fl = torch.device("cuda")  # cpu cuda mps  问题：mps会让get_quality无法得到重复结果，cpu可以,且cpu在独立同分布的数据集上得到的结果相同。
num_clients = 50
num_servers = 5
f = 1
bandwidth = 50
frac = 0.6  # 参与的client比率
test_size = 0.2  # 测试集大小
per = 0.2  # non-iid中分配到每个client的比例，暂时没用。用iid分配。
Q_sample_train_p_iid = 0.2  # iid 测试数据质量的抽样比例
Q_sample_train_p_non_iid = 0.5  # non-iid 测试数据质量的抽样比例，较大的客户端数量导致客户端数据量较小，需要较高的抽样比例来增大方差。
dataset_q_frac = 0.3  # 服务器测试数据中，用于测试各客户端数据质量的数据比例。目前是2000*0.3=600
lr = 0.0001
dirichlet_alpha = 0.5
discrete_action_dim = int(num_clients * frac)
bandwidth_block = 10
b = 0.1e6  # 带宽（赫兹）
B = 2e7  # 带宽（赫兹）
sigma = 1
N0_dBmHz = -174  # 噪声功率谱密度, 单位: dBm/Hz
N0 = 10 ** (N0_dBmHz / 10) * 1e-3  # 转换为 W/Hz
PU = 0.1  # 0.1W
PD = 1  # 1W
m = 0.023
beta = ((3 * 1e8) / (4 * (1.8 * 1e9))) ** 4

actor_lr = 1e-4  # ppo:-5  ddpg:-5   PPO5000:1e-5
critic_lr = 1e-3  # ppo:-6  ddpg:-6  PPO5000:5e-5
ddpg_a_lr = 1e-5
ddpg_c_lr = 1e-2
dqn_lr = 1e-6
ddpg_epochs = 1
ppo_epochs = 10  # ppo5000:100  ddpg:1
num_episodes = 5000  # 500
ddpg_num_episodes = 1000
dqn_eposides = 5000
alpha_1 = 0.5  # 用于调整actor_loss的权重
alpth_2 = 0.5  # 用于调整actor_loss的权重
state_dim = 5
action_dim = 10
