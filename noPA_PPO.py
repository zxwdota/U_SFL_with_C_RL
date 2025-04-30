import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pythonProject import rl_utils
from env import Env
import torch
import random
from tqdm import tqdm

# joint 每次采样后需要调整剩余动作的概率分布。

torch.autograd.set_detect_anomaly(True)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, discrete_action_i_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bandwidth = torch.nn.Linear(hidden_dim, discrete_action_i_dim)  # 离散变量，通过softmax选择动作。输出的是概率分布。将带宽设置为N个，每个递增的带宽。
        self.f = torch.nn.Linear(hidden_dim, discrete_action_i_dim*10)   # 连续变量，那输出的就直接是计算资源的，均值方差
        self.eps = 1e-6
    def forward(self,x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        bandwidth = F.softmax(self.bandwidth(x), dim=1)
        f = F.softmax(self.f(x), dim=1)
        return bandwidth,f

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, discrete_action_i_dim, actor_lr=1e-6,
                 critic_lr=1e-4,lmbda=0.95, epochs=10, eps=0.2, gamma=0.98, device=torch.device("cpu")):
        self.actor = PolicyNet(state_dim, discrete_action_i_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma # 计算td的discount factor
        self.lmbda = lmbda # 计算advantage用
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps # PPO中截断范围的参数
        self.device = device
        self.actor_loss = []
        self.critic_loss = []
    def take_action(self, state,stepj,env):
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        old_b, old_f = self.actor(state)
        # mask = np.ones(10)
        # mask[env.relize_bandwidth[stepj]:] = 0
        # old_b = old_b * torch.tensor(mask).to(self.device)
        # old_b = old_b / old_b.sum()
        b = torch.distributions.Categorical(old_b).sample()
        f = torch.distributions.Categorical(old_f).sample()
        #f = f.clamp(0.1, 1.0)
        action = np.array([np.array(b.item()), np.array(f.item())])

        return action.reshape(-1)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float32).to(
            self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float32).view(-1, 1).to(self.device)
        td_target = rewards.unsqueeze(1) + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)
        # 优势归一化？

        actions_b = actions[:,0].to(int).reshape(50,1)
        actions_f = actions[:,1].to(int).reshape(50,1)

        bb,ff = self.actor(states)
        old_log_probs_b = torch.log(bb.gather(1,actions_b)).detach()
        old_log_probs_f = torch.log(ff.gather(1,actions_f)).detach()

        for _ in range(self.epochs):
            bb,ff = self.actor(states)
            log_probs_b = torch.log(bb.gather(1,actions_b))
            log_probs_f = torch.log(ff.gather(1,actions_f))

            ratio_i = torch.exp(log_probs_b - old_log_probs_b)
            surr1_i = ratio_i * advantage
            surr2_i = torch.clamp(ratio_i, 1 - self.eps,
                                  1 + self.eps) * advantage  # 截断i
            actor_loss_b = torch.mean(-torch.min(surr1_i, surr2_i))  # PPO损失函数

            ratio_f_i = torch.exp(log_probs_f - old_log_probs_f)
            surr1_f_i = ratio_f_i * advantage
            surr2_f_i = torch.clamp(ratio_f_i, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断i
            actor_loss_f = torch.mean(-torch.min(surr1_f_i, surr2_f_i))  # PPO损失函数


            actor_loss = alpha_1 * actor_loss_b + alpth_2 * actor_loss_f
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_loss.append(actor_loss.item())
            self.critic_loss.append(critic_loss.item())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float32)

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 作用应该是将所有eposide分成10 turn，用tqdm来表示进度
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.init_obs()
                done = False
                stepi = 0
                stepj = 0
                while not done:
                    action = agent.take_action(state,stepj,env)
                    # action_ = action
                    action_= np.array([action[0],(((action[1]-50)/25)+0.1)])
                    next_state, reward, done, stepi, stepj = env.step(action_, stepi, stepj)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    rewards_list.append(reward)
                    state = next_state
                    episode_return += reward
                print("this eposide is end")
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_client_prob(actions, action_probs, device):
    # actions: [batch_size, num_actions] (e.g., [3, 2])
    # action_probs: [batch_size, num_possible_actions] (e.g., [3, 4])
    log_probs = []
    size,choose_num = actions.shape
    mask = torch.ones_like(action_probs)
    for j in range(size):
        t_probs = []
        for i in range(choose_num):
            masked_probs = action_probs[j] * mask[j]
            total_prob = masked_probs.sum()
            normalized_probs = masked_probs / total_prob
            m = torch.distributions.Categorical(normalized_probs)
            log_prob = m.log_prob(actions[j][i])
            t_probs.append(log_prob)
            # 更新掩码，排除已抽样的元素
            mask[j][actions[j][i]] = 0
        log_probs.append(sum(t_probs))
    log_probs = torch.tensor(log_probs)
    return log_probs.unsqueeze(dim=1).to(device)
def compute_server_prob(actions, action_probs, device):
    # actions: [batch_size, num_actions] (e.g., [3, 2])
    # action_probs: [batch_size, num_possible_actions] (e.g., [3, 4])
    log_probs = []
    size,choose_num = actions.shape
    mask = torch.ones_like(action_probs)
    for j in range(size):
        t_probs = []
        for i in range(choose_num):
            masked_probs = action_probs[j] * mask[j]
            total_prob = masked_probs.sum()
            normalized_probs = masked_probs / total_prob
            m = torch.distributions.Categorical(normalized_probs)
            log_prob = m.log_prob(actions[j][i])
            t_probs.append(log_prob)
            # 更新掩码，排除已抽样的元素
            # mask[j][actions[j][i]] = 0
        log_probs.append(sum(t_probs))
    log_probs = torch.tensor(log_probs)
    return log_probs.unsqueeze(dim=1).to(device)

# device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True
#     print(torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    device_fl = torch.device("cuda")
elif torch.backends.mps.is_available():
    device_fl = torch.device("mps")
else:
    device_fl = torch.device("cpu")

device = torch.device("cpu")

# 为CuDNN设置确定性选项（这会影响卷积操作的速度和确定性）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



rewards_list = []
total_costs = []

eposide_time = []  # [[2,3,4],[3],[4,6]] 表示3个episode中每个action（联邦的第一个1，2，3，轮次）的time
eposide_energy = []  # 同上
eposide_acc = []  # 同上

# 注意Loss中的的权重参数和常量缩放。
#

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 100  # 强化学习总轮次


alpha_1 = 1  # 用于调整actor_loss的权重
alpth_2 = 1  # 用于调整actor_loss的权重


num_clients = 11
num_servers = 5
frac = 0.6  # 参与的client比率
test_size = 0.2  # 测试集大小
per = 0.2  # 分配到每个client的比例，暂时没用。用平均分配
min_sample_train_p = 0.2  # 每轮client i训练抽样的最小值
#  数据 get_quality 的抽样在函数内部定义，默认为0.2
#  注意acc的设置，超过acc作为跳出episode条件
choose_num = int(frac * num_clients)
discrete_action_dim = 10
state_dim = 5
env = Env()
agent = PPO(state_dim, discrete_action_dim)
return_list = train_on_policy_agent(env, agent, num_episodes)

import pickle
# with open('plt_cache/PPO_actor_loss.pkl', 'wb') as al:
#     pickle.dump(agent.actor_loss, al)
# with open('plt_cache/PPO_critic_loss.pkl', 'wb') as cl:
#     pickle.dump(agent.critic_loss, cl)

# 保存到文件
# with open('plt_cache/PPO_return_list.pkl', 'wb') as r:
#     pickle.dump(return_list, r)

# with open('plt_cache/PPO_train_time.pkl', 'wb') as t:
#     pickle.dump(env.train_time, t)
# with open('plt_cache/PPO_fed_time.pkl', 'wb') as f:
#     pickle.dump(env.fed_time, f)
# with open('plt_cache/PPO_train_energy.pkl', 'wb') as e:
#     pickle.dump(env.train_energy, e)
# with open('plt_cache/PPO_acc_list.pkl', 'wb') as c:
#     pickle.dump(env.acc_list, c)

env_name = "myenv"
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

l1 = list(range(len(agent.critic_loss)))
plt.plot(l1, agent.critic_loss)
plt.xlabel('Episodes')
plt.ylabel('criticloss')
plt.title('PPO on {}'.format(env_name))
plt.show()

plt.plot(l1, agent.actor_loss)
plt.xlabel('Episodes')
plt.ylabel('actorloss')
plt.title('PPO on {}'.format(env_name))
plt.show()
# rewards_len = list(range(len(rewards_list)))
# plt.plot(rewards_len, rewards_list)
# plt.xlabel('Episodes')
# plt.ylabel('rewards')
# plt.title('PPO on {}'.format(env_name))
# plt.show()
