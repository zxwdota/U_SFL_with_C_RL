import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from env_v1 import Env
import torch
import random
from tqdm import tqdm
import config

# joint 每次采样后需要调整剩余动作的概率分布。

torch.autograd.set_detect_anomaly(True)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def mask_actions(actions_space, state):
    action_mask = np.zeros(shape=actions_space)
    if state[4].to(int) < 9:
        for _ in range(state[4].to(int)+1, 10):
            action_mask[_] = 1
    return action_mask


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, discrete_action_i_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bandwidth = torch.nn.Linear(hidden_dim, discrete_action_i_dim)  # 离散变量，通过softmax选择动作。输出的是概率分布。将带宽设置为N个，每个递增的带宽。
        self.fc_mu = torch.nn.Linear(hidden_dim, 1)   # 连续变量，那输出的就直接是计算资源的，均值方差
        self.fc_std = torch.nn.Linear(hidden_dim, 1)
        self.eps = 1e-6
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_1 = self.bandwidth(x)
        x_2 = self.fc_mu(x)
        x_3 = self.fc_std(x)
        bandwidth_logic = x_1
        # bandwidth = F.softmax(x_1, dim=1)
        # mean = (F.tanh(x_2) + 1)/2
        # std = F.softplus(x_3)+self.eps
        alpha = F.softplus(x_2) + 1
        beta = F.softplus(x_3) + 1
        return bandwidth_logic, alpha, beta  # mean,std

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
    def __init__(self, state_dim, discrete_action_i_dim, actor_lr,
                 critic_lr, epochs, lmbda=0.95, eps=0.2, gamma=0.98, device=torch.device("cpu")):
        self.actor = PolicyNet(state_dim, discrete_action_i_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma  # 计算td的discount factor
        self.lmbda = lmbda  # 计算advantage用
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.actor_loss = []
        self.critic_loss = []
    def take_action(self, state,stepj,env):
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        b_logic, old_mean_f, old_std_f = self.actor(state/torch.tensor([100,100,1,1,40]).to(self.device))
        mask = mask_actions(10, state[0])
        b_logic_mask = b_logic + ((-1e6)*torch.tensor(mask,dtype=torch.float32)).to(self.device)
        old_b_m = F.softmax(b_logic_mask,dim=1)
        output_b = torch.distributions.Categorical(old_b_m).sample()
        m_cpu = old_mean_f.to('cpu')
        std_cpu = old_std_f.to('cpu')
        action_dist_f_i = torch.distributions.Beta(m_cpu, std_cpu)
        f = action_dist_f_i.sample().to(self.device)*state[0][3]
        action = np.array([np.array(output_b.item()), np.array(f.item())])

        return action.reshape(-1)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float32).to(self.device)/torch.tensor([100,100,1,1,40]).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float32).to(
            self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float32).to(self.device)/torch.tensor([100,100,1,1,40]).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float32).view(-1, 1).to(self.device)
        rewards = (rewards+100.0)/100.0
        td_target = rewards.unsqueeze(1) + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)
        # 优势归一化？

        actions_b = actions[:,0].to(int).reshape(50, 1)
        actions_f = (actions[:,1]).reshape(50, 1)


        orgin_states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float32).to(self.device)
        bb,mu,st = self.actor(states)

        masks = [mask_actions(10, orgin_states[i]) for i in range(50)]
        b_logic_mask = bb + ((-1e6) * torch.tensor(np.array(masks),dtype=torch.float32).to(self.device))
        old_b_m = F.softmax(b_logic_mask, dim=1)
        dist = torch.distributions.Beta(mu.detach(),st.detach())
        old_log_probs_b = torch.log(old_b_m.gather(1,actions_b)).detach()
        old_log_probs_f = dist.log_prob(actions_f/orgin_states[:,3].reshape(50,1)).detach()

        for _ in range(self.epochs):
            bb,mu,st = self.actor(states)
            b_logic_mask = bb + ((-1e6) * torch.tensor(np.array(masks),dtype=torch.float32).to(self.device))
            b_m = F.softmax(b_logic_mask, dim=1)

            dist = torch.distributions.Beta(mu, st)
            log_probs_b = torch.log(b_m.gather(1,actions_b))
            log_probs_f = dist.log_prob(actions_f/orgin_states[:,3].reshape(50,1))

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


            actor_loss = config.alpha_1 * actor_loss_b + config.alpth_2 * actor_loss_f
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        return actor_loss.item(), critic_loss.item()

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().to('cpu').numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float32)

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    actor_loss_list = []
    critic_loss_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:  # 作用应该是将所有eposide分成10 turn，用tqdm来表示进度
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                pho_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.init_obs()
                done = False
                stepi = 0
                stepj = 0
                time = 0
                while not done:
                    action = agent.take_action(state,stepj,env)
                    # print(state)
                    # print(action)
                    next_state, reward, done, stepi, stepj, time, pho = env.step(action, stepi, stepj, time)
                    # print(next_state)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    rewards_list.append(-reward)
                    state = next_state
                    pho_return += pho
                    episode_return = time
                # print("this eposide is end")
                return_list.append(((pho_return*1e2)+(episode_return*1e4)))
                actor_loss,critic_loss = agent.update(transition_dict)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list,actor_loss_list,critic_loss_list


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

device = torch.device("mps")

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

num_episodes = config.num_episodes  # 强化学习总轮次
hidden_dim = 128
gamma = 0.98  # 计算td
lmbda = 0.95  # 计算advantage用，见ri_utils文件
epochs = config.ppo_epochs # 对当前 episode update更新多少次
actor_lr = config.actor_lr  # actor的学习率
critic_lr = config.critic_lr  # critic的学习率
eps = 0.2  # PPO中截断范围的参数

alpha_1 = 0.5  # 用于调整actor_loss的权重
alpth_2 = 0.5  # 用于调整actor_loss的权重


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
agent = PPO(state_dim, discrete_action_dim,actor_lr,critic_lr,epochs)
return_list,actor_loss_list,critic_loss_list = train_on_policy_agent(env, agent, num_episodes)

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

r_list = list(range(len(rewards_list)))
plt.plot(r_list, rewards_list)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()
mv_reward = rl_utils.moving_average(rewards_list, 9)
plt.plot(r_list, mv_reward)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()






l1 = list(range(len(critic_loss_list)))
plt.plot(l1, critic_loss_list)
plt.xlabel('Episodes')
plt.ylabel('criticloss')
plt.title('PPO on {}'.format(env_name))
plt.show()

plt.plot(l1, actor_loss_list)
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
