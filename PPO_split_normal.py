import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pythonProject import rl_utils
from env_splitpoint import Env
import torch
import random
from tqdm import tqdm
import config
import os

torch.autograd.set_detect_anomaly(True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RunningMeanStd:
    ''' 动态均值标准差，适用于 state 的归一化 '''
    def __init__(self, shape, epsilon=1e-4, device='cpu'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.device = device

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = M2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-6)


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
        self.bandwidth = torch.nn.Linear(hidden_dim, discrete_action_i_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, 1)
        self.fc_std = torch.nn.Linear(hidden_dim, 1)
        self.eps = 1e-6

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        bandwidth_logits = self.bandwidth(x)
        mean = torch.sigmoid(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 1e-3
        return bandwidth_logits, mean, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(self, state_dim, discrete_action_i_dim, actor_lr, critic_lr, epochs, lmbda=0.95, eps=0.2, gamma=0.98, device=torch.device("cpu")):
        self.actor = PolicyNet(state_dim, discrete_action_i_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.state_rms = RunningMeanStd(shape=state_dim, device=device)
        self.actor_loss = []
        self.critic_loss = []

    def take_action(self, state, stepj, env):
        state = torch.tensor(np.array([state]), dtype=torch.float32).to(self.device)
        self.state_rms.update(state)
        norm_state = self.state_rms.normalize(state)

        b_logits, mean, std = self.actor(norm_state.to(self.device))
        mask = mask_actions(10, state[0])
        b_logits_masked = b_logits + ((-1e6) * torch.tensor(mask, dtype=torch.float32)).to(self.device)
        b_probs = F.softmax(b_logits_masked, dim=1)
        discrete_action = torch.distributions.Categorical(b_probs).sample()

        mean_cpu = mean.to('cpu')
        std_cpu = std.to('cpu')
        normal_dist = torch.distributions.Normal(mean_cpu, std_cpu)
        continuous_sample = normal_dist.sample()
        if torch.isnan(continuous_sample).any():
            continuous_sample = mean_cpu
        continuous_sample = continuous_sample.clamp(0, 1)

        continuous_sample_unnorm = continuous_sample.item()
        continuous_sample_scaled = (continuous_sample * state[0][3]).to(self.device)

        action = np.array([np.array(discrete_action.item()), np.array(continuous_sample_scaled.item())])
        return action.reshape(-1), continuous_sample_unnorm

    def update(self, transition_dict, env_client):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32).to(self.device)
        self.state_rms.update(states)
        norm_states = self.state_rms.normalize(states)
        norm_next_states = self.state_rms.normalize(next_states)

        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float32).view(-1, 1).to(self.device)
        td_target = rewards.unsqueeze(1) + self.gamma * self.critic(norm_next_states) * (1 - dones)
        td_delta = td_target - self.critic(norm_states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta).to(self.device)

        actions_b = actions[:, 0].to(int).reshape(env_client, 1)
        actions_f_scaled = actions[:, 1].reshape(env_client, 1)
        actions_f_unnorm = actions[:, 2].reshape(env_client, 1)

        orgin_states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        bb, mu, st = self.actor(norm_states)

        masks = [mask_actions(10, orgin_states[i]) for i in range(env_client)]
        b_logic_mask = bb + ((-1e6) * torch.tensor(np.array(masks), dtype=torch.float32).to(self.device))
        old_b_m = F.softmax(b_logic_mask, dim=1)
        old_log_probs_b = torch.log(old_b_m.gather(1, actions_b)).detach()

        old_normal_dist = torch.distributions.Normal(mu.detach(), st.detach())
        sampled_f_old = actions_f_unnorm.clamp(0, 1)
        old_log_probs_f = old_normal_dist.log_prob(sampled_f_old).detach()
        old_log_probs_f = torch.nan_to_num(old_log_probs_f, nan=0.0)

        for _ in range(self.epochs):
            bb, mu, st = self.actor(norm_states)
            b_logic_mask = bb + ((-1e6) * torch.tensor(np.array(masks), dtype=torch.float32).to(self.device))
            b_m = F.softmax(b_logic_mask, dim=1)

            normal_dist = torch.distributions.Normal(mu, st)
            sampled_f = actions_f_unnorm.clamp(0, 1)
            log_probs_f = normal_dist.log_prob(sampled_f)
            log_probs_f = torch.nan_to_num(log_probs_f, nan=0.0)

            log_probs_b = torch.log(b_m.gather(1, actions_b))

            ratio_i = torch.exp(log_probs_b - old_log_probs_b)
            ratio_f_i = torch.exp(log_probs_f - old_log_probs_f)

            surr1_i = ratio_i * advantage
            surr2_i = torch.clamp(ratio_i, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss_b = torch.mean(-torch.min(surr1_i, surr2_i))

            surr1_f_i = ratio_f_i * advantage
            surr2_f_i = torch.clamp(ratio_f_i, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss_f = torch.mean(-torch.min(surr1_f_i, surr2_f_i))

            actor_loss = config.alpha_1 * actor_loss_b + config.alpth_2 * actor_loss_f
            critic_loss = torch.mean(F.mse_loss(self.critic(norm_states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
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
    t_list = []
    pho_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                client_quit_bool = False
                server_quit_bool = False

                episode_return = 0
                episode_t = 0
                episode_pho = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.init_inf_obs(client_quit_bool, server_quit_bool)
                done = False
                stepi = 0
                stepj = 0

                while not done:
                    # --- 修改：take_action返回两个量 (动作数组，归一化的连续动作) ---
                    action, f_unnorm = agent.take_action(state, stepj, env)
                    next_state, reward, done, stepi, stepj, t, pho = env.step(action, stepi, stepj)

                    # --- 修改：保存动作时，额外保存归一化连续动作 f_unnorm ---
                    action_combined = np.concatenate((action, [f_unnorm]))

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action_combined)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    rewards_list.append(reward)
                    state = next_state
                    episode_return += reward
                    episode_t += t
                    episode_pho += pho

                return_list.append(-episode_return)
                t_list.append(episode_t)
                pho_list.append(episode_pho)

                # --- 传入改后的transition_dict ---
                actor_loss, critic_loss = agent.update(transition_dict, len(env.client_data['location_x']))
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': f'{num_episodes // 10 * i + i_episode + 1}',
                                      'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)

    return return_list, actor_loss_list, critic_loss_list, t_list, pho_list


# device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")
# if config.dynamic_env==False:
SEED = config.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

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
discrete_action_dim = config.action_dim
state_dim = config.state_dim
env = Env()
agent = PPO(state_dim, discrete_action_dim,actor_lr,critic_lr,epochs)
return_list,actor_loss_list,critic_loss_list,t_list,pho_list = train_on_policy_agent(env, agent, num_episodes)

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
# return_list = [i / 50 for i in return_list]
plt.plot(episodes_list, return_list)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()
mv_return = rl_utils.moving_average(return_list, 19)
plt.plot(episodes_list, mv_return)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

# r_list = list(range(len(rewards_list)))
# plt.plot(r_list, rewards_list)
# plt.xlabel('Communication')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()
# mv_reward = rl_utils.moving_average(rewards_list, 9)
# plt.plot(r_list, mv_reward)
# plt.xlabel('Communication')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()

l1 = list(range(len(critic_loss_list)-100))
plt.plot(l1, critic_loss_list[100:])
plt.xlabel('Episodes')
plt.ylabel('criticloss')
plt.title('PPO on {}'.format(env_name))
plt.show()

plt.plot(l1, actor_loss_list[100:])
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

import pickle
with open(f'response_data/normal/return_5i.pkl','wb') as f:
    pickle.dump(return_list,f)
with open(f'response_data/normal/time_5i.pkl','wb') as f:
    pickle.dump(t_list,f)
with open(f'response_data/normal/error_5i.pkl', 'wb') as f:
    pickle.dump(pho_list, f)