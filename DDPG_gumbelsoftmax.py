import random
from env import Env
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils_rf
import rl_utils
import config


# Gumbel-Softmax Function
def gumbel_softmax_sample(logits, tau=1.0):
    """Sample from the Gumbel-Softmax distribution."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)

def gumbel_softmax(logits, tau=1.0, hard=False):
    """Compute Gumbel-Softmax and return either soft or hard one-hot samples."""
    y = gumbel_softmax_sample(logits, tau)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        return (y_hard - y).detach() + y  # Straight-through gradient trick
    return y

def mask_actions(actions_space, state):
    action_mask = np.zeros(shape=(len(state),actions_space))
    for i in range(len(state)):
        if (state[i][4]*40).to(int) < 9:
            if (state[i][4]*40) < 0:
                state[i][4] = 0
            for _ in range((state[i][4]*40).to(int) + 1, 10):
                action_mask[i][_] = 1
    return action_mask


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, discrete_action_i_dim, hidden_dim=128):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bandwidth = torch.nn.Linear(hidden_dim,
                                         discrete_action_i_dim)  # 离散变量，通过softmax选择动作。输出的是概率分布。将带宽设置为N个，每个递增的带宽。
        self.f = torch.nn.Linear(hidden_dim, 1)  # 连续变量，那输出的就直接是计算资源的，均值方差

    def forward(self, s, tau=1.0, hard=False):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mask = mask_actions(10,s)
        x_0 = self.bandwidth(x) + ((-1e9) * torch.tensor(mask, dtype=torch.float32))
        x_1 = gumbel_softmax(x_0, tau, hard)

        x_2 = ((torch.tanh(self.f(x)) + 1) / 2) * (s[:,3].reshape(-1,1))
        return x_1, x_2


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim + 1, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a_1):
        cat = torch.cat([x, a_1], dim=1)  # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, sigma, epochs, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.epochs = epochs
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.actor_loss = []
        self.critic_loss = []

    def take_action(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float32).reshape(1,-1).to(self.device)
            b_logic, f = self.actor(state / torch.tensor([100, 100, 1, 1, 40, 5, 50], dtype=torch.float32).to(self.device))
            # 给动作添加噪声，增加探索DDPG.py
            b = torch.argmax(b_logic,dim=-1).item()
            f = f.item()

            action = np.hstack([b,f])
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device) / torch.tensor(
            [100, 100, 1, 1, 40, 5, 50]).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device) / torch.tensor(
            [100, 100, 1, 1, 40, 5, 50]).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = (rewards + 100.0) / 100.0
        for _ in range(self.epochs):
            temp = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, torch.cat([temp[0], temp[1]], dim=1))
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)
            critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
            temp_2 = self.actor(states)
            actor_loss = -torch.mean(self.critic(states, torch.cat([temp_2[0], temp_2[1]], dim=1)))


            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.actor_loss.append(actor_loss.item())
            self.critic_loss.append(critic_loss.item())

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    t_list = []
    pho_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                episode_t = 0
                episode_pho = 0
                state = env.init_obs()
                done = False
                stepi = 0
                stepj = 0

                state = torch.tensor(np.concatenate([state, np.array([stepi, stepj])]))
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, stepi, stepj, t, pho = env.step(action, stepi, stepj)
                    next_state = torch.tensor(np.concatenate([next_state, np.array([stepi, stepj])]))
                    action_eye = np.eye(10)[int(action[0])]
                    action_con = np.hstack([action_eye, action[1]])
                    replay_buffer.add(state, action_con, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    episode_t += t
                    episode_pho += pho
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns,
                                           'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(-episode_return)
                t_list.append(episode_t)
                pho_list.append(episode_pho)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list, t_list, pho_list


actor_lr = config.ddpg_a_lr
critic_lr = config.ddpg_c_lr
num_episodes = config.ddpg_num_episodes
hidden_dim = 128
gamma = 0.98
tau = 0.01  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
epochs = config.ddpg_epochs
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = config.state_dim + 2
action_dim = config.action_dim
agent = DDPG(state_dim, action_dim, sigma, epochs, actor_lr, critic_lr, tau, gamma, device)

return_list,t_list,pho_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
import pickle
# with open(f'INF_DDPG_return_Nor_28.pkl', 'wb') as f:
#     pickle.dump(return_list, f)
# with open(f'INF_DDPG_t_Nor_28.pkl','wb') as f:
#     pickle.dump(t_list,f)
# with open(f'INF_DDPG_pho_Nor_28.pkl', 'wb') as f:
#     pickle.dump(pho_list, f)

# return_list = [i / 50 for i in return_list]
env_name = "myenv"
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Communication')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()
#
l1 = list(range(len(agent.critic_loss)))
plt.plot(l1, agent.critic_loss)
plt.xlabel('Episodes')
plt.ylabel('criticloss')
plt.title('DDPG on {}'.format(env_name))
plt.show()

plt.plot(l1, agent.actor_loss)
plt.xlabel('Episodes')
plt.ylabel('actorloss')
plt.title('DDPG on {}'.format(env_name))
plt.show()

