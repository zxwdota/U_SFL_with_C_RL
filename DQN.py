import random
from env import Env
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
import config

action_space = []
for i in range(10):
    for j in range(10):
        action_space.append(np.array([i, j/10]))

def mask(state):
    action_mask = np.zeros(action_dim)
    if 0<int(state[0][4]*40):
        if int(state[0][4]*40)<10:
            action_mask[10*int(state[0][4]*40)+10:100] = 1e10
    else:
        action_mask = np.ones(action_dim)
        action_mask = action_mask*1e10
        action_mask[0:10] = 0
    return action_mask


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device
        self.loss = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            m = max(0,int(40*state[4].item()))
            action = np.random.randint(min(self.action_dim,10+10*m))
        else:
            # state = torch.tensor([state], dtype=torch.float).to(self.device)/torch.tensor([100,100,1,1,40]).to(self.device)
            act_mask = torch.tensor(mask(state.reshape(1,-1)))
            act_mask = act_mask.reshape(1,-1)
            q = self.q_net(state.reshape(1,-1)) - act_mask.detach()
            action = q.argmax().item()
        return action

    def max_q_value(self, state):
        #state = torch.tensor([state], dtype=torch.float).to(self.device)
        act_mask = torch.tensor(mask(state.reshape(1, -1)))
        act_mask = act_mask.reshape(1, -1)
        return self.q_net(state).max().item() - act_mask.detach()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN':  # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:  # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.loss.append(dqn_loss.item())
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1





def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    t_list = []
    pho_list = []
    q_loss = []
    for i in range(10):
        with (tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar):
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                episode_t = 0
                episode_pho = 0
                state = torch.tensor(env.init_obs(),dtype=torch.float32
                                     )/torch.tensor([100,100,1,1,40])
                done = False
                stepi=0
                stepj=0
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state.reshape(1,-1)) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = np.copy(action_space[action])
                    action_continuous[1] = action_continuous[1]*max(state[3],0)
                    next_state, reward, done, stepi, stepj, t, pho = env.step(action_continuous,stepi,stepj)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = torch.tensor(next_state,dtype=torch.float32)/torch.tensor([100,100,1,1,40])
                    episode_return += reward
                    episode_t += t
                    episode_pho += pho
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(-episode_return)
                t_list.append(episode_t)
                pho_list.append(episode_pho)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list, t_list, pho_list

lr = config.dqn_lr
num_episodes = config.dqn_eposides
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env = Env()
state_dim = config.state_dim
action_dim = 100  # 将连续动作分成11个离散动作

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)
return_list, max_q_value_list,t_list,pho_list = train_DQN(agent, env, num_episodes,
                                          replay_buffer, minimal_size,
                                          batch_size)
env_name = "myenv"
episodes_list = list(range(len(return_list)))
# return_list = [i / 50 for i in return_list]
mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

# frames_list = list(range(len(max_q_value_list)))
# plt.plot(frames_list, max_q_value_list)
# plt.axhline(0, c='orange', ls='--')
# plt.axhline(10, c='red', ls='--')
# plt.xlabel('Frames')
# plt.ylabel('Q value')
# plt.title('DQN on {}'.format(env_name))
# plt.show()

frames_list = list(range(len(agent.loss)))
plt.plot(frames_list, agent.loss)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('loss')
plt.title('DQN on {}'.format(env_name))
plt.show()

import pickle

with open(f'INF_DQN_return_Nor_28.pkl','wb') as f:
    pickle.dump(return_list,f)
with open(f'INF_DQN_t_Nor_28.pkl','wb') as f:
    pickle.dump(t_list,f)
with open(f'INF_DQN_pho_Nor_28.pkl', 'wb') as f:
    pickle.dump(pho_list, f)