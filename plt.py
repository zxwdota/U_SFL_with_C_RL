import numpy as np
import pickle
import matplotlib.pyplot as plt
import rl_utils
import config


num = config.f



with open(f'INF_PPO_return_{num}f.pkl', 'rb') as f:
    a = pickle.load(f)
with open(f'INF_DQN_return_{num}f.pkl', 'rb') as f:
    b = pickle.load(f)
with open(f'INF_DDPG_return_{num}f.pkl', 'rb') as f:
    c = pickle.load(f)
# 假设 a, b, c 是已有的列表
episodes_list_a = list(range(len(a)))
episodes_list_b = list(range(len(b)))
episodes_list_c = list(range(len(c)))

mv_return_1 = rl_utils.moving_average(a, 9)
mv_return_2 = rl_utils.moving_average(b, 9)
mv_return_3 = rl_utils.moving_average(c, 9)
# 设置图形大小
plt.figure(figsize=(10, 8))
# 绘制曲线
plt.plot(episodes_list_a, mv_return_1, label='Agent PPO Returns', linewidth=2, alpha=0.8)
plt.plot(episodes_list_b, mv_return_2, label='Agent DQN Returns', linewidth=2, alpha=0.8)
plt.plot(episodes_list_c, mv_return_3, label='Agent DDPG Returns', linewidth=2, alpha=0.8)
# 添加标题和轴标签
plt.title(f'HPPO DQN DDPG Performance on env_{config.num_clients}client', fontsize=16)
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Returns', fontsize=14)

# 添加图例
plt.legend(fontsize=12, loc='best')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 展示图形
plt.tight_layout()  # 自动调整子图布局
plt.show()



DQN_reward_list = []
PPO_reward_list = []
DDPG_reward_list = []

for i in range(30, 71, 10):
    with open(f'DQN_{i}.pkl', 'rb') as f:
        DQN_reward_list.append((np.array(pickle.load(f)))/i)
for i in range(30, 71, 10):
    with open(f'PPO_return_{i}.pkl', 'rb') as f:
        PPO_reward_list.append((np.array(pickle.load(f))*(-1))/i)
for i in range(30, 71, 10):
    with open(f'DDPG_gunbel_{i}.pkl', 'rb') as f:
        DDPG_reward_list.append((np.array(pickle.load(f)))/i)

DQN_reward_avg = [np.average(DQN_reward_list[i][400:]) for i in range(5)]
PPO_reward_avg = [np.average(PPO_reward_list[i][400:]) for i in range(5)]
DDPG_reward_avg = [np.average(DDPG_reward_list[i][400:]) for i in range(5)]

episodes_list_DQN_reward_avg = list(range(30,71,10))
episodes_list_PPO_reward_avg = list(range(30,71,10))
episodes_list_DDPG_reward_avg = list(range(30,71,10))

plt.figure(figsize=(10, 8))
plt.plot(episodes_list_PPO_reward_avg, PPO_reward_avg, label='PPO', linewidth=2, alpha=0.8)
plt.plot(episodes_list_DQN_reward_avg, DQN_reward_avg, label='DQN', linewidth=2, alpha=0.8)
plt.plot(episodes_list_DDPG_reward_avg, DDPG_reward_avg, label='DDPG', linewidth=2, alpha=0.8)

# 添加标题和轴标签
plt.title(f'HPPO DQN DDPG Performance on different client', fontsize=16)
plt.xlabel('client', fontsize=14)
plt.ylabel('Returns', fontsize=14)

# 添加图例
plt.legend(fontsize=12, loc='best')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 展示图形
plt.tight_layout()  # 自动调整子图布局
plt.show()





with open(f'PPO_t_50.pkl', 'rb') as f:
    ta = pickle.load(f)
with open(f'DQN_t_50.pkl', 'rb') as f:
    tb = pickle.load(f)
with open(f'DDPG_t_50.pkl', 'rb') as f:
    tc = pickle.load(f)

# 假设 a, b, c 是已有的列表
episodes_list_a = list(range(len(ta)))
episodes_list_b = list(range(len(tb)))
episodes_list_c = list(range(len(tc)))

# mv_return_1 = rl_utils.moving_average(a, 9) * (-1)
# mv_return_2 = rl_utils.moving_average(b, 9)
# mv_return_3 = rl_utils.moving_average(c, 9)
# 设置图形大小
plt.figure(figsize=(10, 8))
# 绘制曲线
plt.plot(episodes_list_a, ta, label='Agent PPO t', linewidth=2, alpha=0.8)
plt.plot(episodes_list_b, tb, label='Agent DQN t', linewidth=2, alpha=0.8)
plt.plot(episodes_list_c, tc, label='Agent DDPG t', linewidth=2, alpha=0.8)
# 添加标题和轴标签
plt.title(f'HPPO DQN DDPG Performance on env_{config.num_clients}client', fontsize=16)
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('t', fontsize=14)
# 添加图例
plt.legend(fontsize=12, loc='best')
# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # 自动调整子图布局
plt.show()


with open(f'PPO_pho_50.pkl', 'rb') as f:
    ta = pickle.load(f)
with open(f'DQN_pho_50.pkl', 'rb') as f:
    tb = pickle.load(f)
with open(f'DDPG_pho_50.pkl', 'rb') as f:
    tc = pickle.load(f)

# 假设 a, b, c 是已有的列表
episodes_list_a = list(range(len(ta)))
episodes_list_b = list(range(len(tb)))
episodes_list_c = list(range(len(tc)))

# mv_return_1 = rl_utils.moving_average(a, 9) * (-1)
# mv_return_2 = rl_utils.moving_average(b, 9)
# mv_return_3 = rl_utils.moving_average(c, 9)
# 设置图形大小
plt.figure(figsize=(10, 8))
# 绘制曲线
plt.plot(episodes_list_a, ta, label='Agent PPO pho', linewidth=2, alpha=0.8)
plt.plot(episodes_list_b, tb, label='Agent DQN pho', linewidth=2, alpha=0.8)
plt.plot(episodes_list_c, tc, label='Agent DDPG pho', linewidth=2, alpha=0.8)
# 添加标题和轴标签
plt.title(f'HPPO DQN DDPG Performance on env_{config.num_clients}client', fontsize=16)
plt.xlabel('Eposides', fontsize=14)
plt.ylabel('pho', fontsize=14)
# 添加图例
plt.legend(fontsize=12, loc='best')
# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # 自动调整子图布局
plt.show()