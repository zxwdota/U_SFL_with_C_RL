import numpy as np
import pickle
import matplotlib.pyplot as plt
import rl_utils
import config
name = 'clients'
pltname = 'Clients'
client_num_list = [30, 40, 50, 60, 70]
y1 = (10,80)
y2 = (0,100)

name = 'bandwidth'
pltname = 'Bandwidth'
client_num_list = [10, 20, 30, 40, 50]
y1 = (20,45)
y2 = (25,55)

name = 'f'
pltname = 'Cpu'
client_num_list = [1, 2, 3, 4, 5]
y1 = (22,35)
y2 = (25,60)


name = 'server'
pltname = 'Servers'
client_num_list = [3, 4, 5, 6, 7]
y1 = (20,45)
y2 = (25,55)

color_1 = (84,97,166)
color_2 = (93,157,180)
color_3 = (152,205,168)

color_1 = (84/255, 97/255, 166/255)
color_2 = (93/255, 157/255, 180/255)
color_3 = (152/255, 205/255, 168/255)


INF_ddpg_list = [f'INF_DDPG_return_{num}{name}.pkl' for num in client_num_list]
INF_ppo_list = [f'INF_PPO_return_{num}{name}.pkl' for num in client_num_list]
INF_dqn_list = [f'INF_DQN_return_{num}{name}.pkl' for num in client_num_list]
INF_ppo = []
INF_ddpg = []
INF_dqn = []

AGG_ddpg_list = [f'AGG_DDPG_return_{num}{name}.pkl' for num in client_num_list]
AGG_ppo_list = [f'AGG_PPO_return_{num}{name}.pkl' for num in client_num_list]
AGG_dqn_list = [f'AGG_DQN_return_{num}{name}.pkl' for num in client_num_list]
AGG_ppo = []
AGG_ddpg = []
AGG_dqn = []

for i in range(5):
    with open(INF_ppo_list[i], 'rb') as f:
        INF_ppo.append(pickle.load(f))
    with open(INF_ddpg_list[i], 'rb') as f:
        INF_ddpg.append(pickle.load(f))
    with open(INF_dqn_list[i], 'rb') as f:
        INF_dqn.append(pickle.load(f))

for i in range(5):
    with open(AGG_ppo_list[i], 'rb') as f:
        AGG_ppo.append(pickle.load(f))
    with open(AGG_ddpg_list[i], 'rb') as f:
        AGG_ddpg.append(pickle.load(f))
    with open(AGG_dqn_list[i], 'rb') as f:
        AGG_dqn.append(pickle.load(f))

# with open(ppo_list[0],'rb') as f:
#     ppo.append(pickle.load(f))
# with open(ddpg_list[0],'rb') as f:
#     ddpg.append(pickle.load(f))
# with open(dqn_list[0],'rb') as f:
#     dqn.append(pickle.load(f))
INF_a = []
INF_b = []
INF_c = []
for i in range(5):
    INF_c.append(np.mean(INF_dqn[i][-100:])/50)
    INF_b.append(np.mean(INF_ddpg[i][-100:])/50)
    INF_a.append(np.mean(INF_ppo[i][-100:])/50)
# 假设 a, b, c 是已有的列表
episodes_list_a = list(range(len(INF_a)))
episodes_list_b = list(range(len(INF_b)))
episodes_list_c = list(range(len(INF_c)))

AGG_a = []
AGG_b = []
AGG_c = []
for i in range(5):
    AGG_c.append(np.mean(AGG_dqn[i][-100:]))
    AGG_b.append(np.mean(AGG_ddpg[i][-100:]))
    AGG_a.append(np.mean(AGG_ppo[i][-100:]))


fig, ax1 = plt.subplots(figsize=(4, 4), dpi=400)

bar_width = 0.3  # 每个柱状图的宽度
x = np.linspace(0, len(client_num_list) * 2, len(client_num_list))  # 增加簇之间的间距

# 绘制 INF 数据的柱状图（左侧 y 轴）
bars1 = ax1.bar(x - 2.5*bar_width, INF_a, bar_width, label='CHPPO (INF QoS)', alpha=0.8, edgecolor='black',linewidth=1.2,color=color_1)
bars2 = ax1.bar(x - 1.5*bar_width, INF_b, bar_width, label='Gumbel-DDPG (INF QoS)', alpha=0.8, edgecolor='black',linewidth=1.2,color=color_2)
bars3 = ax1.bar(x - 0.5*bar_width, INF_c, bar_width, label='DDQN (INF QoS)', alpha=0.8, edgecolor='black',linewidth=1.2,color=color_3)

# 设置左侧 y 轴
ax1.set_xlabel(pltname, fontsize=12)
ax1.set_ylabel('QoS', fontsize=12)
ax1.set_xticks(x)  # 设置横坐标位置
ax1.set_xticklabels(client_num_list)  # 设置横坐标标签
ax1.set_ylim(y1)  # 设置 y 轴范围
ax1.tick_params(axis='y')
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()

# 绘制 AGG 数据的柱状图（右侧 y 轴）
bars4 = ax2.bar(x + 0.5*bar_width, AGG_a, bar_width, label='CHPPO (AGG T)', alpha=0.8, hatch='///',edgecolor='black',linewidth=1.2,color=color_1)
bars5 = ax2.bar(x + 1.5*bar_width, AGG_b, bar_width, label='Gumbel-DDPG (AGG T)', alpha=0.8, hatch='///',edgecolor='black',linewidth=1.2,color=color_2)
bars6 = ax2.bar(x + 2.5*bar_width, AGG_c, bar_width, label='DDQN (AGG T)', alpha=0.8, hatch='///', edgecolor='black', linewidth=1.2,color=color_3)

# 设置右侧 y 轴
ax2.set_ylabel('T', fontsize=12)
ax2.set_ylim(y2)  # 设置 y 轴范围
ax2.tick_params(axis='y')

# for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
#     for bar in bars:
#         bar.set_linewidth(0.5)  # 设置边框粗细为2.5

# 合并图例
handles1, labels1 = ax1.get_legend_handles_labels()  # 获取 ax1 的图例信息
handles2, labels2 = ax2.get_legend_handles_labels()  # 获取 ax2 的图例信息
fig.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='upper left', bbox_to_anchor=(0.65, 0.9), ncol=1)

# 调整布局
fig.tight_layout()
plt.show()