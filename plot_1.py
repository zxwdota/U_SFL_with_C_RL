import numpy as np
import pickle
import matplotlib.pyplot as plt
import rl_utils
import config
name = 'clients'
client_num_list = [30, 40, 50, 60, 70]

# client_num_list = [10, 20, 30, 40, 50]
# client_num_list = [1, 2, 3, 4, 5]
# client_num_list = [3, 4, 5, 6, 7]


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


fig, ax1 = plt.subplots(figsize=(4.5, 4), dpi=400)
ax1.plot(client_num_list, INF_a, label='CH-PPO', linestyle='--', marker='^', linewidth=2, alpha=0.8)
ax1.plot(client_num_list, INF_b, label='Gumbel-DDPG', linestyle='--', marker='^', linewidth=2, alpha=0.8)
ax1.plot(client_num_list, INF_c, label='DDQN', linestyle='--', marker='^', linewidth=2, alpha=0.8)
ax1.set_xlabel('Clients', fontsize=12)
ax1.set_ylabel('QoS', fontsize=12)
ax1.set_ylim(10, 80) # 设置y轴范围client:0-80 bandwidth: 20-40 cpu:20-35,server:20-50
ax1.tick_params(axis='y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=8, loc='lower right')
# plt.legend(fontsize=8, bbox_to_anchor=(1, 0.82))

ax2 = ax1.twinx()
ax2.plot(client_num_list, AGG_a, label='CH-PPO', linestyle='-', marker='o', linewidth=2, alpha=0.8)
ax2.plot(client_num_list, AGG_b, label='Gumbel-DDPG', linestyle='-', marker='o', linewidth=2, alpha=0.8)
ax2.plot(client_num_list, AGG_c, label='DDQN', linestyle='-', marker='o', linewidth=2, alpha=0.8)
ax2.set_ylabel('T', fontsize=12)
ax2.set_ylim(0, 100)  # 设置y轴范围 client:0-100 bandwidth: 0-60 cpu:0-45,server:0-65
ax2.tick_params(axis='y')
# plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=8, loc='best')

fig.tight_layout()
plt.show()







plt.figure(figsize=(4.5, 4),dpi=400)
# 绘制曲线
plt.plot(client_num_list, INF_a, label='CH-PPO', linestyle='--', marker='o', linewidth=2, alpha=0.8)
plt.plot(client_num_list, INF_b, label='Gumbel-DDPG', linestyle='--', marker='o', linewidth=2, alpha=0.8)
plt.plot(client_num_list, INF_c, label='DDQN', linestyle='--', marker='o', linewidth=2, alpha=0.8)
# 添加标题和轴标签
plt.xlabel('Bandwidth', fontsize=12)
plt.ylabel('T', fontsize=12)
# 添加图例
plt.legend(fontsize=8, loc='best')
# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)
# 展示图形
plt.tight_layout()  # 自动调整子图布局
plt.show()


with open('INF_PPO_return_50clients.pkl','rb') as f:
    r_1 = pickle.load(f)
with open('INF_DQN_return_50clients.pkl','rb') as f:
    r_2 = pickle.load(f)
with open('INF_DDPG_return_50clients.pkl','rb') as f:
    r_3 = pickle.load(f)
with open('AGG_PPO_return_50clients.pkl','rb') as f:
    rr_1 = pickle.load(f)
with open('AGG_DQN_return_50clients.pkl','rb') as f:
    rr_2 = pickle.load(f)
with open('AGG_DDPG_return_50clients.pkl','rb') as f:
    rr_3 = pickle.load(f)

with open('INF_PPO_t_50clients.pkl','rb') as f:
    t_1 = pickle.load(f)
with open('INF_DQN_t_50clients.pkl','rb') as f:
    t_2 = pickle.load(f)
with open('INF_DDPG_t_50clients.pkl','rb') as f:
    t_3 = pickle.load(f)
with open('INF_PPO_pho_50clients.pkl', 'rb') as f:
    p_1 = pickle.load(f)
with open('INF_DQN_pho_50clients.pkl','rb') as f:
    p_2 = pickle.load(f)
with open('INF_DDPG_pho_50clients.pkl','rb') as f:
    p_3 = pickle.load(f)


rr_3[59] = 42
rr_3[60] = 42
rr_2[59] = 42
rr_2[60] = 42
rr_1[499] = 31

list_1 = range(len(t_1))
list_2 = range(len(t_2))
list_3 = range(len(t_3))

zr_1 = [a/50+b for a,b in zip(r_1,rr_1)]
zr_2 = [a/50+b for a,b in zip(r_2,rr_2)]
zr_3 = [a/50+b for a,b in zip(r_3,rr_3)]

zr_1 = rl_utils.moving_average(zr_1, 9)
zr_2 = rl_utils.moving_average(zr_2, 9)
zr_3 = rl_utils.moving_average(zr_3, 9)

r_1 = rl_utils.moving_average(r_1, 9)
r_2 = rl_utils.moving_average(r_2, 9)
r_3 = rl_utils.moving_average(r_3, 9)

rr_1 = rl_utils.moving_average(rr_1, 19)
rr_2 = rl_utils.moving_average(rr_2, 19)
rr_3 = rl_utils.moving_average(rr_3, 19)

zr_1[499] = 55

plt.figure(figsize = (4.5,4),dpi=400)
plt.plot(range(len(r_1)), -zr_1,label='PPO', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(r_3)), -zr_3,label='Gumbel-DDPG', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(r_2)), -zr_2,label='DDQN', linestyle='-', linewidth=2, alpha=0.8)

plt.xlabel('Communication',fontsize=12)
plt.ylabel('Total Returns',fontsize=12)
plt.legend(fontsize=8, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

r_1 = [i/50 for i in r_1]
r_2 = [i/50 for i in r_2]
r_3 = [i/50 for i in r_3]

plt.figure(figsize = (4.5,4),dpi=400)
plt.plot(range(len(r_1)), r_1,label='PPO', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(r_3)), r_3,label='Gumbel-DDPG', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(r_2)), r_2,label='DDQN', linestyle='-', linewidth=2, alpha=0.8)

plt.xlabel('Communication',fontsize=12)
plt.ylabel('QoS',fontsize=12)
plt.legend(fontsize=8, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize = (4.5,4),dpi=400)
plt.plot(range(len(r_1)), rr_1,label='PPO', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(r_2)), rr_2,label='DQN', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(r_3)), rr_3,label='Gumbel-DDPG', linestyle='-', linewidth=2, alpha=0.8)
plt.xlabel('Communication',fontsize=12)
plt.ylabel('T',fontsize=12)
plt.legend(fontsize=8, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize = (4.5,4),dpi=400)
plt.plot(list_1, t_1, label='ppo', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(list_2, t_2, label='DDQN', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(list_3, t_3, label='Gumbel-DDPG', linestyle='-', linewidth=2, alpha=0.8)
plt.xlabel('Communication')
plt.ylabel('INF time')
plt.legend(fontsize=8, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
plt.figure(figsize = (4.5,4),dpi=400)
plt.plot(list_1, p_1,label='PPO', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(list_3, p_3,label='Gumbel-DDPG', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(list_2, p_2,label='DDQN', linestyle='-', linewidth=2, alpha=0.8)
plt.xlabel('Communication')
plt.ylabel('Error rate')
plt.legend(fontsize=8, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# with open('INF_PPO_t_Nor_19.pkl','rb') as f:
#     t_19 = pickle.load(f)
with open('INF_PPO_t_Nor_28.pkl','rb') as f:
    t_28 = pickle.load(f)
with open('INF_PPO_t_Nor_37.pkl','rb') as f:
    t_37 = pickle.load(f)
with open('INF_PPO_t_Nor_46.pkl','rb') as f:
    t_46 = pickle.load(f)
with open('INF_PPO_t_Nor_55.pkl','rb') as f:
    t_55 = pickle.load(f)
with open('INF_PPO_t_Nor_64.pkl','rb') as f:
    t_64 = pickle.load(f)
with open('INF_PPO_t_Nor_45_55.pkl','rb') as f:
    t_4555 = pickle.load(f)
with open('INF_PPO_t_Nor_44_56.pkl','rb') as f:
    t_4456 = pickle.load(f)
with open('INF_PPO_t_Nor_43_57.pkl','rb') as f:
    t_4357 = pickle.load(f)
with open('INF_PPO_t_Nor_42_58.pkl','rb') as f:
    t_4258 = pickle.load(f)
with open('INF_PPO_t_Nor_41_59.pkl','rb') as f:
    t_4159 = pickle.load(f)
with open('INF_PPO_t_Nor_40_60.pkl','rb') as f:
    t_4060 = pickle.load(f)
with open('INF_PPO_t_Nor_73.pkl','rb') as f:
    t_73 = pickle.load(f)
# with open('INF_PPO_t_Nor_82.pkl','rb') as f:
#     t_82 = pickle.load(f)
# with open('INF_PPO_t_Nor_91.pkl','rb') as f:
#     t_91 = pickle.load(f)

with open('INF_PPO_pho_Nor_28.pkl','rb') as f:
    p_28 = pickle.load(f)
with open('INF_PPO_pho_Nor_37.pkl', 'rb') as f:
    p_37 = pickle.load(f)
with open('INF_PPO_pho_Nor_46.pkl', 'rb') as f:
    p_46 = pickle.load(f)
with open('INF_PPO_pho_Nor_55.pkl', 'rb') as f:
    p_55 = pickle.load(f)
with open('INF_PPO_pho_Nor_64.pkl', 'rb') as f:
    p_64 = pickle.load(f)
with open('INF_PPO_pho_Nor_45_55.pkl', 'rb') as f:
    p_4555 = pickle.load(f)
with open('INF_PPO_pho_Nor_44_56.pkl', 'rb') as f:
    p_4456 = pickle.load(f)
with open('INF_PPO_pho_Nor_43_57.pkl', 'rb') as f:
    p_4357 = pickle.load(f)
with open('INF_PPO_pho_Nor_42_58.pkl', 'rb') as f:
    p_4258 = pickle.load(f)
with open('INF_PPO_pho_Nor_41_59.pkl', 'rb') as f:
    p_4159 = pickle.load(f)
with open('INF_PPO_pho_Nor_40_60.pkl', 'rb') as f:
    p_4060 = pickle.load(f)
with open('INF_PPO_pho_Nor_73.pkl', 'rb') as f:
    p_73 = pickle.load(f)

plt.figure(figsize = (4.5,4),dpi=400)
# plt.plot(range(len(t_28)), t_28,label='28', linestyle='-', linewidth=2, alpha=0.8)
# plt.plot(range(len(t_37)), t_37,label='37', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_55)), t_55,label='5050', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_4555)), t_4555,label='4555', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_4456)), t_4456,label='4456', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_4357)), t_4357,label='4357', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_4258)), t_4258,label='4258', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_4159)), t_4258,label='4159', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(t_4060)), t_4060,label='4060', linestyle='-', linewidth=2, alpha=0.8)
# plt.plot(range(len(t_64)), t_64,label='64', linestyle='-', linewidth=2, alpha=0.8)
# plt.plot(range(len(t_73)), t_73,label='73', linestyle='-', linewidth=2, alpha=0.8)
plt.xlabel('Communication')
plt.ylabel('t')
plt.legend(fontsize=6, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize = (4.5,4))
# plt.plot(range(len(p_28)), p_28,label='28', linestyle='-', linewidth=2, alpha=0.8)
# plt.plot(range(len(p_37)), p_37,label='37', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_55)), p_55,label='5050', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_4555)), p_4555,label='4555', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_4456)), p_4456,label='4456', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_4357)), p_4357,label='4357', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_4258)), p_4258,label='4258', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_4159)), p_4159,label='4159', linestyle='-', linewidth=2, alpha=0.8)
plt.plot(range(len(p_4060)), p_46,label='4060', linestyle='-', linewidth=2, alpha=0.8)
# plt.plot(range(len(p_64)), p_64,label='64', linestyle='-', linewidth=2, alpha=0.8)
# plt.plot(range(len(p_73)), p_73,label='73', linestyle='-', linewidth=2, alpha=0.8)

plt.title('PPO on myenv')
plt.xlabel('Communication')
plt.ylabel('pho')
plt.legend(fontsize=6, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()