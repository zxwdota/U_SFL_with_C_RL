import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rl_utils

# 加载数据
with open('return.pkl', 'rb') as f:
    old_split_return = -np.array(pickle.load(f))[0:4999]/50
with open('return_5i.pkl', 'rb') as f:
    haed_split_return = -np.array(pickle.load(f))[0:4999]/50
with open('return.pkl', 'rb') as f:
    mid_quitserver_return = -np.array(pickle.load(f))[0:4999]/50
# 特殊处理异常点
haed_split_return[305] = haed_split_return[306]

a = np.where(old_split_return > 0)[0]  # 0-499
for idex in range(len(old_split_return)):
    if old_split_return[idex] > 0 or old_split_return[idex] < -3000/50:
        old_split_return[idex] = old_split_return[idex - 1]
    if haed_split_return[idex] > 0 or haed_split_return[idex] < -3000/50:
        haed_split_return[idex] = haed_split_return[idex - 1]
    if mid_quitserver_return[idex] > 0 or mid_quitserver_return[idex] < -3000/50:
        mid_quitserver_return[idex] = mid_quitserver_return[idex - 1]
# haed_split_return = old_split_return
# 移动平均
window_size_1 = 199
window_size_2 = 199
window_size_3 = 199
haed_split_return_mv = rl_utils.moving_average(haed_split_return, window_size_1)
old_split_return_mv = rl_utils.moving_average(old_split_return, window_size_2)
mid_quitserver_return_mv = rl_utils.moving_average(mid_quitserver_return,window_size_3)
old_split_return_mv[-5:] = old_split_return_mv[-10:-5]

# 计算滑动标准差
def moving_std(arr, window_size):
    stds = []
    half = window_size // 2
    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        stds.append(np.std(arr[start:end]))
    return np.array(stds)

fix_std = moving_std(haed_split_return, window_size_1)
free_std = moving_std(old_split_return, window_size_2)
mid_std = moving_std(mid_quitserver_return, window_size_3)

# 横轴对齐
y_mv = np.arange(len(haed_split_return_mv))

# 标注函数
def annotate_stats(y, mv,x, std, label, color):
    # 最大值
    max_idx = np.argmax(mv)
    l = len(mv)

    # 最终值
    final_idx = -1
    # plt.text(y[final_idx] - 100, mv[final_idx] + std[final_idx] + 2,
    #          f'Final: {mv[final_idx]:.1f}±{std[final_idx]:.1f}', color=color, fontsize=8)

    # 最后平均
    last_k = 2600
    last_avg = np.mean(x[-last_k:])
    last_std = np.std(x[-last_k:])
    # plt.axhline(y=converge_vals['total_time'], color=red, alpha=0.7, linestyle='--', linewidth=2)
    # plt.axhline(y=converge_vals['total_no_split_time'], color=blue, alpha=0.7, linestyle='--', linewidth=2)
    #
    # plt.text(10500, converge_vals['total_time'], f"{converge_vals['total_time']:.1f}", color=red, fontsize=12,
    #          fontweight='bold')
    # plt.text(10500, converge_vals['total_no_split_time'], f"{converge_vals['total_no_split_time']:.1f}", color=blue,
    #          fontsize=12, fontweight='bold')
    if label=='old':
        plt.text(
            y[final_idx] - 0.2 * l, last_avg +2,
            f'{last_avg:.1f}±{last_std:.1f}',
            color=color, fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', linewidth=1, alpha=0.5)
        )
        plt.axhline(
            y=last_avg,
            color='black', linestyle='dashed', linewidth=2, alpha=0.7
        )
    else:
        plt.text(
            max_idx+200, mv[max_idx] - 3,
            f'{mv[max_idx]:.1f}',
            color=color, fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white', linewidth=1, alpha=0.5)
        )
        # plt.axhline(
        #     y=last_avg,
        #     color='black', linestyle='dashed', linewidth=2, alpha=0.7
        # )
    # plt.scatter(y[max_idx], mv[max_idx], marker='^', color='red', s=40, zorder=10)
    # plt.text(y[max_idx] + 5, mv[max_idx] + 2, f'Max: {mv[max_idx]:.1f}±{std[max_idx]:.1f}',
    #          color=color, fontsize=8, fontweight='bold')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# 绘图
plt.figure(figsize=(4.5, 4), dpi=300)
plt.axes([0.17, 0.12, 0.8, 0.85])

# 曲线 + 区域
# plt.plot(y_mv, haed_split_return,color=colors[0], alpha=0.3, label='Static Env')
# plt.plot(y_mv, old_split_return,color=colors[1], alpha=0.3, label='Dynamic Env')
plt.plot(y_mv, old_split_return_mv, label='Beta PPO', color=colors[1])
plt.plot(y_mv, haed_split_return_mv, label='Normal-clip PPO', color=colors[0])
# plt.plot(y_mv, mid_quitserver_return_mv, label='Quit server set', color=colors[4])

plt.fill_between(y_mv, old_split_return_mv - free_std, old_split_return_mv + free_std,
                 alpha=0.3, color=colors[1],hatch='///'
)
plt.fill_between(y_mv, haed_split_return_mv - fix_std, haed_split_return_mv + fix_std,
                 alpha=0.3, color=colors[0],hatch='///'
)
# plt.fill_between(y_mv, mid_quitserver_return_mv - mid_std, mid_quitserver_return_mv + mid_std,
#                  alpha=0.3, color=colors[4],hatch='///')
# plt.scatter(2400,mid_quitserver_return_mv[2400], marker='*', color='red', s=40, zorder=10)
# plt.text(2400, mid_quitserver_return_mv[2400] + 2, f'Server\nBreakdown',fontsize=8, color='red', fontweight='bold',ha='center' )
# plt.arrow(
#     2400, mid_quitserver_return_mv[2400],  # 起点稍微往上一点
#     100, 2,   # 向右下倾斜
#     head_width=40, head_length=20,
#     fc='red', ec='red', linewidth=2
# )
# plt.annotate(
#     'Server\nBreakdown',
#     xy=(2420, mid_quitserver_return_mv[2400]+0.1),
#     xytext=(2600, mid_quitserver_return_mv[2400]+4),
#     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
#     fontsize=8, color='black', fontweight='bold', ha='center'
# )
# 添加统计注释
annotate_stats(y_mv, haed_split_return_mv, haed_split_return, fix_std, label='head', color=colors[0])
annotate_stats(y_mv, old_split_return_mv, old_split_return, free_std, label='old', color=colors[1])
# annotate_stats(y_mv, mid_quitserver_return_mv, mid_quitserver_return, mid_std, label='mid', color=colors[4])
# 图形美化
plt.xlabel('Episode')
plt.ylabel('Return')
plt.legend(loc='lower right')
plt.grid()
plt.show()
plt.close()

