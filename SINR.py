import numpy as np
import matplotlib.pyplot as plt
def resample_tail_dense_cdf(x, y, num_points=100):
    """
    x: 已排序后的 x 数据，例如 log10(SINR)
    y: 累积分布函数 (CDF)，范围应为 [0, 1]
    num_points: 重新采样点数
    返回: 新的 x, y，用于 scatter 绘制
    """

    # 创建一个 [0, 1] 范围内的非均匀采样分布：两头密集中间稀疏
    u = np.linspace(-2, 2, num_points)  # [-2, 2] 是 tanh 的非线性区间
    y_target = (np.tanh(u) + 1) / 2     # 映射到 [0, 1]

    # 使用插值法，从原始 CDF 中查出这些 y 所对应的 x 值
    x_interp = np.interp(y_target, y, x)
    return x_interp, y_target

def resample_head_sparse_tail_dense_cdf(x, y, num_points=100, factor=1):
    """
    重新采样 CDF 点，使得前面稀疏，后面密集
    参数：
        x: 已排序的 x（如 SINR dB）
        y: 累积分布值（在 [0, 1] 范围内）
        num_points: 要采样的点数
        factor: 控制非线性程度（越大，尾部越密）
    返回：
        x_interp, y_target: 重新采样的 x/y 值
    """

    # 创建一个偏向尾部的采样点分布（指数映射）
    t = np.linspace(0, 1, num_points)
    y_target = t**factor  # factor 越大，越密集靠近 y=1 的尾部

    # 根据 y 反查 x（使用插值）
    x_interp = np.interp(y_target, y, x)
    return x_interp, y_target
# ---------- 参数设置 ----------
N = 2000  # 样本数量

Pt = 1.0  # W
Gt = 10.0  # dBi（用线性值就是 10 倍增益）
Gr = 1.0  # 用户设备天线增益
fc = 3.5e9  # 5G Sub-6GHz
lambda_c = 3e8 / fc
bandwidth = 20e6
N0 = 4e-21  # 更接近物理热噪声（W/Hz）
# N0 = 1e-9  # 噪声功率谱密度 (W/Hz)

I = 1e-9  # 邻区干扰功率
sigma_list = [0.5, 1, 2, 3]  # 瑞利衰落的标准差
colors_simulation = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']  # 仿真曲线的颜色
colors_theory = ['tab:red', 'tab:cyan', 'purple', 'olive']  # 理论曲线的颜色
np.random.seed(42)

plt.figure(figsize=(4.5, 4), dpi=300)
plt.axes([0.17, 0.12, 0.8, 0.85])
# ---------- 客户端与服务器之间的距离 ----------
d = np.random.uniform(10, 100, N)  # 随机距离（单位：米）
d = 100  # 假设距离为 100 米
scatter_shape = ['o','s','D','v']
for i, sigma in enumerate(sigma_list):
    # ---------- 大尺度路径损耗（Free Space Path Loss） ----------
    FSPL = Pt * Gt * Gr * (lambda_c / (4 * np.pi)) ** 2  # 衰减随 d^2

    # ---------- 小尺度瑞利衰落（复高斯） ----------
    h = (np.random.normal(0, sigma / np.sqrt(2), N) + 1j * np.random.normal(0, sigma / np.sqrt(2), N))
    Rayleigh_gain = np.abs(h) ** 2  # |h|^2 ~ Exp(sigma^2)

    # ---------- 总信道增益 ----------
    channel_gain = FSPL * Rayleigh_gain / (d**2)

    # ---------- SINR ----------
    SINR = channel_gain / (I + N0 * bandwidth)

    # ---------- 仿真 SINR 的经验 CDF ----------
    SINR_sorted = np.sort(SINR)
    cdf = np.arange(1, N + 1) / N

    # ---------- 理论 CDF（指数分布） ----------
    gamma = np.mean(SINR)  # 使用线性单位的平均 SINR

    # 计算理论 CDF 的范围
    SINR_range = np.linspace(0, np.percentile(SINR, 99.5), 1000)
    CDF_theory = 1 - np.exp(-SINR_range / gamma)


    # 统一 x 轴范围：以仿真数据的最大值为终点
    SINR_max = np.max(SINR)

    # 修改理论 CDF 的范围：与仿真数据的最大值一致
    SINR_range = np.linspace(0, SINR_max, 1000)
    # SINR_sorted_re,cdf_re = resample_head_sparse_tail_dense_cdf(SINR_sorted,cdf,num_points=40)
    SINR_sorted_re,cdf_re = resample_tail_dense_cdf(SINR_sorted,cdf,num_points=35)

    # 计算理论 CDF
    CDF_theory = 1 - np.exp(-SINR_range / gamma)
    plt.plot(10 * np.log10(SINR_range), CDF_theory,'-',  label=f'Theory $\\sigma = {sigma}$',linewidth=1, color=colors_theory[i],alpha=1)
    # plt.plot(10 * np.log10(SINR_sorted), cdf,label=f'Simu $\\sigma = {sigma}$', linewidth=2, color=colors_simulation[i])
    plt.scatter(10 * np.log10(SINR_sorted_re), cdf_re, s=20,label=f'Simu $\\sigma = {sigma}$',marker=scatter_shape[i], color=colors_theory[i],facecolors='none', alpha=1)  # 添加散点图
# 绘制对比图


plt.xlabel('SINR (dB)')
plt.ylabel('CDF')
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()