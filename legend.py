import matplotlib.pyplot as plt

# 数据
x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 6, 8, 10]

# 创建图形
fig, ax = plt.subplots()

# 绘制两条线
line1, = ax.plot(x, y1, label='QoS', linestyle='--',marker='^',linewidth=2,color='black')
line2, = ax.plot(x, y2, label='T', linestyle='-',marker='o',linewidth=2,color='black')

# 创建图例对象
figlegend = plt.figure(figsize=(10,2),dpi=400)  # 新建一个空白图形
legend = figlegend.legend(handles=[line1, line2], loc='center', frameon=True, ncol=2, handlelength=4)  # 居中显示图例
legend.get_frame().set_edgecolor('black')  # 设置框的边框颜色
legend.get_frame().set_linewidth(1.5)      # 设置框的边框线宽
legend.get_frame().set_alpha(0.5)         # 设置框的透明度
# 调整图例的大小
figlegend.canvas.draw()  # 更新图形
bbox = legend.get_window_extent().transformed(figlegend.dpi_scale_trans.inverted())
figlegend.savefig("legend.png", bbox_inches='tight', pad_inches=0.05)  # 增加边距

plt.show()