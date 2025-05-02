import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rl_utils

# 加载数据
with open('iid_acc.pkl', 'rb') as f:
    old_split_return = np.array(pickle.load(f))
old_split_return_mv = rl_utils.moving_average(old_split_return,1)

plt.scatter(range(len(old_split_return)), old_split_return, marker='s')
plt.plot(range(len(old_split_return)), old_split_return, alpha=0.3)
plt.plot(range(len(old_split_return)), old_split_return_mv)
plt.show()
plt.close()