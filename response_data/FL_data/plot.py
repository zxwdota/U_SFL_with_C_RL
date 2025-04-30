import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rl_utils

# 加载数据
with open('acc.pkl', 'rb') as f:
    old_split_return = -np.array(pickle.load(f))


a=1