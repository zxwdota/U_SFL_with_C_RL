import torch
from thop import profile  # 不建议使用profile，会污染模型，变成float64。不建议实用float64，速度慢且支持性差（MPS不支持）
# 将profile 包中的float64改为float32 现在变得可用了。
from ptflops import get_model_complexity_info
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np

from utils.resnet import ResNet50, ResNet152
from mysfl_multi_splitpoint import ResNet18_head_side, Baseblock, ResNet18_mid_side, ResNet18_tail_side

input_list = [0,1,1,2,2,7]

cut_point_1 = 7
cut_point_2 = 7

device = torch.device('cpu')
head_model = ResNet18_head_side(Baseblock, [2, 2, 2, 2], 7).to(device)
mid_model = ResNet18_mid_side(Baseblock, [2, 2, 2, 2], 7).to(device)
tail_model = ResNet18_tail_side(Baseblock, [2, 2, 2, 2], 7).to(device)
input_tensor = torch.randn(64, 3, 64, 64).to(device)
# smashed_data_1 = head_model(input_tensor,cut_point_1,cut_point_2)
# smashed_data_2 = mid_model(smashed_data_1,cut_point_1,cut_point_2)
# ylabel = tail_model(smashed_data_2,cut_point_1,cut_point_2)

def kb_test(torch_tensor):
    num_elements = torch_tensor.numel()
    bytes_per_element = torch_tensor.element_size()
    total_bytes = num_elements * bytes_per_element
    total_kb = total_bytes / 1024
    print(f"Total size of input_tensor: {total_kb:.2f} KB")
kb_test(input_tensor)
# kb_test(smashed_data_1)
# kb_test(smashed_data_2)
# kb_test(ylabel)
macs_profile, params_profile = profile(head_model, inputs=(input_tensor,cut_point_1,cut_point_2))
print('macs: %.2f M, params: %.2f M' % (macs_profile / 1e6, params_profile / 1e6))


# macs_profile, params_profile = profile(head_model, inputs=(input_tensor,))

def input_constructor(input_res):
    # 构造适配模型 forward() 的输入格式
    x = torch.randn(1, *input_res).to('cpu')
    cut_point_1 = 0
    cut_point_2 = 7
    return dict(x=x, cut_point_1=cut_point_1, cut_point_2=cut_point_2)

# macs, params = get_model_complexity_info(head_model, (3, 64, 64), input_constructor=input_constructor, as_strings=True, print_per_layer_stat=True)
# print('macs: ', macs, 'params: ', params)

#
# macs_profile, params_profile = profile(mid_model, inputs=(smashed_data_1,))
# macs, params = get_model_complexity_info(mid_model, (64, 16, 16), as_strings=True, print_per_layer_stat=True)
#
# print('macs: %.2f M, params: %.2f M' % (macs_profile / 1e6, params_profile / 1e6))
#

#
# macs_profile, params_profile = profile(tail_model, inputs=(smashed_data_2,))
# macs, params = get_model_complexity_info(tail_model, (512, 2, 2), as_strings=True, print_per_layer_stat=True)
#
# print('macs: %.2f , params: %.2f ' % (macs_profile, params_profile))

# print('resnet18!!!/n')
# net = ResNet18().to(device)
# macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=True)
# print('resnet34!!!/n')
# net = ResNet34().to(device)
# macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=True)
print('resnet50!!!/n')
net = ResNet50().to(device)
macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=True)
# print('resnet101!!!/n')
# net = ResNet101().to(device)
# macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=True)
print('resnet152!!!/n')
net = ResNet152().to(device)
macs, params = get_model_complexity_info(net, (3, 64, 64), as_strings=True, print_per_layer_stat=True)

# from inception_v3 import GoogLeNet
#
# print('inception_v3!!!/n')
# net = GoogLeNet(1000).to(device)
# macs, params = get_model_complexity_info(net, (3, 299, 299), as_strings=True, print_per_layer_stat=True)



head_model_time_list = []
head_model_FPS_list = []
mid_model_time_list = []
mid_model_FPS_list = []
tail_model_time_list = []
tail_model_FPS_list = []

#测试SFL时间FPS
# for i in range(200):
#     torch.mps.synchronize()
#     start = time.time()
#     with torch.no_grad():
#         result = head_model(input_tensor)
#     torch.mps.synchronize()
#     end = time.time()
#     infer_time = end - start
#     FPS = 1 / infer_time
#     head_model_time_list.append(infer_time)
#     head_model_FPS_list.append(FPS)
#
#     torch.mps.synchronize()
#     start = time.time()
#     with torch.no_grad():
#         result = mid_model(smashed_data_1)
#     torch.mps.synchronize()
#     end = time.time()
#     infer_time = end - start
#     FPS = 1 / infer_time
#     mid_model_time_list.append(infer_time)
#     mid_model_FPS_list.append(FPS)
#
#     torch.mps.synchronize()
#     start = time.time()
#     with torch.no_grad():
#         result = tail_model(smashed_data_2)
#     torch.mps.synchronize()
#     end = time.time()
#     infer_time = end - start
#     FPS = 1 / infer_time
#     tail_model_time_list.append(infer_time)
#     tail_model_FPS_list.append(FPS)


res18_model_time_list=[]
res18_model_FPS_list=[]

size_in_bytes = input_tensor.element_size() * input_tensor.nelement()
size_in_kb = size_in_bytes / 1024
size_in_mb = size_in_kb / 1024

print(f"Smashed data size: {size_in_bytes} bytes = {size_in_kb:.2f} KB = {size_in_mb:.2f} MB")

#测试总res18时间FPS
for i in range(200):
    torch.mps.synchronize()
    start = time.time()
    with torch.no_grad():
        smashed_data_1 = head_model(input_tensor,cut_point_1,cut_point_2)
        # size_in_bytes = smashed_data_1.element_size() * smashed_data_1.nelement()
        # size_in_kb = size_in_bytes / 1024
        # size_in_mb = size_in_kb / 1024
        #
        # print(f"Smashed data size: {size_in_bytes} bytes = {size_in_kb:.2f} KB = {size_in_mb:.2f} MB")

    torch.mps.synchronize()
    end = time.time()
    infer_time = end - start
    FPS = 1 / infer_time
    res18_model_time_list.append(infer_time)
    res18_model_FPS_list.append(FPS)
print('avg_res18_model_time_:', np.mean(res18_model_time_list))
print('avg_res18_model_FPS_:', np.mean(res18_model_FPS_list))


import matplotlib.pyplot as plt

plt.plot(head_model_time_list)
plt.plot(mid_model_time_list)
plt.plot(tail_model_time_list)
plt.xlabel('iteration')
plt.ylabel('infer_time')
plt.title('infer_time')
plt.show()
plt.plot(head_model_FPS_list)
plt.plot(mid_model_FPS_list)
plt.plot(tail_model_FPS_list)
plt.xlabel('iteration')
plt.ylabel('FPS')
plt.title('FPS')
plt.show()

plt.plot(res18_model_time_list)
plt.xlabel('iteration')
plt.ylabel('infer_time')
plt.title('infer_time')
plt.show()
plt.plot(res18_model_FPS_list)
plt.xlabel('iteration')
plt.ylabel('FPS')
plt.title('FPS')
plt.show()


maclist = [0.341, 27.186, 24.140, 24.111, 24.096, 0.024]

list = []
def split_into_three(lst):
    n = len(lst)
    results = []
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            part1 = lst[:i]
            part2 = lst[i:j]
            part3 = lst[j:]
            sum1 = sum(part1)+sum(part3)
            sum2 = sum(part2)
            results.append({
                'split': (part1, part2, part3),
                'sums': (sum1, sum2)})
            list.append([sum1,sum2])
    return results

splits = split_into_three(maclist)
list = np.array(list)

# 打印所有切分方式
for idx, result in enumerate(splits):
    p1, p2, p3 = result['split']
    s1, s2 = result['sums']
    print(f"Split {idx + 1}:\n"
          f"  Part 1:  (sum = {s1:.3f})\n"
          f"  Part 2:  (sum = {s2:.3f})\n")
