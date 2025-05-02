import numpy as np
import random
from collections import deque


def bfs_clustering(adj_list, lc, max_len=None):
    """
    算法1 BFS-Clustering: 生成有序哈密顿路径（簇）
    :param adj_list: dict, 顶点->邻居列表
    :param lc: 初始学习客户端节点
    :param max_len: 最大簇长度（包含LC），若None则遍历所有长度
    :return: list of paths (each path is list of nodes)
    """
    C_space = []
    queue = deque([[lc]])
    while queue:
        path = queue.popleft()
        C_space.append(path)
        if max_len is not None and len(path) >= max_len:
            continue
        last = path[-1]
        for nei in adj_list.get(last, []):
            if nei not in path:  # 防止重复
                new_path = path + [nei]
                queue.append(new_path)
    return C_space


def heuristic_msa(path, flops_per_layer, client_flops):
    """
    算法2 Heuristic MSA: 模型切分与分配
    :param path: list of client indices in簇内顺序
    :param flops_per_layer: list of每层前向FLOPs
    :param client_flops: dict client->算力权重
    :return: split_points: list of layer indices where模型切分
    """
    n = len(path)
    total_flops = sum(flops_per_layer)
    # 目标分配工作量
    targets = [client_flops[c] / sum(client_flops.values()) * total_flops for c in path]

    split_points = []
    cum = 0
    idx = 0
    for t in targets[:-1]:  # 最后一个segment由剩余层承担
        cum_target = cum + t
        running = 0
        for j in range(idx, len(flops_per_layer)):
            running += flops_per_layer[j]
            if running >= cum_target:
                split_points.append(j)
                idx = j + 1
                cum = running
                break
    return split_points


def tc_and_bandwidth(clusters, adj_weights, dl_rates, ul_rates, lambdas, B):
    """
    算法3 TC 选择和带宽分配
    :param clusters: dict lc->path
    :param adj_weights: matrix邻居连接权重R_{i,j}
    :param dl_rates: dict lc->(client_idx->downlink rate)
    :param ul_rates: dict lc->(client_idx->uplink rate)
    :param lambdas: dict lc->权重 lambda_l
    :param B: 总带宽
    :return: tcs: dict lc->selected TC, bws: dict lc->带宽分配
    """
    # 计算并选择 TC
    tcs = {}
    for lc, path in clusters.items():
        best_val = -np.inf
        best_tc = None
        for i in path:
            # 计算TCV
            neigh_sum = sum(adj_weights[i][j] for j in path if j != i)
            rate_sum = dl_rates[lc][i] + ul_rates[lc][i]
            tcv = neigh_sum + len(path) * rate_sum
            if tcv > best_val:
                best_val = tcv
                best_tc = i
        tcs[lc] = best_tc
    # 带宽分配闭式
    denom = sum(
        np.sqrt(lambdas[lc] * (1 / dl_rates[lc][tcs[lc]] + 1 / ul_rates[lc][tcs[lc]]))
        for lc in clusters
    )
    bws = {}
    for lc in clusters:
        bws[lc] = B * np.sqrt(
            lambdas[lc] * (1 / dl_rates[lc][tcs[lc]] + 1 / ul_rates[lc][tcs[lc]])
        ) / denom
    return tcs, bws


def best_response_clustering(adj_list, client_flops, flops_per_layer,
                              adj_weights, dl_rates, ul_rates, lambdas,
                              B, max_iter=50):
    """
    算法4 Best-Response Distributed Clustering
    :return: clusters, split_points, tcs, bws
    """
    # 离线阶段: 生成策略空间
    C_space = {lc: bfs_clustering(adj_list, lc) for lc in client_flops}
    # 初始化随机策略 (选择最短路径)
    clusters = {lc: [paths[0] for paths in [C_space[lc]]][0] for lc in C_space}

    for it in range(max_iter):
        updated = False
        for lc in clusters:
            best_u = -np.inf
            best_cfg = None
            # 对每个候选簇尝试最佳回应
            for path in C_space[lc]:
                # 1. MSA 切分
                splits = heuristic_msa(path, flops_per_layer, {c: client_flops[c] for c in path})
                # 2. TC & 带宽
                tmp_clusters = clusters.copy()
                tmp_clusters[lc] = path
                tcs, bws = tc_and_bandwidth(tmp_clusters, adj_weights, dl_rates, ul_rates, lambdas, B)
                # 3. 计算效用 u = -Σ φ_l
                total_cost = 0
                for l, p in tmp_clusters.items():
                    tc = tcs[l]
                    bw = bws[l]
                    # 延迟近似为 (模型大小 / bw) + 逆速率
                    dl = 1 / dl_rates[l][tc]
                    ul = 1 / ul_rates[l][tc]
                    total_cost += lambdas[l] * (dl + ul)
                u = -total_cost
                if u > best_u:
                    best_u = u
                    best_cfg = (path, splits)
            # 更新簇
            if best_cfg is not None and best_cfg[0] != clusters[lc]:
                clusters[lc] = best_cfg[0]
                updated = True
        if not updated:
            break
    # 收敛后计算最终配置
    split_points = {lc: heuristic_msa(clusters[lc], flops_per_layer, {c: client_flops[c] for c in clusters[lc]}) for lc in clusters}
    tcs, bws = tc_and_bandwidth(clusters, adj_weights, dl_rates, ul_rates, lambdas, B)
    return clusters, split_points, tcs, bws


if __name__ == "__main__":
    # 示例调用
    # 定义网络拓扑
    adj_list = {
        0: [1,2], 1: [0,2], 2: [0,1,3], 3: [2]
    }
    client_flops = {0:1.0, 1:2.0, 2:1.5, 3:1.0}
    flops_per_layer = [100,200,150,50]
    adj_weights = np.ones((4,4))  # 简化为全1
    dl_rates = {lc: {i: random.uniform(1,5) for i in adj_list} for lc in adj_list}
    ul_rates = {lc: {i: random.uniform(1,5) for i in adj_list} for lc in adj_list}
    lambdas = {lc:1.0 for lc in adj_list}
    B = 10.0

    clusters, splits, tcs, bws = best_response_clustering(
        adj_list, client_flops, flops_per_layer,
        adj_weights, dl_rates, ul_rates, lambdas, B
    )
    print("Final clusters:", clusters)
    print("Split points:", splits)
    print("TCs:", tcs)
    print("Bandwidths:", bws)
