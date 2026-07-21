"""
拥堵因果溯源模块 — 稀疏矩阵迭代传播

核心思想：每辆车携带"根因水分"，通过拥堵传导矩阵 A 逆流传播，
迭代收敛后分数最高的车辆即为拥堵的"罪魁祸首"。

算法
----
1. 构建 N×N 稀疏邻接矩阵 A:
   A_ij = conf(P_i) × max(0, (v_j - v_i) / v_ref)  如果 j 在 i 前方 D_max 内且方向相近
          0                                           否则

2. 迭代传播 x_{t+1} = x_t + α · (A^T @ x_t)，收敛于 x*

3. 归一化为百分比: root_cause_pct_i = x_i / sum(x) × 100

依赖
----
仅 numpy — 不引入新依赖，纯 CPU 运算。
"""

import numpy as np


def compute_root_cause(
    vehicles: list[dict],
    influences: list[float],
    conflict_field: np.ndarray,
    grid_cfg,
    alpha: float = 0.4,
    n_iters: int = 5,
    max_fwd_dist_m: float = 15.0,
    v_ref: float = 5.0,
) -> np.ndarray:
    """
    计算每辆车的拥堵根因分数。

    Parameters
    ----------
    vehicles : list[dict]
        当前帧车辆列表，每项需含:
        - 'track_id': int
        - 'world_x', 'world_y': 世界坐标 (m)
        - 'speed_mps': 速度 (m/s), 可选
        - 'heading_deg': 朝向角度 (度)
    influences : list[float]
        当前归因分数（长度同 vehicles）
    conflict_field : (G, G) float32
        冲突场
    grid_cfg : GridConfig
        网格配置（cell_size_m, grid_size, origin_x, origin_y）
    alpha : float
        传播率，建议 0.3-0.5
    n_iters : int
        迭代次数，5 次足够收敛
    max_fwd_dist_m : float
        前向搜索距离（米），超过此距离的车不认为有直接依赖
    v_ref : float
        参考速度，用于归一化速度差

    Returns
    -------
    root_cause_scores : (N,) float64
        每辆车的根因分数（未归一化，值越高问题越大）
    """
    N = len(vehicles)
    if N < 2:
        return np.ones(N, dtype=np.float64)

    # ── 提取必要数据 ──
    world_pos = np.zeros((N, 2), dtype=np.float64)
    speed = np.zeros(N, dtype=np.float64)
    heading = np.zeros(N, dtype=np.float64)
    cell_size = grid_cfg.cell_size_m
    ox, oy = grid_cfg.origin_x, grid_cfg.origin_y

    for i, v in enumerate(vehicles):
        world_pos[i] = [v.get('world_x', 0), v.get('world_y', 0)]
        speed[i] = v.get('speed_mps', 0)
        heading_deg = v.get('heading_deg', 0)
        heading[i] = np.radians(90.0 - heading_deg)

    # ── 第 1 步：构建 N×N 邻接矩阵 A ──
    # A_ij = 水从车辆 i 流向车辆 j 的强度
    # 连接依据：两车之间的冲突场值 —— 有冲突就有因果传递，无关方向
    A = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            dx = world_pos[j, 0] - world_pos[i, 0]
            dy = world_pos[j, 1] - world_pos[i, 1]
            dist = np.hypot(dx, dy)

            if dist > max_fwd_dist_m or dist < 0.5:
                continue

            # ── 方向因果检查：j 必须在 i 的前方 ──
            # 将 i→j 的向量投影到 i 的行驶方向，正投影 = j 在前方
            proj = dx * np.cos(heading[i]) + dy * np.sin(heading[i])
            if proj < 0:
                continue

            # ── 冲突场连接因子 ──
            mid_x = (world_pos[i, 0] + world_pos[j, 0]) / 2.0
            mid_y = (world_pos[i, 1] + world_pos[j, 1]) / 2.0
            mgx = int((mid_x - ox) / cell_size)
            mgy = int((mid_y - oy) / cell_size)
            c_factor = 0.0
            if 0 <= mgx < grid_cfg.grid_size and 0 <= mgy < grid_cfg.grid_size:
                c_factor = float(conflict_field[mgy, mgx])
            if c_factor <= 0:
                continue

            # ── 距离因子 ──
            d_factor = np.exp(-0.5 * (dist / (max_fwd_dist_m * 0.5)) ** 2)

            A[i, j] = c_factor * d_factor

    # ── 第 2 步：行归一化 + 迭代传播 ──
    row_sums = A.sum(axis=1, keepdims=True)
    A_norm = A / np.maximum(row_sums, 1e-10)

    x_prop = np.ones(N, dtype=np.float64)
    effective_iters = max(n_iters, int(N * 1.5))

    for _ in range(effective_iters):
        x_prop = x_prop + alpha * (A_norm.T @ x_prop)
        x_prop = np.clip(x_prop, 0, 1e6)

    # ── 第 3 步：冲突场加权 — 水只汇聚到真正拥堵的位置 ──
    # 孤立车辆：冲突场≈0 → 分数≈0 → 不会被标记为根因
    # 拥堵车辆：冲突场>0 → 分数>0 → 可能标记为根因
    x_result = x_prop.copy()
    for i in range(N):
        gx = int((world_pos[i, 0] - ox) / cell_size)
        gy = int((world_pos[i, 1] - oy) / cell_size)
        pos_conf = 0.0
        if 0 <= gx < grid_cfg.grid_size and 0 <= gy < grid_cfg.grid_size:
            pos_conf = float(conflict_field[gy, gx])
        # 没有冲突的地方水不停留（穿过继续往前传）
        x_result[i] = x_prop[i] * min(pos_conf * 10.0, 1.0)

    return x_result


def root_cause_to_pct(x: np.ndarray) -> np.ndarray:
    """根因分数 → 百分比（sum = 100%）"""
    total = x.sum()
    if total > 0:
        return x / total * 100.0
    return np.zeros_like(x)
