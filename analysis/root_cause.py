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
    conf_val = np.zeros(N, dtype=np.float64)
    cell_size = grid_cfg.cell_size_m
    ox, oy = grid_cfg.origin_x, grid_cfg.origin_y

    for i, v in enumerate(vehicles):
        world_pos[i] = [v.get('world_x', 0), v.get('world_y', 0)]
        speed[i] = v.get('speed_mps', 0)
        # heading_deg 是 OpenCV 格式 (0°=N, 顺时针)，转为数学弧度 (0°=E, 逆时针)
        heading_deg = v.get('heading_deg', 0)
        heading[i] = np.radians(90.0 - heading_deg)

        # 冲突场强度 = 该车所在网格位置的冲突值
        gx = int((world_pos[i, 0] - ox) / cell_size)
        gy = int((world_pos[i, 1] - oy) / cell_size)
        if 0 <= gx < grid_cfg.grid_size and 0 <= gy < grid_cfg.grid_size:
            conf_val[i] = conflict_field[gy, gx]
        else:
            conf_val[i] = 0.0

    # ── 第 1 步：构建 N×N 邻接矩阵 A ──
    # A_ij = 车 j 对车 i 的拥堵依赖强度
    # 物理含义：如果车 j 在车 i 前方且比 i 慢，则 j 是 i 的拥堵原因
    A = np.zeros((N, N), dtype=np.float64)
    max_dist_cell = max_fwd_dist_m / cell_size

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # 方向向量 i → j
            dx = world_pos[j, 0] - world_pos[i, 0]
            dy = world_pos[j, 1] - world_pos[i, 1]
            dist = np.hypot(dx, dy)

            # 条件 1：距离在有效范围内
            if dist > max_fwd_dist_m or dist < 0.5:
                continue

            # 条件 2：j 在 i 的前方（指向 i 的行驶方向 ±60°）
            fwd_angle = np.arctan2(dy, dx)
            angle_diff = abs(fwd_angle - heading[i])
            if angle_diff > np.pi and angle_diff < 2 * np.pi:
                angle_diff = 2 * np.pi - angle_diff
            if angle_diff > np.pi / 3:  # ±60° 以内才算前方
                continue

            # 条件 3：j 比 i 慢（或是排队场景）才是拥堵源
            speed_diff = speed[j] - speed[i]
            if speed_diff > 0.5:  # j 比 i 更快 → j 不是 i 的拥堵源
                continue

            # ── 速度差因子 ──
            # 如果 j 明显比 i 慢: v_factor 高 (j 拖累了 i)
            # 如果两者速度相近: 用位置打破平局
            v_factor = max(0, min(-speed_diff / v_ref, 1.0))

            # ── 位置因子：排队场景下，前方的车是根因 ──
            pos_along_i = world_pos[i, 0] * np.cos(heading[i]) + world_pos[i, 1] * np.sin(heading[i])
            pos_along_j = world_pos[j, 0] * np.cos(heading[j]) + world_pos[j, 1] * np.sin(heading[j])
            pos_diff = pos_along_j - pos_along_i

            # j 必须在 i 的前方（沿行驶方向正投影）
            if pos_diff < -2.0:  # j 在 i 后方 → 不可能引起 i 拥堵
                continue

            # 位置增强因子：j 越靠前，接收的水越多
            # 当速度差小时用位置打破平局
            p_factor = 1.0 + min(pos_diff / 8.0, 1.0)

            # 基础连接：即使速度差为 0 也有最小连接（排队场景）
            base = 0.15 if abs(speed_diff) < 0.1 else v_factor

            # 距离因子：越近权重越高（高斯衰减）
            d_factor = np.exp(-0.5 * (dist / (max_fwd_dist_m * 0.4)) ** 2)

            A[i, j] = conf_val[i] * base * p_factor * d_factor

    # ── 第 2 步：迭代传播 ──
    x_prop = np.ones(N, dtype=np.float64)
    effective_iters = max(n_iters, int(N * 1.5))

    for _ in range(effective_iters):
        x_prop = x_prop + alpha * (A.T @ x_prop)
        x_prop = np.clip(x_prop, 0, 1e6)

    # ── 第 3 步：位置梯度信号（排队场景——所有车速度相近时使用） ──
    # 沿行驶方向投影：越靠前的车根因分数越高
    pos_along = np.array([
        world_pos[i, 0] * np.cos(heading[i]) + world_pos[i, 1] * np.sin(heading[i])
        for i in range(N)
    ])
    pos_norm = (pos_along - pos_along.min()) / max(pos_along.max() - pos_along.min(), 1e-8)
    # 位置信号：队首=1.0，队尾=0.0，中间线性
    pos_score = pos_norm * N * 0.5

    # ── 第 4 步：混合两个信号 ──
    # 速度方差大时，用传播信号；速度方差小时（全停/全慢），用位置信号
    speed_var = np.var(speed) / max(v_ref * v_ref, 1e-8)
    w_pos = np.exp(-5.0 * speed_var)  # speed_var→0 → w→1（全用位置）
    w_adj = 1.0 - w_pos               # speed_var大 → w→1（全用传播）

    x = w_adj * x_prop + w_pos * pos_score
    return x


def root_cause_to_pct(x: np.ndarray) -> np.ndarray:
    """根因分数 → 百分比（sum = 100%）"""
    total = x.sum()
    if total > 0:
        return x / total * 100.0
    return np.zeros_like(x)
