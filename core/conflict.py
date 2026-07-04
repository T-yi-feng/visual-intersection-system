"""
卷积冲突检测模块 (Convolution-based Conflict Detection)

替代原 O(N²) 两两箭头线段求交方法，通过网格化 + 方向场卷积实现 GPU 可加速的冲突检测。

核心思路：
1. 将 BEV 平面离散化为 G×G 网格
2. 将车辆位置散布到网格，构建占用场 O(x,y)、速度场 V(x,y)、方向场 θ(x,y)
3. 将方向场分解为 8 个方向 bin 的二值占用场 O_0..O_7
4. 对每个方向场做各向异性高斯卷积（沿方向拉伸），得到路径影响力场 R_0..R_7
5. 冲突对逐元素相乘，得到冲突场 C(x,y)
6. 每辆车的归因分数 = 其所在位置的影响力场值 × 冲突方向场值之和
"""

import numpy as np
import cv2
from math import pi, cos, sin, exp
from typing import Optional

# ============================================================
# 常量
# ============================================================

# 12 方向 bin：每 30° 一个，覆盖全方位
DIRECTION_BINS = 12
BIN_NAMES = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW']
BIN_DEGREES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# 精选冲突对：12 对（对向 6 + 正交 6）
# 交织发生在方向差异足够大的路径之间：
# - 对向（180°）：N-S, NNE-SSW, NE-SW, ENE-WSW, E-W, ESE-WNW
# - 正交（~90°）：N-E, NNE-ENE, NE-ESE, E-SSE, SE-S, SSE-SSW
# 相邻方向（<60°）的卷积核在空间上高度重叠，不会产生真正交织，跳过
DEFAULT_CONFLICT_PAIRS = [
    # 对向 (180°)
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
    # 正交 (~90°)
    (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
]

# 扩展冲突对（与默认一致，已全量覆盖）
EXTENDED_CONFLICT_PAIRS = DEFAULT_CONFLICT_PAIRS


# ============================================================
# 网格参数
# ============================================================

class GridConfig:
    """BEV 网格配置"""
    def __init__(
        self,
        grid_size: int = 64,
        cell_size_m: float = 0.625,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        self.grid_size = grid_size
        self.cell_size_m = cell_size_m
        self.origin_x = origin_x
        self.origin_y = origin_y

    @classmethod
    def from_world_bounds(
        cls,
        world_width_m: float,
        world_height_m: float,
        grid_size: int = 64,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        """从世界坐标范围自动计算网格参数"""
        cell_size_m = max(world_width_m, world_height_m) / grid_size
        return cls(grid_size, cell_size_m, origin_x, origin_y)


class KernelConfig:
    """方向冲突核配置"""
    def __init__(
        self,
        arrow_half_len: int = 15,
        kernel_half_width: int = 4,
        sigma_along: float = 5.0,
        sigma_perp: float = 1.5,
        density_radius: int = 3,
        speed_sigma: float = 2.0,
    ):
        self.arrow_half_len = arrow_half_len      # 沿方向半长（cells）
        self.kernel_half_width = kernel_half_width  # 垂直方向半宽（cells）
        self.sigma_along = sigma_along              # 沿方向衰减
        self.sigma_perp = sigma_perp                # 垂直方向衰减
        self.density_radius = density_radius        # 密度核半径
        self.speed_sigma = speed_sigma              # 速度高斯核 σ


# ============================================================
# 1. 散布：车辆 → 网格场
# ============================================================

def scatter_vehicles_to_grid(
    vehicles: list[dict],
    grid_cfg: GridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将离散车辆散布到网格上，构建占用场、速度场、方向场。

    Parameters
    ----------
    vehicles : list[dict]
        每个 dict 包含：
        - cx, cy: 世界坐标 (m)
        - speed_mps: 速度 (m/s)
        - heading_deg: 航向角 (度, 0=北, 顺时针)
    grid_cfg : GridConfig
        网格配置

    Returns
    -------
    O : (H, W) float32  占用场（有车=1, 无车=0）
    V : (H, W) float32  速度场
    Theta : (H, W) float32  方向场（弧度）
    mask : (H, W) float32  有效标记
    """
    H = W = grid_cfg.grid_size
    O = np.zeros((H, W), dtype=np.float32)
    V = np.zeros((H, W), dtype=np.float32)
    Theta = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    speed_sum = np.zeros((H, W), dtype=np.float32)
    cos_sum = np.zeros((H, W), dtype=np.float32)
    sin_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)

    for v in vehicles:
        if not all(k in v for k in ('cx', 'cy', 'speed_mps', 'heading_deg')):
            continue
        gx = int((v['cx'] - grid_cfg.origin_x) / grid_cfg.cell_size_m)
        gy = int((v['cy'] - grid_cfg.origin_y) / grid_cfg.cell_size_m)
        if 0 <= gx < W and 0 <= gy < H:
            O[gy, gx] = 1.0
            speed_sum[gy, gx] += v['speed_mps']
            rad = v['heading_deg'] * pi / 180.0
            cos_sum[gy, gx] += cos(rad)
            sin_sum[gy, gx] += sin(rad)
            count[gy, gx] += 1
            mask[gy, gx] = 1.0

    # 同格多车：速度取平均，航向取矢量平均
    valid = count > 0
    V[valid] = speed_sum[valid] / count[valid]
    Theta[valid] = np.arctan2(sin_sum[valid], cos_sum[valid])

    return O, V, Theta, mask


# ============================================================
# 2. 密度场（Box Filter，速度最快）
# ============================================================

def compute_density_field(O: np.ndarray, kernel_radius: int = 3) -> np.ndarray:
    """
    局部密度场：均匀方核卷积。
    cv2.boxFilter 使用积分图，O(1) per pixel。

    Parameters
    ----------
    O : (H, W) 占用场
    kernel_radius : 核半径，实际核大小 = 2*radius+1

    Returns
    -------
    rho : (H, W) float32, 局部密度 [0, 1]
    """
    ksize = 2 * kernel_radius + 1
    return cv2.boxFilter(O, ddepth=-1, ksize=(ksize, ksize), normalize=True)


# ============================================================
# 3. 速度衰减场（高斯平滑）
# ============================================================

def compute_speed_decay_field(
    V: np.ndarray,
    mask: np.ndarray,
    v_ref: float,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    速度衰减场：高斯平滑后计算 1 - V/v_ref。
    无车区域不参与平滑。

    Parameters
    ----------
    V : (H, W) 速度场
    mask : (H, W) 有效标记
    v_ref : 参考速度 (m/s)
    sigma : 高斯核 σ

    Returns
    -------
    eta : (H, W) float32, 速度衰减 [0, 1]
    """
    V_filled = np.where(mask > 0, V, v_ref)
    V_smooth = cv2.GaussianBlur(V_filled, ksize=(0, 0), sigmaX=sigma)
    return np.maximum(0.0, 1.0 - V_smooth / v_ref).astype(np.float32)


# ============================================================
# 4. 方向场分解
# ============================================================

def decompose_direction(
    Theta: np.ndarray,
    mask: np.ndarray,
    n_bins: int = DIRECTION_BINS,
) -> list[np.ndarray]:
    """
    将连续方向场分解为 n_bins 个二值占用场。

    Parameters
    ----------
    Theta : (H, W) 方向场（弧度）
    mask : (H, W) 有效标记
    n_bins : 方向 bin 数量

    Returns
    -------
    layers : list of (H, W) float32, 每个 bin 的占用场
    """
    deg = (Theta * 180.0 / pi) % 360.0
    bin_size = 360.0 / n_bins

    layers = []
    for k in range(n_bins):
        center = k * bin_size
        lo = (center - bin_size / 2) % 360.0
        hi = (center + bin_size / 2) % 360.0

        if lo < hi:
            in_bin = ((deg >= lo) & (deg < hi)).astype(np.float32)
        else:
            # 跨越 0°/360°
            in_bin = ((deg >= lo) | (deg < hi)).astype(np.float32)

        layers.append(in_bin * mask)

    return layers


# ============================================================
# 5. 方向扩展核构建（各向异性，可分离）
# ============================================================

def build_directional_kernel(
    heading_deg: float,
    arrow_half_len: int,
    kernel_half_width: int,
    sigma_along: float,
    sigma_perp: float,
) -> np.ndarray:
    """
    构建沿指定方向拉伸的各向异性高斯核。

    Parameters
    ----------
    heading_deg : 核的方向（度，0=北）
    arrow_half_len : 沿方向半长（cells）
    kernel_half_width : 垂直方向半宽（cells）
    sigma_along : 沿方向衰减
    sigma_perp : 垂直方向衰减

    Returns
    -------
    K : (2*kw+1, 2*al+1) float32, 归一化核
    """
    # 核尺寸：沿方向更长
    k_h = 2 * kernel_half_width + 1  # 垂直方向（行）
    k_w = 2 * arrow_half_len + 1     # 沿方向（列）

    K = np.zeros((k_h, k_w), dtype=np.float32)

    cx = arrow_half_len       # 沿方向中心
    cy = kernel_half_width    # 垂直方向中心

    theta = heading_deg * pi / 180.0
    ux, uy = cos(theta), sin(theta)       # 方向向量
    nx, ny = -sin(theta), cos(theta)      # 法向量

    for i in range(k_h):
        for j in range(k_w):
            dx = j - cx
            dy = i - cy

            # 沿方向的投影距离
            along = dx * ux + dy * uy
            # 垂直方向的投影距离
            perp = dx * nx + dy * ny

            # 沿方向：在 arrow_half_len 范围内均匀，超出为 0
            if abs(along) <= arrow_half_len:
                f_along = 1.0
            else:
                f_along = 0.0

            # 垂直方向：高斯衰减
            f_perp = exp(-0.5 * (perp / sigma_perp) ** 2)

            K[i, j] = f_along * f_perp

    # 归一化
    total = K.sum()
    if total > 0:
        K /= total

    return K


def build_all_directional_kernels(
    n_bins: int = DIRECTION_BINS,
    kernel_cfg: Optional[KernelConfig] = None,
) -> list[np.ndarray]:
    """
    为所有方向 bin 构建方向扩展核。

    Returns
    -------
    kernels : list of n_bins 个核
    """
    if kernel_cfg is None:
        kernel_cfg = KernelConfig()

    bin_size = 360.0 / n_bins
    kernels = []
    for k in range(n_bins):
        heading = k * bin_size
        K = build_directional_kernel(
            heading,
            kernel_cfg.arrow_half_len,
            kernel_cfg.kernel_half_width,
            kernel_cfg.sigma_along,
            kernel_cfg.sigma_perp,
        )
        kernels.append(K)

    return kernels


# ============================================================
# 6. 冲突场计算（核心）
# ============================================================

def compute_conflict_field(
    layers: list[np.ndarray],
    kernels: list[np.ndarray],
    conflict_pairs: list[tuple[int, int]] = DEFAULT_CONFLICT_PAIRS,
) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    """
    冲突场计算：替代原来的 O(N²) 箭头交叉检测。

    Parameters
    ----------
    layers : 8 个方向占用场 [O_0, O_1, ..., O_7]
    kernels : 8 个方向扩展核 [K_0, K_1, ..., K_7]
    conflict_pairs : 冲突方向对列表

    Returns
    -------
    C_total : (H, W) float32, 总冲突场
    pair_results : dict, 每个冲突对的冲突场
    R : list of (H, W) float32, 每个方向的路径影响力场
    """
    # 第一步：对每个方向场做方向扩展卷积
    # R_k = O_k ⊛ K_k
    R = []
    for k in range(len(layers)):
        r = cv2.filter2D(layers[k], ddepth=-1, kernel=kernels[k])
        R.append(r)

    # 第二步：冲突对逐元素相乘
    C_total = np.zeros_like(R[0])
    pair_results = {}

    for (k1, k2) in conflict_pairs:
        c_pair = R[k1] * R[k2]
        pair_results[(k1, k2)] = c_pair
        C_total += c_pair

    return C_total, pair_results, R


# ============================================================
# 7. 车辆归因分数（Influence Field）
# ============================================================

def compute_vehicle_influence(
    vehicles: list[dict],
    R: list[np.ndarray],
    grid_cfg: GridConfig,
    conflict_pairs: list[tuple[int, int]] = DEFAULT_CONFLICT_PAIRS,
) -> list[float]:
    """
    计算每辆车的拥堵归因分数（Influence）。

    Influence_i = R[k_i](x_i, y_i) × Σ_{k' ∈ conflicts(k_i)} R[k'](x_i, y_i)

    即：车辆自身路径的强度 × 其所在位置上所有冲突方向路径的强度之和。

    Parameters
    ----------
    vehicles : list[dict], 每个有 cx, cy, heading_deg
    R : list of (H, W), 各方向的路径影响力场
    grid_cfg : GridConfig
    conflict_pairs : 冲突方向对

    Returns
    -------
    influences : list of float, 每辆车的归因分数
    """
    H = W = grid_cfg.grid_size

    # 预计算每个方向 bin 的冲突方向集合
    conflict_map = {k: set() for k in range(DIRECTION_BINS)}
    for (k1, k2) in conflict_pairs:
        conflict_map[k1].add(k2)
        conflict_map[k2].add(k1)

    bin_size = 360.0 / DIRECTION_BINS
    influences = []

    for v in vehicles:
        if not all(k in v for k in ('cx', 'cy', 'speed_mps', 'heading_deg')):
            continue
        gx = int((v['cx'] - grid_cfg.origin_x) / grid_cfg.cell_size_m)
        gy = int((v['cy'] - grid_cfg.origin_y) / grid_cfg.cell_size_m)

        if not (0 <= gx < W and 0 <= gy < H):
            influences.append(0.0)
            continue

        # 确定车辆所在的方向 bin
        heading = v['heading_deg'] % 360.0
        k_i = int(heading / bin_size) % DIRECTION_BINS

        # 自身路径强度
        self_strength = R[k_i][gy, gx]

        # 冲突方向强度之和
        conflict_strength = 0.0
        for k_prime in conflict_map.get(k_i, []):
            conflict_strength += R[k_prime][gy, gx]

        influences.append(float(self_strength * conflict_strength))

    return influences


# ============================================================
# 8. 综合 Phi 场计算
# ============================================================

def compute_phi_field(
    rho: np.ndarray,
    eta: np.ndarray,
    C: np.ndarray,
    w_rho: float = 0.3,
    w_v: float = 0.4,
    w_c: float = 0.3,
) -> np.ndarray:
    """
    综合拥堵指数场。

    Parameters
    ----------
    rho : 密度场 [0, 1]
    eta : 速度衰减场 [0, 1]
    C : 冲突场（内部归一化到 [0, 1]）
    w_rho : 密度权重
    w_v : 速度衰减权重
    w_c : 冲突权重

    Returns
    -------
    Phi : (H, W) float32, 拥堵指数场 [0, 1]
    """
    c_max = C.max()
    C_norm = C / c_max if c_max > 0 else C

    Phi = w_rho * rho + w_v * eta + w_c * C_norm
    return np.clip(Phi, 0.0, 1.0).astype(np.float32)


# ============================================================
# 9. 一站式接口
# ============================================================

class ConflictAnalyzer:
    """
    冲突分析器：一站式封装网格配置、核构建、冲突检测、归因计算。

    Usage
    -----
    >>> analyzer = ConflictAnalyzer(grid_size=64, world_width_m=40, world_height_m=40)
    >>> result = analyzer.analyze(vehicles)
    >>> result.phi_field        # 拥堵指数场 (64, 64)
    >>> result.influences       # 每辆车的归因分数
    >>> result.conflict_field   # 冲突场
    """

    def __init__(
        self,
        grid_size: int = 64,
        world_width_m: float = 40.0,
        world_height_m: float = 40.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        kernel_cfg: Optional[KernelConfig] = None,
        conflict_pairs: Optional[list[tuple[int, int]]] = None,
        w_rho: float = 0.3,
        w_v: float = 0.4,
        w_c: float = 0.3,
        v_ref: float = 6.2,
    ):
        self.grid_cfg = GridConfig.from_world_bounds(
            world_width_m, world_height_m, grid_size, origin_x, origin_y
        )
        self.kernel_cfg = kernel_cfg or KernelConfig()
        self.conflict_pairs = conflict_pairs or DEFAULT_CONFLICT_PAIRS
        self.w_rho = w_rho
        self.w_v = w_v
        self.w_c = w_c
        self.v_ref = v_ref

        # 预构建核（只构建一次）
        self.kernels = build_all_directional_kernels(
            DIRECTION_BINS, self.kernel_cfg
        )

    def analyze(self, vehicles: list[dict]) -> 'ConflictResult':
        """
        分析一批车辆的冲突情况。

        Parameters
        ----------
        vehicles : list[dict]
            每个 dict 包含 cx, cy (m), speed_mps, heading_deg

        Returns
        -------
        ConflictResult
        """
        # 散布到网格
        O, V, Theta, mask = scatter_vehicles_to_grid(vehicles, self.grid_cfg)

        # 密度场
        rho = compute_density_field(O, self.kernel_cfg.density_radius)

        # 速度衰减场
        eta = compute_speed_decay_field(V, mask, self.v_ref, self.kernel_cfg.speed_sigma)

        # 方向分解
        layers = decompose_direction(Theta, mask)

        # 冲突场
        C, pair_results, R = compute_conflict_field(
            layers, self.kernels, self.conflict_pairs
        )

        # 归因分数
        influences = compute_vehicle_influence(
            vehicles, R, self.grid_cfg, self.conflict_pairs
        )

        # Phi 场
        Phi = compute_phi_field(rho, eta, C, self.w_rho, self.w_v, self.w_c)

        return ConflictResult(
            phi_field=Phi,
            conflict_field=C,
            density_field=rho,
            speed_decay_field=eta,
            directional_fields=layers,
            influence_fields=R,
            pair_results=pair_results,
            influences=influences,
            grid_cfg=self.grid_cfg,
            vehicles=vehicles,
            kernel_cfg=self.kernel_cfg,
            conflict_pairs=self.conflict_pairs,
            kernels=self.kernels,
            w_rho=self.w_rho,
            w_v=self.w_v,
            w_c=self.w_c,
        )


class ConflictResult:
    """冲突分析结果"""

    def __init__(
        self,
        phi_field: np.ndarray,
        conflict_field: np.ndarray,
        density_field: np.ndarray,
        speed_decay_field: np.ndarray,
        directional_fields: list[np.ndarray],
        influence_fields: list[np.ndarray],
        pair_results: dict,
        influences: list[float],
        grid_cfg: GridConfig,
        vehicles: list[dict],
        kernel_cfg: Optional[KernelConfig] = None,
        conflict_pairs: Optional[list[tuple[int, int]]] = None,
        kernels: Optional[list[np.ndarray]] = None,
        w_rho: float = 0.3,
        w_v: float = 0.4,
        w_c: float = 0.3,
    ):
        self.phi_field = phi_field
        self.conflict_field = conflict_field
        self.density_field = density_field
        self.speed_decay_field = speed_decay_field
        self.directional_fields = directional_fields
        self.influence_fields = influence_fields
        self.pair_results = pair_results
        self.influences = influences
        self.grid_cfg = grid_cfg
        self.vehicles = vehicles
        self.kernel_cfg = kernel_cfg or KernelConfig()
        self.conflict_pairs = conflict_pairs or DEFAULT_CONFLICT_PAIRS
        self.kernels = kernels or build_all_directional_kernels(
            DIRECTION_BINS, self.kernel_cfg)
        self.w_rho = w_rho
        self.w_v = w_v
        self.w_c = w_c

    @property
    def phi_max(self) -> float:
        """全场最大 Phi 值"""
        return float(self.phi_field.max())

    @property
    def phi_mean(self) -> float:
        """全场平均 Phi 值"""
        return float(self.phi_field.mean())

    @property
    def conflict_max(self) -> float:
        """全场最大冲突值"""
        return float(self.conflict_field.max())

    @property
    def conflict_hotspots(self) -> list[tuple[int, int, float]]:
        """冲突热点位置 (gy, gx, value)，按冲突值降序"""
        H, W = self.conflict_field.shape
        flat = self.conflict_field.ravel()
        top_indices = np.argsort(flat)[::-1]
        hotspots = []
        for idx in top_indices[:20]:
            val = flat[idx]
            if val <= 0:
                break
            gy, gx = divmod(idx, W)
            hotspots.append((int(gy), int(gx), float(val)))
        return hotspots

    def get_vehicles_ranked_by_influence(self) -> list[tuple[int, float, dict]]:
        """
        按归因分数降序排列车辆。

        Returns
        -------
        ranked : list of (index, influence, vehicle_dict)
        """
        indexed = list(enumerate(self.influences))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(idx, inf, self.vehicles[idx]) for idx, inf in indexed]

    def ablate_vehicle(self, vehicle_index: int) -> 'ConflictResult':
        """
        模拟移除一辆车后的冲突分析结果（用于 ablation 验证）。
        只 copy 受影响的 1 个 bin，其余引用原始字段，减少内存开销。
        """
        v = self.vehicles[vehicle_index]
        gx = int((v['cx'] - self.grid_cfg.origin_x) / self.grid_cfg.cell_size_m)
        gy = int((v['cy'] - self.grid_cfg.origin_y) / self.grid_cfg.cell_size_m)

        heading = v['heading_deg'] % 360.0
        bin_size = 360.0 / DIRECTION_BINS
        k_i = int(heading / bin_size) % DIRECTION_BINS

        # 只 copy 受影响的 1 个方向层，其余引用原始（不可变）
        layers_new = list(self.directional_fields)  # 浅拷贝列表
        if 0 <= gx < self.grid_cfg.grid_size and 0 <= gy < self.grid_cfg.grid_size:
            layers_new[k_i] = self.directional_fields[k_i].copy()
            layers_new[k_i][gy, gx] = 0.0

        # 只重算受影响方向的 R
        R_new = list(self.influence_fields)  # 浅拷贝列表
        R_new[k_i] = cv2.filter2D(layers_new[k_i], ddepth=-1, kernel=self.kernels[k_i])

        # 重算冲突场
        C_new = np.zeros_like(self.conflict_field)
        for (k1, k2) in self.conflict_pairs:
            C_new += R_new[k1] * R_new[k2]

        # 重算 Phi 场
        Phi_new = compute_phi_field(
            self.density_field, self.speed_decay_field, C_new,
            self.w_rho, self.w_v, self.w_c,
        )

        # 重算归因
        influences_new = compute_vehicle_influence(
            self.vehicles, R_new, self.grid_cfg,
            self.conflict_pairs,
        )
        influences_new[vehicle_index] = 0.0  # 已移除

        return ConflictResult(
            phi_field=Phi_new,
            conflict_field=C_new,
            density_field=self.density_field,
            speed_decay_field=self.speed_decay_field,
            directional_fields=layers_new,
            influence_fields=R_new,
            pair_results={},
            influences=influences_new,
            grid_cfg=self.grid_cfg,
            vehicles=self.vehicles,
        )
