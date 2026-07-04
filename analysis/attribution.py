"""
车辆拥堵归因模块 (Vehicle Congestion Attribution)

替代原 gamma_id + lambda_k 分组方法，直接通过卷积影响力场进行单车归因。

原方法：gamma_id(交织次数) → 分组 → lambda_k(组内交织密度) → 排序 → ablation
新方法：Influence_i = R[k_i] × Σ R[k'] → 直接排序 → ablation（单次卷积）

优势：
1. 不需要分组，直接单车归因
2. 包含空间信息（路口中心的交织 > 边缘的交织）
3. Ablation 只需重算 1 个方向的 1 次卷积
"""

import numpy as np
from typing import Optional

from core.conflict import (
    ConflictAnalyzer,
    ConflictResult,
    GridConfig,
    KernelConfig,
    DEFAULT_CONFLICT_PAIRS,
    DIRECTION_BINS,
    build_all_directional_kernels,
    compute_vehicle_influence,
)


# ============================================================
# 归因结果
# ============================================================

class AttributionResult:
    """单车归因结果"""

    def __init__(
        self,
        vehicle_index: int,
        vehicle_info: dict,
        influence_score: float,
        direction_bin: int,
        grid_position: tuple[int, int],
        rank: int = 0,
    ):
        self.vehicle_index = vehicle_index
        self.vehicle_info = vehicle_info
        self.influence_score = influence_score
        self.direction_bin = direction_bin
        self.grid_position = grid_position  # (gy, gx)
        self.rank = rank

    def __repr__(self):
        return (
            f"Attribution(id={self.vehicle_index}, "
            f"influence={self.influence_score:.4f}, "
            f"dir_bin={self.direction_bin}, rank={self.rank})"
        )


# ============================================================
# 拥堵归因分析器
# ============================================================

class CongestionAttributor:
    """
    拥堵归因分析器。

    核心功能：
    1. 基于卷积影响力场计算每辆车的拥堵贡献分数
    2. 按贡献排序，识别最致堵车辆
    3. 支持逐车 ablation 验证

    Usage
    -----
    >>> attributor = CongestionAttributor(grid_size=64, world_width_m=40)
    >>> result = attributor.attribute(vehicles)
    >>> for attr in result[:5]:
    ...     print(f"车辆 {attr.vehicle_index}: Influence={attr.influence_score:.4f}")
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
        v_ref: float = 6.2,
    ):
        self.analyzer = ConflictAnalyzer(
            grid_size=grid_size,
            world_width_m=world_width_m,
            world_height_m=world_height_m,
            origin_x=origin_x,
            origin_y=origin_y,
            kernel_cfg=kernel_cfg,
            conflict_pairs=conflict_pairs,
            v_ref=v_ref,
        )

    def attribute(
        self,
        vehicles: list[dict],
        top_k: Optional[int] = None,
    ) -> tuple[list[AttributionResult], ConflictResult]:
        """
        计算每辆车的拥堵归因分数并排序。

        Parameters
        ----------
        vehicles : list[dict]
            每个 dict 包含 cx, cy (m), speed_mps, heading_deg, 可选 track_id, label
        top_k : int, optional
            只返回前 K 个结果

        Returns
        -------
        attributions : list[AttributionResult], 按 influence 降序
        conflict_result : ConflictResult, 完整的冲突分析结果
        """
        # 运行冲突分析
        conflict_result = self.analyzer.analyze(vehicles)

        # 构建归因结果
        bin_size = 360.0 / DIRECTION_BINS
        grid_cfg = conflict_result.grid_cfg

        attributions = []
        for i, (v, inf) in enumerate(zip(vehicles, conflict_result.influences)):
            gx = int((v['cx'] - grid_cfg.origin_x) / grid_cfg.cell_size_m)
            gy = int((v['cy'] - grid_cfg.origin_y) / grid_cfg.cell_size_m)
            heading = v.get('heading_deg', 0) % 360.0
            k_i = int(heading / bin_size) % DIRECTION_BINS

            attr = AttributionResult(
                vehicle_index=i,
                vehicle_info=v,
                influence_score=inf,
                direction_bin=k_i,
                grid_position=(gy, gx),
            )
            attributions.append(attr)

        # 按 influence 降序排序
        attributions.sort(key=lambda a: a.influence_score, reverse=True)

        # 设置 rank
        for rank, attr in enumerate(attributions):
            attr.rank = rank + 1

        if top_k is not None:
            attributions = attributions[:top_k]

        return attributions, conflict_result

    def ablate_and_reevaluate(
        self,
        vehicles: list[dict],
        ablation_levels: int = 3,
    ) -> list[dict]:
        """
        方向维度消融实验：按方向 bin 分组，逐级移除交织最严重的方向组。

        注意：此方法按方向 bin 分组消融（回答"哪个方向贡献最大"）。
        如需按单车消融（回答"哪辆车是主因"），请使用 AblationStudy.run()。

        Parameters
        ----------
        vehicles : list[dict]
        ablation_levels : int
            消融级数（移除前 K 个方向组）

        Returns
        -------
        ablation_results : list of dict
            每个 dict 包含：
            - level: ablation 级数
            - removed_indices: 被移除的车辆索引
            - phi_before: 移除前的 Phi
            - phi_after: 移除后的 Phi
            - phi_reduction: Phi 下降量
            - conflict_before: 移除前的冲突数
            - conflict_after: 移除后的冲突数
        """
        # 初始分析
        attributions, conflict_result = self.attribute(vehicles)
        phi_before = conflict_result.phi_max
        conflict_before = float(conflict_result.conflict_field.sum())

        ablation_results = []

        # 按 influence 分组（相同 bin 的车辆为一组）
        bin_groups = {}
        for attr in attributions:
            bin_key = attr.direction_bin
            if bin_key not in bin_groups:
                bin_groups[bin_key] = []
            bin_groups[bin_key].append(attr)

        # 按组的总 influence 排序
        sorted_groups = sorted(
            bin_groups.items(),
            key=lambda kv: sum(a.influence_score for a in kv[1]),
            reverse=True,
        )

        # 逐级移除
        removed_indices = set()
        for level in range(min(ablation_levels, len(sorted_groups))):
            dir_bin, group_attrs = sorted_groups[level]
            group_indices = [a.vehicle_index for a in group_attrs]
            removed_indices.update(group_indices)

            # 重新分析（排除已移除车辆）
            remaining_vehicles = [
                v for i, v in enumerate(vehicles)
                if i not in removed_indices
            ]

            if len(remaining_vehicles) == 0:
                break

            _, result_after = self.attribute(remaining_vehicles)
            phi_after = result_after.phi_max
            conflict_after = float(result_after.conflict_field.sum())

            ablation_results.append({
                'level': level + 1,
                'removed_indices': list(removed_indices),
                'removed_count': len(removed_indices),
                'remaining_count': len(remaining_vehicles),
                'phi_before': phi_before,
                'phi_after': phi_after,
                'phi_reduction': phi_before - phi_after,
                'phi_reduction_ratio': (phi_before - phi_after) / max(phi_before, 1e-6),
                'conflict_before': conflict_before,
                'conflict_after': conflict_after,
                'conflict_reduction': conflict_before - conflict_after,
                'removed_direction_bin': dir_bin,
                'removed_vehicle_ids': [
                    vehicles[i].get('track_id', i) for i in group_indices
                ],
            })

        return ablation_results
