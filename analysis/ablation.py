"""
消融实验模块 (Ablation Study)

负责：
- 按归因分数逐级移除车辆
- 重算移除后的 Phi 和冲突场
- 输出消融对比图表和 CSV
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Optional

from core.phi import compute_phi, RiskParams
from analysis.attribution import CongestionAttributor


# ============================================================
# 消融实验
# ============================================================

class AblationStudy:
    """
    消融实验：逐级移除高归因车辆，量化其对拥堵的贡献。

    Usage
    -----
    >>> study = AblationStudy(grid_size=64, world_width_m=40)
    >>> results = study.run(vehicles, risk_params, levels=3)
    >>> study.export_csv(results, 'ablation_results.csv')
    """

    def __init__(
        self,
        grid_size: int = 64,
        world_width_m: float = 40.0,
        world_height_m: float = 40.0,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
        v_ref: float = 6.2,
    ):
        self.attributor = CongestionAttributor(
            grid_size=grid_size,
            world_width_m=world_width_m,
            world_height_m=world_height_m,
            origin_x=origin_x,
            origin_y=origin_y,
            v_ref=v_ref,
        )

    def run(
        self,
        vehicles: list[dict],
        risk_params: RiskParams,
        levels: int = 3,
    ) -> list[dict]:
        """
        运行消融实验（增量模式）。

        按归因分数从高到低逐辆移除交织最严重的车辆，
        使用 ConflictResult.ablate_vehicle() 增量更新冲突场，
        避免每级都重新运行完整冲突分析。

        Parameters
        ----------
        vehicles : list[dict]
        risk_params : RiskParams
        levels : 消融级数（移除几辆车）

        Returns
        -------
        results : list of dict
        """
        if not vehicles:
            return []

        # 初始归因（一次完整分析）
        attributions, conflict_result = self.attributor.attribute(vehicles)

        conflict_before = float(conflict_result.conflict_field.sum())
        n_conflict_before = int((conflict_result.conflict_field > 0).sum())

        # 计算初始标量 Phi
        active_vehicles = [v for v in vehicles if v.get('speed_mps', 0) > 0 or v.get('heading_deg', 0) != 0]
        avg_speed = np.mean([v.get('speed_mps', 0) for v in active_vehicles]) if active_vehicles else 0.0
        active_count = len(vehicles)
        scalar_phi_before = compute_phi(active_count, avg_speed, risk_params)

        # 按 influence_score 从高到低排序（交织最严重的车排最前）
        ranked = sorted(
            enumerate(attributions),
            key=lambda kv: kv[1].influence_score,
            reverse=True,
        )

        results = []
        removed_indices = set()
        current_result = conflict_result  # 增量更新，不重新分析

        for level in range(min(levels, len(ranked))):
            idx, attr = ranked[level]
            removed_indices.add(idx)

            if len(removed_indices) >= len(vehicles):
                break

            # 增量消融：只重算受影响的方向 bin
            current_result = current_result.ablate_vehicle(idx)

            conflict_after = float(current_result.conflict_field.sum())
            n_conflict_after = int((current_result.conflict_field > 0).sum())

            # 冲突场变化率（核心指标）
            conflict_reduction = conflict_before - conflict_after
            conflict_reduction_pct = conflict_reduction / max(conflict_before, 1e-6) * 100

            # 重算标量 Phi（用剩余车辆的实际速度）
            remaining = [v for i, v in enumerate(vehicles) if i not in removed_indices]
            active_remaining = [v for v in remaining if v.get('speed_mps', 0) > 0 or v.get('heading_deg', 0) != 0]
            avg_speed_ab = np.mean([v.get('speed_mps', 0) for v in active_remaining]) if active_remaining else 0.0
            active_after = len(remaining)
            scalar_phi_after = compute_phi(active_after, avg_speed_ab, risk_params)
            scalar_phi_reduction = scalar_phi_before - scalar_phi_after

            results.append({
                'level': level + 1,
                'removed_vehicle_id': vehicles[idx].get('track_id', idx),
                'removed_influence': attr.influence_score,
                'removed_direction_bin': attr.direction_bin,
                'removed_count': len(removed_indices),
                'remaining_count': len(remaining),
                'conflict_field_before': conflict_before,
                'conflict_field_after': conflict_after,
                'conflict_field_reduction': conflict_reduction,
                'conflict_field_reduction_pct': conflict_reduction_pct,
                'n_conflict_cells_before': n_conflict_before,
                'n_conflict_cells_after': n_conflict_after,
                'scalar_phi_before': scalar_phi_before,
                'scalar_phi_after': scalar_phi_after,
                'scalar_phi_reduction': scalar_phi_reduction,
                'removed_vehicle_ids': [
                    vehicles[i].get('track_id', i) for i in removed_indices
                ],
                'removed_influence_scores': [
                    attributions[i].influence_score for i in removed_indices
                ],
            })

            # 更新基准（累计效果）
            conflict_before = conflict_after
            n_conflict_before = n_conflict_after
            scalar_phi_before = scalar_phi_after

        return results

    def export_csv(self, results: list[dict], out_path: str | Path):
        """导出消融结果为 CSV"""
        if not results:
            return

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            'level', 'removed_vehicle_id', 'removed_influence', 'removed_direction_bin',
            'removed_count', 'remaining_count',
            'conflict_field_before', 'conflict_field_after',
            'conflict_field_reduction', 'conflict_field_reduction_pct',
            'n_conflict_cells_before', 'n_conflict_cells_after',
            'scalar_phi_before', 'scalar_phi_after', 'scalar_phi_reduction',
            'removed_vehicle_ids',
        ]

        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in results:
                row_copy = dict(row)
                if 'removed_vehicle_ids' in row_copy:
                    row_copy['removed_vehicle_ids'] = str(row_copy['removed_vehicle_ids'])
                writer.writerow(row_copy)

    def export_json(self, results: list[dict], out_path: str | Path):
        """导出消融结果为 JSON"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 序列化处理
        def _serialize(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean = []
        for r in results:
            clean.append({k: _serialize(v) for k, v in r.items()})

        out_path.write_text(
            json.dumps(clean, indent=2, ensure_ascii=False, default=_serialize),
            encoding='utf-8',
        )


