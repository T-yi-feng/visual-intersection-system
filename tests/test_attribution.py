"""
analysis/attribution.py 单元测试

覆盖：归因排序、空输入、消融实验
"""

import sys
import pytest

sys.path.insert(0, '.')

from analysis.attribution import CongestionAttributor, AttributionResult


class TestCongestionAttributor:
    def test_basic_attribution(self):
        attributor = CongestionAttributor(grid_size=64, world_width_m=40, world_height_m=40)
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
            {'cx': 20.0, 'cy': 21.0, 'speed_mps': 2.0, 'heading_deg': 180},
        ]
        attributions, conflict_result = attributor.attribute(vehicles)
        assert len(attributions) == 3
        # 应该按 influence 降序排列
        for i in range(len(attributions) - 1):
            assert attributions[i].influence_score >= attributions[i + 1].influence_score

    def test_empty_vehicles(self):
        attributor = CongestionAttributor(grid_size=64, world_width_m=40, world_height_m=40)
        attributions, conflict_result = attributor.attribute([])
        assert len(attributions) == 0
        assert conflict_result.phi_max == 0

    def test_top_k(self):
        attributor = CongestionAttributor(grid_size=64, world_width_m=40, world_height_m=40)
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
            {'cx': 20.0, 'cy': 21.0, 'speed_mps': 2.0, 'heading_deg': 180},
        ]
        attributions, _ = attributor.attribute(vehicles, top_k=2)
        assert len(attributions) == 2

    def test_rank_assignment(self):
        attributor = CongestionAttributor(grid_size=64, world_width_m=40, world_height_m=40)
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
        ]
        attributions, _ = attributor.attribute(vehicles)
        assert attributions[0].rank == 1
        assert attributions[1].rank == 2


class TestAttributionResult:
    def test_repr(self):
        attr = AttributionResult(
            vehicle_index=0,
            vehicle_info={'track_id': 1, 'heading_deg': 90},
            influence_score=0.5,
            direction_bin=2,
            grid_position=(32, 32),
            rank=1,
        )
        repr_str = repr(attr)
        assert 'id=0' in repr_str
        assert '0.5000' in repr_str


class TestAblation:
    def test_ablate_and_reevaluate(self):
        attributor = CongestionAttributor(grid_size=64, world_width_m=40, world_height_m=40)
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
            {'cx': 20.0, 'cy': 21.0, 'speed_mps': 2.0, 'heading_deg': 180},
            {'cx': 19.0, 'cy': 20.0, 'speed_mps': 5.0, 'heading_deg': 270},
        ]
        results = attributor.ablate_and_reevaluate(vehicles, ablation_levels=2)
        assert len(results) >= 1
        assert results[0]['level'] == 1
        assert 'phi_before' in results[0]
        assert 'phi_after' in results[0]
        assert 'removed_count' in results[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
