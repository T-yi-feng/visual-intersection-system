"""
core/conflict.py 单元测试

覆盖：网格散布、方向分解、核构建、冲突场计算、归因分数、ablate_vehicle
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '.')

from core.conflict import (
    GridConfig, KernelConfig,
    scatter_vehicles_to_grid,
    compute_density_field,
    compute_speed_decay_field,
    decompose_direction,
    build_directional_kernel,
    build_all_directional_kernels,
    compute_conflict_field,
    compute_vehicle_influence,
    compute_phi_field,
    ConflictAnalyzer,
    ConflictResult,
    DEFAULT_CONFLICT_PAIRS,
    DIRECTION_BINS,
)


# ============================================================
# GridConfig
# ============================================================

class TestGridConfig:
    def test_from_world_bounds(self):
        cfg = GridConfig.from_world_bounds(40.0, 40.0, grid_size=64)
        assert cfg.grid_size == 64
        assert abs(cfg.cell_size_m - 0.625) < 1e-6

    def test_custom_origin(self):
        cfg = GridConfig(64, 0.5, origin_x=10.0, origin_y=20.0)
        assert cfg.origin_x == 10.0
        assert cfg.origin_y == 20.0


# ============================================================
# scatter_vehicles_to_grid
# ============================================================

class TestScatterVehicles:
    def test_basic_scatter(self):
        cfg = GridConfig(64, 0.625)
        vehicles = [
            {'cx': 20.0, 'cy': 20.0, 'speed_mps': 5.0, 'heading_deg': 90},
        ]
        O, V, Theta, mask = scatter_vehicles_to_grid(vehicles, cfg)

        # 车辆应该落在网格中心附近
        gx = int(20.0 / 0.625)
        gy = int(20.0 / 0.625)
        assert O[gy, gx] == 1.0
        assert V[gy, gx] == 5.0
        assert mask[gy, gx] == 1.0

    def test_out_of_bounds(self):
        cfg = GridConfig(64, 0.625)
        vehicles = [{'cx': 100.0, 'cy': 100.0, 'speed_mps': 0, 'heading_deg': 0}]
        O, V, Theta, mask = scatter_vehicles_to_grid(vehicles, cfg)
        assert O.sum() == 0  # 车辆在网格外

    def test_empty_vehicles(self):
        cfg = GridConfig(64, 0.625)
        O, V, Theta, mask = scatter_vehicles_to_grid([], cfg)
        assert O.sum() == 0


# ============================================================
# compute_density_field
# ============================================================

class TestDensityField:
    def test_single_vehicle(self):
        O = np.zeros((64, 64), dtype=np.float32)
        O[32, 32] = 1.0
        rho = compute_density_field(O, kernel_radius=3)
        assert rho.max() > 0
        assert rho[32, 32] > 0  # 中心有密度

    def test_empty_field(self):
        O = np.zeros((64, 64), dtype=np.float32)
        rho = compute_density_field(O, kernel_radius=3)
        assert rho.max() == 0


# ============================================================
# decompose_direction
# ============================================================

class TestDirectionDecomposition:
    def test_north_heading(self):
        Theta = np.zeros((64, 64), dtype=np.float32)  # 0 rad = 北
        mask = np.ones((64, 64), dtype=np.float32)
        layers = decompose_direction(Theta, mask)
        assert len(layers) == 12
        assert layers[0].sum() > 0   # bin 0 (北, 0°) 应有值
        assert layers[6].sum() == 0  # bin 6 (南, 180°) 应无值

    def test_east_heading(self):
        Theta = np.full((64, 64), np.pi / 2, dtype=np.float32)  # 90° = 东
        mask = np.ones((64, 64), dtype=np.float32)
        layers = decompose_direction(Theta, mask)
        assert layers[3].sum() > 0  # bin 3 (东, 90°) 应有值


# ============================================================
# build_directional_kernel
# ============================================================

class TestDirectionalKernel:
    def test_kernel_shape(self):
        K = build_directional_kernel(0, arrow_half_len=15, kernel_half_width=3,
                                      sigma_along=5.0, sigma_perp=1.0)
        assert K.shape == (31, 31)  # 正方形核 (2*15+1)

    def test_kernel_normalized(self):
        K = build_directional_kernel(0, arrow_half_len=10, kernel_half_width=3,
                                      sigma_along=5.0, sigma_perp=1.0)
        assert abs(K.sum() - 1.0) < 1e-5

    def test_kernel_asymmetric(self):
        """核前向应比后向长（沿方向不对称）"""
        K = build_directional_kernel(0, arrow_half_len=10, kernel_half_width=3,
                                      sigma_along=5.0, sigma_perp=1.0)
        h, w = K.shape
        mid = w // 2
        front = K[:, mid:].sum()   # 前向（沿方向正半轴）
        back  = K[:, :mid].sum()   # 后向（沿方向负半轴）
        assert front > back * 1.5, f"front={front:.4f} should be >> back={back:.4f}"


# ============================================================
# compute_conflict_field
# ============================================================

class TestConflictField:
    def test_no_conflict(self):
        # 所有车辆朝同一方向，不应有冲突（同 bin 之间无冲突对）
        layers = [np.zeros((64, 64), dtype=np.float32) for _ in range(12)]
        layers[0][32, 32] = 1.0  # 只有朝北的车
        kernels = build_all_directional_kernels()
        C, _, R = compute_conflict_field(layers, kernels, conflict_pairs=[])
        assert C.max() == 0

    def test_cross_conflict(self):
        # 朝北(bin0) 和朝东(bin3) 的车在同一点
        layers = [np.zeros((64, 64), dtype=np.float32) for _ in range(12)]
        layers[0][32, 32] = 1.0  # 朝北
        layers[3][32, 32] = 1.0  # 朝东
        kernels = build_all_directional_kernels()
        C, _, R = compute_conflict_field(layers, kernels, DEFAULT_CONFLICT_PAIRS)
        assert C.max() > 0  # 应有冲突


# ============================================================
# compute_vehicle_influence
# ============================================================

class TestVehicleInfluence:
    def test_no_influence_for_isolated_vehicle(self):
        # 孤立车辆，附近无冲突方向的车
        R = [np.zeros((64, 64), dtype=np.float32) for _ in range(12)]
        R[0][32, 32] = 0.1  # 只有朝北的影响力

        cfg = GridConfig(64, 0.625)
        vehicles = [{'cx': 20.0, 'cy': 20.0, 'heading_deg': 0, 'speed_mps': 3.0}]
        # 用空冲突对测试：孤立车辆无冲突
        influences = compute_vehicle_influence(vehicles, R, cfg, conflict_pairs=[])
        assert influences[0] == 0


# ============================================================
# ConflictAnalyzer
# ============================================================

class TestConflictAnalyzer:
    def test_analyze_basic(self):
        analyzer = ConflictAnalyzer(grid_size=64, world_width_m=40, world_height_m=40)
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
        ]
        result = analyzer.analyze(vehicles)
        assert isinstance(result, ConflictResult)
        assert result.phi_max >= 0
        assert len(result.influences) == 2

    def test_analyze_empty(self):
        analyzer = ConflictAnalyzer(grid_size=64, world_width_m=40, world_height_m=40)
        result = analyzer.analyze([])
        assert result.phi_max == 0
        assert len(result.influences) == 0


# ============================================================
# ConflictResult.ablate_vehicle
# ============================================================

class TestAblateVehicle:
    def test_ablate_uses_stored_config(self):
        """验证 ablate_vehicle 使用存储的配置而非默认值"""
        custom_pairs = [(0, 3), (0, 9)]  # 自定义冲突对 (N/E, N/W) - 12-bin
        analyzer = ConflictAnalyzer(
            grid_size=64, world_width_m=40, world_height_m=40,
            conflict_pairs=custom_pairs,
        )
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
        ]
        result = analyzer.analyze(vehicles)

        # 验证配置被正确存储
        assert result.conflict_pairs == custom_pairs
        assert result.kernel_cfg is not None

        # ablate 不应崩溃
        result2 = result.ablate_vehicle(0)
        assert isinstance(result2, ConflictResult)

    def test_ablate_removes_vehicle(self):
        analyzer = ConflictAnalyzer(grid_size=64, world_width_m=40, world_height_m=40)
        vehicles = [
            {'cx': 20.0, 'cy': 19.0, 'speed_mps': 3.0, 'heading_deg': 0},
            {'cx': 21.0, 'cy': 20.0, 'speed_mps': 4.0, 'heading_deg': 90},
            {'cx': 20.0, 'cy': 21.0, 'speed_mps': 2.0, 'heading_deg': 180},
        ]
        result = analyzer.analyze(vehicles)
        result2 = result.ablate_vehicle(0)

        # 移除后归因分数应该变化
        assert result2.influences[0] == 0  # 被移除的车


# ============================================================
# compute_phi_field
# ============================================================

class TestPhiField:
    def test_basic_phi(self):
        rho = np.full((64, 64), 0.5, dtype=np.float32)
        eta = np.full((64, 64), 0.3, dtype=np.float32)
        C = np.full((64, 64), 0.2, dtype=np.float32)
        Phi = compute_phi_field(rho, eta, C)
        assert 0 <= Phi.min() <= Phi.max() <= 1

    def test_zero_conflict(self):
        rho = np.full((64, 64), 0.5, dtype=np.float32)
        eta = np.full((64, 64), 0.3, dtype=np.float32)
        C = np.zeros((64, 64), dtype=np.float32)
        Phi = compute_phi_field(rho, eta, C)
        # 冲突为 0 时，Phi 应该只由 rho 和 eta 决定
        expected = 0.3 * 0.5 + 0.4 * 0.3  # w_rho * rho + w_v * eta
        assert abs(Phi[0, 0] - expected) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
