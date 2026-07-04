"""
BEV 变换单元测试
"""

import numpy as np
import pytest
from core.bev_transform import load_homography, pixel_to_world


class TestPixelToWorld:
    def test_origin_maps_to_origin(self):
        """单位矩阵下，原点映射到原点"""
        H = np.eye(3, dtype=np.float64)
        result = pixel_to_world(H, (0, 0))
        assert abs(result[0]) < 1e-6
        assert abs(result[1]) < 1e-6

    def test_translation(self):
        """纯平移变换"""
        H = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float64)
        result = pixel_to_world(H, (5, 5))
        assert abs(result[0] - 15) < 1e-6
        assert abs(result[1] - 25) < 1e-6

    def test_scaling(self):
        """缩放变换"""
        H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=np.float64)
        result = pixel_to_world(H, (10, 10))
        assert abs(result[0] - 20) < 1e-6
        assert abs(result[1] - 20) < 1e-6


class TestLoadHomography:
    def test_load_example_config(self):
        """加载示例标定文件"""
        h, img_pts, world_pts = load_homography('configs/homography_points_example.json')
        assert h.shape == (3, 3)
        assert img_pts.shape[0] == 4
        assert world_pts.shape[0] == 4

    def test_load_yongdukou_config(self):
        """加载 yongdukou 标定文件"""
        h, img_pts, world_pts = load_homography('configs/homography_yongdukou.json')
        assert h.shape == (3, 3)
        # 4 个唯一点
        unique = set(map(tuple, img_pts.astype(int)))
        assert len(unique) == 4
