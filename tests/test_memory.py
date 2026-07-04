"""
core/memory.py 单元测试

覆盖：颜色直方图、相似度计算、运动连续性、IoU、匹配打分、MemoryBank
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '.')

from core.memory import (
    extract_color_histogram,
    color_similarity,
    motion_continuity_score,
    compute_match_score,
    _compute_iou,
    VehicleMemory,
    MemoryBank,
)


# ============================================================
# 颜色直方图
# ============================================================

class TestColorHistogram:
    def test_extracts_correct_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist = extract_color_histogram(frame, (100, 100, 200, 200))
        assert hist.shape == (32 * 32,)  # 32×32 bins

    def test_same_region_similar(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist1 = extract_color_histogram(frame, (100, 100, 200, 200))
        hist2 = extract_color_histogram(frame, (100, 100, 200, 200))
        assert color_similarity(hist1, hist2) > 0.99

    def test_out_of_bounds_returns_zeros(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist = extract_color_histogram(frame, (-10, -10, -5, -5))  # 完全越界
        assert hist.sum() == 0.0

    def test_empty_roi_returns_zeros(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist = extract_color_histogram(frame, (100, 100, 100, 100))  # 零面积
        assert hist.sum() == 0.0


# ============================================================
# 颜色相似度
# ============================================================

class TestColorSimilarity:
    def test_identical_histograms(self):
        h = np.random.rand(1024).astype(np.float32)
        h /= h.sum()
        assert color_similarity(h, h) > 0.99

    def test_different_histograms(self):
        h1 = np.zeros(1024, dtype=np.float32)
        h1[:100] = 1.0 / 100
        h2 = np.zeros(1024, dtype=np.float32)
        h2[-100:] = 1.0 / 100
        sim = color_similarity(h1, h2)
        assert sim < 0.5

    def test_none_handling(self):
        assert color_similarity(None, np.zeros(1024)) == 0.0
        assert color_similarity(np.zeros(1024), None) == 0.0
        assert color_similarity(None, None) == 0.0


# ============================================================
# 运动连续性
# ============================================================

class TestMotionContinuity:
    def test_same_position(self):
        score = motion_continuity_score((100, 100), (100, 100))
        assert score == 1.0

    def test近距离高分(self):
        score = motion_continuity_score((100, 100), (110, 100))
        assert score > 0.8

    def test远距离低分(self):
        score = motion_continuity_score((100, 100), (500, 500))
        assert score < 0.2

    def test超过最大距离(self):
        score = motion_continuity_score((0, 0), (200, 0), max_dist=100)
        assert score == 0.0


# ============================================================
# IoU
# ============================================================

class TestIoU:
    def test_perfect_overlap(self):
        assert _compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert _compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = _compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        # intersection=25, union=175, iou=25/175≈0.143
        assert 0.1 < iou < 0.2


# ============================================================
# 匹配打分
# ============================================================

class TestMatchScore:
    def test_perfect_match(self):
        mem = VehicleMemory(
            track_id=1, last_seen_frame=0,
            bbox=(100, 100, 200, 200), center=(150, 150),
            size=(100, 100), heading=0.0,
        )
        score = compute_match_score(
            (100, 100, 200, 200), None, (150, 150), mem,
            w_iou=1.0, w_color=0.0, w_motion=0.0,
        )
        assert score == 1.0

    def test_no_match(self):
        mem = VehicleMemory(
            track_id=1, last_seen_frame=0,
            bbox=(100, 100, 200, 200), center=(150, 150),
            size=(100, 100), heading=0.0,
        )
        score = compute_match_score(
            (500, 500, 600, 600), None, (550, 550), mem,
            w_iou=0.4, w_color=0.3, w_motion=0.3,
        )
        assert score < 0.3


# ============================================================
# MemoryBank
# ============================================================

class TestMemoryBank:
    def test_store_and_expire(self):
        bank = MemoryBank(max_age=3)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 第0帧：存入
        vehicles = [{'track_id': 1, 'bbox': (10, 10, 30, 30),
                      'center': (20, 20), 'label': 'car', 'heading_deg': 0}]
        bank.match_and_restore(vehicles, fake_frame, 0)
        assert bank.size == 1

        # 第1-2帧：车辆消失，记忆保留
        bank.match_and_restore([], fake_frame, 1)
        bank.match_and_restore([], fake_frame, 2)
        assert bank.size == 1

        # 第4帧：超过max_age，记忆过期
        bank.match_and_restore([], fake_frame, 4)
        assert bank.size == 0

    def test_id_recovery(self):
        """模拟遮挡后ID恢复"""
        bank = MemoryBank(max_age=5, match_threshold=0.3)
        # 创建一个纯色帧（蓝色）
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # 蓝色

        # 第0帧：车辆出现
        vehicles = [{'track_id': 1, 'bbox': (50, 50, 100, 80),
                      'center': (75, 65), 'label': 'car', 'heading_deg': 0}]
        result = bank.match_and_restore(vehicles, frame, 0)
        assert result[0]['track_id'] == 1

        # 第1帧：车辆消失（遮挡）
        bank.match_and_restore([], frame, 1)
        assert bank.size == 1

        # 第2帧：车辆重新出现，位置略有移动
        vehicles2 = [{'track_id': 2, 'bbox': (55, 55, 105, 85),
                       'center': (80, 70), 'label': 'car', 'heading_deg': 0}]
        result2 = bank.match_and_restore(vehicles2, frame, 2)
        # 应该恢复为 track_id=1
        assert result2[0]['track_id'] == 1

    def test_no_recovery_for_different_color(self):
        """不同颜色的车不应匹配"""
        bank = MemoryBank(max_age=5, match_threshold=0.5,
                          w_iou=0.2, w_color=0.6, w_motion=0.2)

        # 红色车
        frame_red = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_red[:, :, 2] = 200
        vehicles = [{'track_id': 1, 'bbox': (50, 50, 100, 80),
                      'center': (75, 65), 'label': 'car', 'heading_deg': 0}]
        bank.match_and_restore(vehicles, frame_red, 0)

        # 消失
        bank.match_and_restore([], frame_red, 1)

        # 蓝色车出现（同一位置）
        frame_blue = np.zeros((200, 200, 3), dtype=np.uint8)
        frame_blue[:, :, 0] = 200
        vehicles2 = [{'track_id': 2, 'bbox': (50, 50, 100, 80),
                       'center': (75, 65), 'label': 'car', 'heading_deg': 0}]
        result = bank.match_and_restore(vehicles2, frame_blue, 2)
        # 不同颜色，不应恢复
        assert result[0]['track_id'] == 2

    def test_clear(self):
        bank = MemoryBank()
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vehicles = [{'track_id': 1, 'bbox': (10, 10, 30, 30),
                      'center': (20, 20), 'label': 'car', 'heading_deg': 0}]
        bank.match_and_restore(vehicles, fake_frame, 0)
        assert bank.size == 1
        bank.clear()
        assert bank.size == 0

    def test_multiple_vehicles(self):
        """多辆车同时存在"""
        bank = MemoryBank(max_age=5, match_threshold=0.3)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        vehicles = [
            {'track_id': 1, 'bbox': (10, 10, 40, 40), 'center': (25, 25),
             'label': 'car', 'heading_deg': 0},
            {'track_id': 2, 'bbox': (100, 100, 140, 130), 'center': (120, 115),
             'label': 'bus', 'heading_deg': 90},
        ]
        bank.match_and_restore(vehicles, frame, 0)
        assert bank.size == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
