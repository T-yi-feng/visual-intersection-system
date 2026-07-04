"""
core/motion.py 单元测试

覆盖：运动状态分类、速度计算
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, '.')

from core.motion import summarize_motion_stats, default_track_motion_state, MotionState


class TestSummarizeMotionStats:
    def test_moving_vehicle(self):
        # 车辆有明显位移（时间跨度需 >= sample_dt_s=0.5s）
        trajectories = {
            1: [(0.0, 100, 100), (0.6, 120, 100)],  # 20px 位移
        }
        current_meta = {
            1: {'center': (110, 100), 'label': 'car'},
        }
        track_motion_state = {}
        stats = summarize_motion_stats(
            trajectories, current_meta, None, track_motion_state,
            frame_shape=(1080, 1920),
        )
        assert stats['moving_count'] >= 1
        assert stats['active_count'] >= 1

    def test_stationary_vehicle(self):
        # 车辆无位移（时间跨度需 >= sample_dt_s=0.5s）
        trajectories = {
            1: [(0.0, 100, 100), (0.6, 100, 100)],
        }
        current_meta = {
            1: {'center': (100, 100), 'label': 'car'},
        }
        track_motion_state = {}
        stats = summarize_motion_stats(
            trajectories, current_meta, None, track_motion_state,
            frame_shape=(1080, 1920),
        )
        assert stats['moving_count'] == 0
        assert stats['stationary_count'] >= 1

    def test_empty_trajectories(self):
        stats = summarize_motion_stats(
            {}, {}, None, {},
            frame_shape=(1080, 1920),
        )
        assert stats['total_count'] == 0
        assert stats['avg_speed_mps'] == 0

    def test_parked_initial(self):
        # 车辆长时间未移动
        state = default_track_motion_state()
        state['initial_stationary_s'] = 35.0  # 超过阈值
        state['parked_initial'] = True

        trajectories = {
            1: [(0.0, 100, 100), (1.0, 100, 100)],
        }
        current_meta = {
            1: {'center': (100, 100), 'label': 'car'},
        }
        track_motion_state = {1: state}
        stats = summarize_motion_stats(
            trajectories, current_meta, None, track_motion_state,
            frame_shape=(1080, 1920),
            initial_stationary_exclude_seconds=30.0,
        )
        assert stats['parked_count'] >= 1


class TestMotionState:
    def test_constants(self):
        assert MotionState.MOVING == 'moving'
        assert MotionState.STATIONARY == 'stationary'
        assert MotionState.PARKED_INITIAL == 'parked_initial'


class TestDefaultTrackMotionState:
    def test_returns_dict(self):
        state = default_track_motion_state()
        assert isinstance(state, dict)
        assert state['ever_moved'] is False
        assert state['speed_mps'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
