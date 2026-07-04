"""
core/phi.py 单元测试

覆盖：Phi 计算、事件跟踪、风险参数
"""

import sys
import pytest

sys.path.insert(0, '.')

from core.phi import compute_phi, RiskParams, PhiEventTracker


class TestComputePhi:
    def test_basic_phi(self):
        params = RiskParams(N_sat=40, v_ref=6.2, w_rho=0.4, w_v=0.6)
        phi = compute_phi(active_count=20, avg_speed_mps=3.1, params=params)
        assert 0 <= phi <= 1

    def test_full_congestion(self):
        params = RiskParams(N_sat=40, v_ref=6.2, w_rho=0.4, w_v=0.6)
        phi = compute_phi(active_count=40, avg_speed_mps=0, params=params)
        assert phi == 1.0  # 满密度 + 零速度 = 最大拥堵

    def test_no_congestion(self):
        params = RiskParams(N_sat=40, v_ref=6.2, w_rho=0.4, w_v=0.6)
        phi = compute_phi(active_count=0, avg_speed_mps=6.2, params=params)
        assert phi == 0.0  # 零密度 + 自由流速度 = 无拥堵

    def test_clamping(self):
        params = RiskParams(N_sat=10, v_ref=5.0, w_rho=0.5, w_v=0.5)
        phi = compute_phi(active_count=100, avg_speed_mps=-1, params=params)
        assert 0 <= phi <= 1  # 应该被 clamp


class TestRiskParams:
    def test_weight_normalization(self):
        params = RiskParams(w_rho=2.0, w_v=3.0)
        assert abs(params.w_rho + params.w_v - 1.0) < 1e-6

    def test_from_json(self, tmp_path):
        import json
        config = {'N_sat': 30, 'v_ref': 5.0, 'w_rho': 0.5, 'w_v': 0.5}
        path = tmp_path / 'risk.json'
        path.write_text(json.dumps(config))
        params = RiskParams.from_json(path)
        assert params.N_sat == 30
        assert params.v_ref == 5.0


class TestPhiEventTracker:
    def test_no_event_below_threshold(self):
        tracker = PhiEventTracker(threshold=0.75, warmup_frames=0)
        result = tracker.update(0.5, timestamp=1.0, frame_index=1)
        assert result is None
        assert not tracker.in_event

    def test_event_start(self):
        tracker = PhiEventTracker(threshold=0.75, warmup_frames=0)
        result = tracker.update(0.8, timestamp=1.0, frame_index=1)
        assert result is not None
        assert result['type'] == 'start'
        assert tracker.in_event

    def test_event_end(self):
        tracker = PhiEventTracker(threshold=0.75, warmup_frames=0)
        tracker.update(0.8, timestamp=1.0, frame_index=1)
        result = tracker.update(0.5, timestamp=2.0, frame_index=2)
        assert result is not None
        assert result['type'] == 'end'
        assert not tracker.in_event
        assert len(tracker.events) == 1

    def test_warmup_period(self):
        tracker = PhiEventTracker(threshold=0.75, warmup_frames=5)
        result = tracker.update(0.9, timestamp=1.0, frame_index=1)
        assert result is None  # 预热期不触发事件

    def test_peak_tracking(self):
        tracker = PhiEventTracker(threshold=0.75, warmup_frames=0)
        tracker.update(0.8, timestamp=1.0, frame_index=1)
        tracker.update(0.9, timestamp=2.0, frame_index=2)
        tracker.update(0.85, timestamp=3.0, frame_index=3)
        assert tracker.peak_phi == 0.9
        assert tracker.peak_frame == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
