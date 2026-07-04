"""
拥堵指数计算模块 (Phi - Congestion Risk Index)

负责：
- 标量 Phi 计算（兼容原方法）
- 风险参数加载与热重载
- Phi 事件跟踪（阈值穿越）
"""

import json
from pathlib import Path
from typing import Optional
from collections import deque


# ============================================================
# 风险参数
# ============================================================

class RiskParams:
    """交通风险参数"""

    def __init__(
        self,
        N_sat: float = 40.0,
        v_ref: float = 6.2,
        v_free: float = 0.0,
        w_rho: float = 0.4,
        w_v: float = 0.6,
        refresh_seconds: float = 0.5,
        phi_plot_threshold: float = 0.75,
        chart_window_seconds: float = 120.0,
        edge_margin_ratio: float = 0.12,
        long_stationary_seconds: float = 30.0,
        paper_export_width: int = 2200,
        paper_export_height: int = 1400,
    ):
        self.N_sat = N_sat
        self.v_ref = v_ref
        self.v_free = v_free
        self.w_rho = w_rho
        self.w_v = w_v
        self.refresh_seconds = refresh_seconds
        self.phi_plot_threshold = phi_plot_threshold
        self.chart_window_seconds = chart_window_seconds
        self.edge_margin_ratio = edge_margin_ratio
        self.long_stationary_seconds = long_stationary_seconds
        self.paper_export_width = paper_export_width
        self.paper_export_height = paper_export_height

        # 存储原始值（用于 to_dict 序列化），归一化值用于计算
        self.w_rho_raw = w_rho
        self.w_v_raw = w_v
        total = w_rho + w_v
        if total > 0:
            self.w_rho = w_rho / total
            self.w_v = w_v / total

    @classmethod
    def from_json(cls, path: str | Path) -> 'RiskParams':
        """从 JSON 文件加载"""
        import inspect
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

    def to_dict(self) -> dict:
        return {
            'N_sat': self.N_sat,
            'v_ref': self.v_ref,
            'v_free': self.v_free,
            'w_rho': self.w_rho_raw,
            'w_v': self.w_v_raw,
            'refresh_seconds': self.refresh_seconds,
            'phi_plot_threshold': self.phi_plot_threshold,
            'chart_window_seconds': self.chart_window_seconds,
            'edge_margin_ratio': self.edge_margin_ratio,
            'long_stationary_seconds': self.long_stationary_seconds,
            'paper_export_width': self.paper_export_width,
            'paper_export_height': self.paper_export_height,
        }


# ============================================================
# Phi 计算
# ============================================================

def compute_phi(
    active_count: int,
    avg_speed_mps: float,
    params: RiskParams,
) -> float:
    """
    计算综合拥堵指数 Phi（标量版本，兼容原方法）。

    Phi_t = w_rho * rho_t + w_v * eta_t

    其中：
    - rho_t = min(1.0, active_count / N_sat)
    - eta_t = max(0.0, 1.0 - avg_speed / v_ref)

    Returns
    -------
    phi : float, [0, 1]
    """
    rho_term = min(1.0, active_count / max(params.N_sat, 1))
    vel_term = max(0.0, 1.0 - avg_speed_mps / max(params.v_ref, 0.1))
    phi = params.w_rho * rho_term + params.w_v * vel_term
    return max(0.0, min(1.0, phi))


# ============================================================
# Phi 事件跟踪
# ============================================================

class PhiEventTracker:
    """
    Phi 阈值穿越事件跟踪器。

    当 Phi 超过阈值时开始事件，低于阈值时结束事件。
    事件期间记录峰值帧信息。
    """

    def __init__(self, threshold: float = 0.75, warmup_frames: int = 30,
                 max_history: int = 2000):
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self.frame_count = 0

        # 当前事件状态
        self.in_event = False
        self.event_start_time = 0.0
        self.event_start_frame = 0
        self.peak_phi = 0.0
        self.peak_frame = 0
        self.peak_candidates = []

        # 历史（deque 限制大小，防止长时间运行内存泄漏）
        self.phi_history = deque(maxlen=max_history)  # (time_s, phi)
        self.events = []       # 已完成的事件

    def update(self, phi: float, timestamp: float, frame_index: int) -> Optional[dict]:
        """
        更新 Phi 值，返回事件状态变化。

        Returns
        -------
        event_info : dict or None
            - 'type': 'start' | 'peak' | 'end'
            - 其他事件信息
        """
        self.frame_count += 1
        self.phi_history.append((timestamp, phi))

        # 预热期跳过
        if self.frame_count < self.warmup_frames:
            return None

        result = None

        if phi > self.threshold:
            if not self.in_event:
                # 事件开始
                self.in_event = True
                self.event_start_time = timestamp
                self.event_start_frame = frame_index
                self.peak_phi = phi
                self.peak_frame = frame_index
                self.peak_candidates = []
                result = {'type': 'start', 'time': timestamp, 'phi': phi}

            # 记录峰值
            self.peak_candidates.append({
                'frame_index': frame_index,
                'time_s': timestamp,
                'phi': phi,
            })

            if phi > self.peak_phi:
                self.peak_phi = phi
                self.peak_frame = frame_index
                result = {'type': 'peak', 'time': timestamp, 'phi': phi}

        elif self.in_event:
            # 事件结束
            event = {
                'type': 'end',
                'start_time': self.event_start_time,
                'end_time': timestamp,
                'start_frame': self.event_start_frame,
                'end_frame': frame_index,
                'peak_phi': self.peak_phi,
                'peak_frame': self.peak_frame,
                'peak_candidates': self.peak_candidates.copy(),
                'duration_s': timestamp - self.event_start_time,
            }
            self.events.append(event)
            self.in_event = False
            result = event

        return result

    def close_current_event(self, timestamp: float, frame_index: int) -> Optional[dict]:
        """强制关闭当前事件（视频结束时调用）"""
        if not self.in_event:
            return None

        event = {
            'type': 'end',
            'start_time': self.event_start_time,
            'end_time': timestamp,
            'start_frame': self.event_start_frame,
            'end_frame': frame_index,
            'peak_phi': self.peak_phi,
            'peak_frame': self.peak_frame,
            'peak_candidates': self.peak_candidates.copy(),
            'duration_s': timestamp - self.event_start_time,
        }
        self.events.append(event)
        self.in_event = False
        return event

    @property
    def recent_phi(self) -> list[tuple[float, float]]:
        """最近的 Phi 历史"""
        return self.phi_history

    def clear_history_before(self, timestamp: float):
        """清理旧的 Phi 历史（deque 已自动淘汰旧数据，此方法保留兼容性）"""
        while self.phi_history and self.phi_history[0][0] < timestamp:
            self.phi_history.popleft()


# ============================================================
# 风险参数热重载
# ============================================================

class RiskParamsReloader:
    """风险参数热重载器"""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.params = RiskParams.from_json(config_path)
        self._last_mtime = self.config_path.stat().st_mtime

    def check_and_reload(self) -> bool:
        """检查文件是否更新，如果是则重新加载。返回是否重载。"""
        try:
            mtime = self.config_path.stat().st_mtime
            if mtime > self._last_mtime:
                self.params = RiskParams.from_json(self.config_path)
                self._last_mtime = mtime
                return True
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] 风险参数重载失败: {e}")
            pass
        return False
