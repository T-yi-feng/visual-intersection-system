"""
运动状态分析模块 (Motion State Analysis)

负责：
- 车辆运动状态分类（移动/静止/停放）
- 速度计算（世界坐标系）
- 航向跟踪与锁定
- 运动统计汇总
"""

import numpy as np
from math import hypot, atan2, pi
from typing import Optional

from core.bev_transform import pixel_to_world


# ============================================================
# 运动状态
# ============================================================

class MotionState:
    """运动状态常量（状态机用字典存储，此类仅提供字符串常量）"""
    MOVING = 'moving'
    STATIONARY = 'stationary'
    PARKED_INITIAL = 'parked_initial'       # 从未移动
    PARKED_AFTER_STOP = 'parked_after_stop' # 移动后静止过久
    PARKED_PERIPHERAL = 'parked_peripheral' # 边缘区域长期静止


def default_track_motion_state() -> dict:
    """返回默认的跟踪运动状态字典（兼容旧接口）"""
    return {
        'ever_moved': False,
        'heading_locked': False,
        'last_heading': (0.0, 0.0),
        'post_move_stationary_s': 0.0,
        'initial_stationary_s': 0.0,
        'peripheral_stationary_s': 0.0,
        'parked_after_stop': False,
        'parked_initial': False,
        'parked_peripheral': False,
        'speed_mps': 0.0,
        'state': 'stationary',
    }


# ============================================================
# 运动统计汇总
# ============================================================

def summarize_motion_stats(
    trajectories: dict,
    current_meta: dict,
    h_mat: Optional[np.ndarray],
    track_motion_state: dict,
    motion_threshold_px: float = 4.0,
    heading_unlock_speed_mps: float = 0.8,
    post_stop_park_seconds: float = 20.0,
    initial_stationary_exclude_seconds: float = 30.0,
    long_stationary_seconds: float = 30.0,
    edge_margin_ratio: float = 0.12,
    frame_shape: tuple = (1080, 1920),
    fallback_pixels_per_meter: float = 10.0,
    max_valid_speed_mps: float = 45.0,
    sample_dt_s: float = 0.5,
) -> dict:
    """
    计算当前帧的运动统计。

    Returns
    -------
    stats : dict
        - moving_count: 移动车辆数
        - stationary_count: 静止车辆数
        - parked_count: 停放车辆数
        - active_count: 活跃车辆数（非停放）
        - avg_speed_mps: 活跃车辆平均速度
        - active_count_for_rho: 用于密度计算的车辆数
    """
    moving_count = 0
    stationary_count = 0
    parked_count = 0
    speed_sum = 0.0
    speed_count = 0

    fh, fw = frame_shape[:2]
    margin_x = fw * edge_margin_ratio
    margin_y = fh * edge_margin_ratio

    for tid, pts in trajectories.items():
        if len(pts) < 2:
            stationary_count += 1
            continue

        # 预热期：轨迹时间跨度不足 sample_dt_s，跳过速度计算
        track_age = pts[-1][0] - pts[0][0]
        if track_age < sample_dt_s:
            state_tmp = track_motion_state.get(tid)
            if state_tmp is None:
                state_tmp = default_track_motion_state()
                track_motion_state[tid] = state_tmp
            state_tmp['speed_mps'] = 0.0
            state_tmp['state'] = MotionState.STATIONARY
            stationary_count += 1
            continue

        meta = current_meta.get(tid, {})
        label = meta.get('label', 'car')

        # 获取运动状态
        if tid not in track_motion_state:
            track_motion_state[tid] = default_track_motion_state()
        state = track_motion_state[tid]

        # 速度计算：固定时间窗口，FPS 无关
        # 回溯 sample_dt_s 秒，用两点位移算速度
        t_new, x_new, y_new = pts[-1]
        t_old, x_old, y_old = t_new, x_new, y_new  # 默认：静止
        for i in range(len(pts) - 2, -1, -1):
            if pts[i][0] <= t_new - sample_dt_s:
                t_old, x_old, y_old = pts[i]
                break
        dt = t_new - t_old
        if dt <= 0:
            dt = sample_dt_s

        norm_px = hypot(x_new - x_old, y_new - y_old)

        # 速度计算：优先用世界坐标，异常时回退到像素空间
        dist_m_world = 0.0
        bev_valid = False
        if h_mat is not None:
            w_old = pixel_to_world(h_mat, (x_old, y_old))
            w_new = pixel_to_world(h_mat, (x_new, y_new))
            # 边缘过滤：世界坐标超出合理范围说明标定外推失效
            WORLD_MAX = 100.0  # 米
            if (abs(w_old[0]) < WORLD_MAX and abs(w_old[1]) < WORLD_MAX and
                abs(w_new[0]) < WORLD_MAX and abs(w_new[1]) < WORLD_MAX):
                dist_m_world = hypot(w_new[0] - w_old[0], w_new[1] - w_old[1])
                bev_valid = True
        # 当世界坐标位移异常（<1cm 但像素位移>5px）时用像素速度
        if bev_valid and dist_m_world < 0.01 and norm_px > 5:
            dist_m = norm_px / fallback_pixels_per_meter
        elif bev_valid:
            dist_m = dist_m_world
        else:
            dist_m = norm_px / fallback_pixels_per_meter

        speed_mps = dist_m / dt
        if speed_mps > max_valid_speed_mps:
            speed_mps = 0.0

        state['speed_mps'] = speed_mps

        # 运动判断
        moving_recent = norm_px >= motion_threshold_px

        # 航向更新
        if speed_mps > heading_unlock_speed_mps:
            state['ever_moved'] = True
            state['heading_locked'] = False
            norm = max(norm_px, 1e-6)
            state['last_heading'] = ((x_new - x_old) / norm, (y_new - y_old) / norm)
            state['post_move_stationary_s'] = 0.0
        elif state['ever_moved']:
            state['heading_locked'] = True
            state['post_move_stationary_s'] += sample_dt_s
            if state['post_move_stationary_s'] > post_stop_park_seconds:
                state['parked_after_stop'] = True

        # 初始静止判断
        if not state['ever_moved']:
            state['initial_stationary_s'] += sample_dt_s
            if state['initial_stationary_s'] > initial_stationary_exclude_seconds:
                state['parked_initial'] = True

        # 边缘停放判断
        cx, cy = meta.get('center', (x_new, y_new))
        is_edge = (cx < margin_x or cx > fw - margin_x or
                   cy < margin_y or cy > fh - margin_y)
        if is_edge and not moving_recent:
            state['peripheral_stationary_s'] += sample_dt_s
            if state['peripheral_stationary_s'] > long_stationary_seconds:
                state['parked_peripheral'] = True

        # 状态分类
        is_parked = (state['parked_initial'] or
                     state['parked_after_stop'] or
                     state['parked_peripheral'])

        if is_parked:
            state['state'] = MotionState.PARKED_INITIAL
            parked_count += 1
        elif moving_recent or speed_mps > heading_unlock_speed_mps:
            state['state'] = MotionState.MOVING
            moving_count += 1
            speed_sum += speed_mps
            speed_count += 1
        else:
            state['state'] = MotionState.STATIONARY
            stationary_count += 1

    active_count = moving_count + stationary_count
    avg_speed = speed_sum / max(speed_count, 1)

    return {
        'moving_count': moving_count,
        'stationary_count': stationary_count,
        'parked_count': parked_count,
        'active_count': active_count,
        'active_count_for_rho': active_count,
        'avg_speed_mps': avg_speed,
        'total_count': moving_count + stationary_count + parked_count,
    }
