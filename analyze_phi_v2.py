"""
分析前 60 帧的 Phi — 使用真实轨迹计算速度
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path
from math import hypot

sys.path.insert(0, str(Path(__file__).parent))

from core.phi import compute_phi, RiskParams
from core.detector import VehicleDetector
from core.bev_transform import load_homography, pixel_to_world
from core.motion import summarize_motion_stats

# 配置
VIDEO_PATH = "data/videos/default/vedio_000.mp4"
HOMOGRAPHY_PATH = "configs/homography_points_example.json"
RISK_PARAMS_PATH = "configs/traffic_risk_params.json"
MODEL_PATH = "data/models/yolo11m.pt"
MAX_FRAMES = 60
SAMPLE_DT = 0.5  # 速度计算时间窗口（秒）
WARMUP_FRAMES = 30  # 预热期帧数，不输出 Phi

# 加载
params = RiskParams.from_json(RISK_PARAMS_PATH)
detector = VehicleDetector(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
H, _, _ = load_homography(HOMOGRAPHY_PATH)

print("=" * 70)
print(f"  Phi 分析 — 真实轨迹速度 (sample_dt={SAMPLE_DT}s)")
print("=" * 70)
print(f"  v_ref = {params.v_ref} m/s ({params.v_ref*3.6:.1f} km/h)")
print(f"  N_sat = {params.N_sat}")
print(f"  w_rho = {params.w_rho}, w_v = {params.w_v}")
print(f"  视频 FPS = {fps}, 窗口 = {SAMPLE_DT}s ({int(SAMPLE_DT*fps)} 帧)")

# 轨迹存储: {track_id: [(t, x, y), ...]}
trajectories = {}
# 车辆元信息: {track_id: {label, center, conf, bbox}}
current_meta = {}
# 运动状态
track_motion_state = {}

results = []
frame_idx = 0

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    detections = detector.detect_frame(frame, timestamp=timestamp).vehicles

    # 更新轨迹和元信息
    seen_ids = set()
    for det in detections:
        tid = det.get('track_id', det.get('trackId', None))
        if tid is None:
            continue
        seen_ids.add(tid)

        if 'bbox' in det:
            x1, y1, x2, y2 = det['bbox']
            px, py = (x1 + x2) / 2, (y1 + y2) / 2
        elif 'center' in det:
            px, py = det['center']
        else:
            continue

        if tid not in trajectories:
            trajectories[tid] = []
        trajectories[tid].append((timestamp, px, py))
        # 只保留最近 2 秒的轨迹
        trajectories[tid] = [(t, x, y) for t, x, y in trajectories[tid]
                             if t >= timestamp - 2.0]

        current_meta[tid] = {
            'label': det.get('label', det.get('class', 'car')),
            'center': (px, py),
            'conf': det.get('conf', 0),
            'bbox': det.get('bbox', (0, 0, 0, 0)),
        }

    # 清理消失的 track
    for tid in list(trajectories.keys()):
        if tid not in seen_ids:
            if trajectories[tid][-1][0] < timestamp - 2.0:
                del trajectories[tid]
                track_motion_state.pop(tid, None)
                current_meta.pop(tid, None)

    # 计算运动统计
    h, w = frame.shape[:2]
    stats = summarize_motion_stats(
        trajectories=trajectories,
        current_meta=current_meta,
        h_mat=H,
        track_motion_state=track_motion_state,
        fps=fps,
        sample_dt_s=SAMPLE_DT,
        frame_shape=(h, w),
    )

    rho = min(1.0, stats['active_count'] / params.N_sat)
    eta = max(0.0, 1.0 - stats['avg_speed_mps'] / params.v_ref)
    phi = params.w_rho * rho + params.w_v * eta

    results.append({
        'frame': frame_idx,
        'count': stats['active_count'],
        'moving': stats['moving_count'],
        'stationary': stats['stationary_count'],
        'parked': stats['parked_count'],
        'avg_speed': stats['avg_speed_mps'],
        'rho': rho,
        'eta': eta,
        'phi': phi,
    })

    frame_idx += 1

cap.release()

# 输出（跳过预热期）
print(f"\n{'帧':>4} | {'活跃':>4} | {'移动':>4} | {'静止':>4} | {'停放':>4} | {'速度m/s':>8} | {'ρ':>6} | {'η':>6} | {'Phi':>6}")
print(f"{'-'*4} | {'-'*4} | {'-'*4} | {'-'*4} | {'-'*4} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6}")

for r in results:
    if r['frame'] < WARMUP_FRAMES:
        print(f"{r['frame']:>4} | {'---':>4} | {'---':>4} | {'---':>4} | {'---':>4} | {'预热期':>8} | {'---':>6} | {'---':>6} | {'---':>6}")
    else:
        print(f"{r['frame']:>4} | {r['count']:>4} | {r['moving']:>4} | {r['stationary']:>4} | {r['parked']:>4} | {r['avg_speed']:>8.2f} | {r['rho']:>6.3f} | {r['eta']:>6.3f} | {r['phi']:>6.3f}")

# 统计（只统计预热期之后）
valid_results = [r for r in results if r['frame'] >= WARMUP_FRAMES]
phi_vals = [r['phi'] for r in valid_results]
speed_vals = [r['avg_speed'] for r in valid_results if r['avg_speed'] > 0]
rho_vals = [r['rho'] for r in valid_results]

print(f"\n{'='*70}")
print(f"  统计")
print(f"{'='*70}")
print(f"  Phi:     mean={np.mean(phi_vals):.3f}  max={max(phi_vals):.3f}  min={min(phi_vals):.3f}")
print(f"  速度:    mean={np.mean(speed_vals):.2f} m/s ({np.mean(speed_vals)*3.6:.1f} km/h)")
print(f"  ρ:       mean={np.mean(rho_vals):.3f}")
print(f"  活跃车辆: mean={np.mean([r['count'] for r in results]):.1f}")

print(f"\n  对比 v_ref = {params.v_ref} m/s ({params.v_ref*3.6:.1f} km/h)")
print(f"  η = 1 - 实际速度/v_ref")
print(f"  如果实际速度 ≈ {np.mean(speed_vals):.1f} m/s，η ≈ {1 - np.mean(speed_vals)/params.v_ref:.3f}")
