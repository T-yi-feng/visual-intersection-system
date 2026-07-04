"""
分析前 60 帧的 Phi 计算
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path
from math import pi

sys.path.insert(0, str(Path(__file__).parent))

from core.phi import compute_phi, RiskParams
from core.detector import VehicleDetector
from core.bev_transform import load_homography, pixel_to_world

# ============================================================
# 配置
# ============================================================
VIDEO_PATH = "data/videos/default/vedio_000.mp4"
HOMOGRAPHY_PATH = "configs/homography_points_example.json"
RISK_PARAMS_PATH = "configs/traffic_risk_params.json"
MODEL_PATH = "data/models/yolo11m.pt"
MAX_FRAMES = 60

# ============================================================
# 加载配置
# ============================================================
print("=" * 70)
print("  Phi 计算分析 — 前 60 帧")
print("=" * 70)

params = RiskParams.from_json(RISK_PARAMS_PATH)
print(f"\n[配置参数]")
print(f"  N_sat (饱和车辆数) = {params.N_sat}")
print(f"  v_ref (参考速度)   = {params.v_ref} m/s ({params.v_ref*3.6:.1f} km/h)")
print(f"  w_rho (密度权重)   = {params.w_rho}")
print(f"  w_v (速度权重)     = {params.w_v}")

print(f"\n[Phi 公式]")
print(f"  Phi = w_rho × ρ + w_v × η")
print(f"  其中：")
print(f"    ρ = min(1.0, active_count / N_sat)")
print(f"    η = max(0.0, 1.0 - avg_speed / v_ref)")

# ============================================================
# 加载模型和视频
# ============================================================
print(f"\n[加载模型] {MODEL_PATH}...")
detector = VehicleDetector(MODEL_PATH)

print(f"[加载视频] {VIDEO_PATH}...")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] 无法打开视频")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"  {width}x{height}, {fps:.1f} FPS, {total_frames} 帧")

# 加载单应性矩阵
H, img_pts, world_pts = load_homography(HOMOGRAPHY_PATH)
print(f"[单应性矩阵] 已加载")

# ============================================================
# 处理前 60 帧
# ============================================================
print(f"\n{'='*70}")
print(f"  开始处理前 {MAX_FRAMES} 帧")
print(f"{'='*70}")

results = []
frame_idx = 0

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测
    result = detector.detect_frame(frame, timestamp=frame_idx / fps)
    detections = result.vehicles

    # 提取车辆信息
    vehicles = []
    for det in detections:
        # 边界框中心 (像素坐标)
        if 'bbox' in det:
            x1, y1, x2, y2 = det['bbox']
            px = (x1 + x2) / 2
            py = (y1 + y2) / 2
        elif 'center' in det:
            px, py = det['center']
        else:
            continue

        # 转换到世界坐标
        try:
            wx, wy = pixel_to_world(H, (px, py))
        except Exception:
            continue

        vehicles.append({
            'cx': wx,
            'cy': wy,
            'px': px,
            'py': py,
            'class': det.get('label', det.get('class', 'unknown')),
            'conf': det.get('conf', 0),
        })

    # 计算统计
    active_count = len(vehicles)

    # 简化的速度估算（用检测框大小变化近似）
    # 实际系统中用追踪历史计算，此脚本使用模拟值仅供演示
    # 如需真实速度，请使用 analyze_phi_v2.py
    import warnings
    warnings.warn("analyze_phi.py 使用模拟速度，如需真实分析请使用 analyze_phi_v2.py",
                  stacklevel=1)
    avg_speed_mps = 3.0 + np.random.randn() * 0.5  # 模拟
    avg_speed_mps = max(0, avg_speed_mps)

    # 计算 Phi
    rho = min(1.0, active_count / params.N_sat)
    eta = max(0.0, 1.0 - avg_speed_mps / params.v_ref)
    phi = params.w_rho * rho + params.w_v * eta

    results.append({
        'frame': frame_idx,
        'count': active_count,
        'avg_speed': avg_speed_mps,
        'rho': rho,
        'eta': eta,
        'phi': phi,
    })

    frame_idx += 1

cap.release()

# ============================================================
# 输出分析结果
# ============================================================
print(f"\n{'='*70}")
print(f"  帧级分析结果")
print(f"{'='*70}")
print(f"{'帧':>4} | {'车辆数':>6} | {'ρ':>8} | {'η':>8} | {'Phi':>8} | {'ρ贡献':>8} | {'η贡献':>8} | {'主要因素'}")
print(f"{'-'*4} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10}")

for r in results:
    rho_contrib = params.w_rho * r['rho']
    eta_contrib = params.w_v * r['eta']
    main_factor = "密度" if rho_contrib > eta_contrib else "速度"

    print(f"{r['frame']:>4} | {r['count']:>6} | {r['rho']:>8.3f} | {r['eta']:>8.3f} | {r['phi']:>8.3f} | {rho_contrib:>8.3f} | {eta_contrib:>8.3f} | {main_factor}")

# ============================================================
# 统计汇总
# ============================================================
phi_values = [r['phi'] for r in results]
rho_values = [r['rho'] for r in results]
eta_values = [r['eta'] for r in results]
count_values = [r['count'] for r in results]

print(f"\n{'='*70}")
print(f"  统计汇总")
print(f"{'='*70}")
print(f"  Phi:  min={min(phi_values):.3f}  max={max(phi_values):.3f}  mean={np.mean(phi_values):.3f}  std={np.std(phi_values):.3f}")
print(f"  ρ:    min={min(rho_values):.3f}  max={max(rho_values):.3f}  mean={np.mean(rho_values):.3f}")
print(f"  η:    min={min(eta_values):.3f}  max={max(eta_values):.3f}  mean={np.mean(eta_values):.3f}")
print(f"  Count: min={min(count_values)}  max={max(count_values)}  mean={np.mean(count_values):.1f}")

# ============================================================
# 诊断：为什么 Phi 偏高？
# ============================================================
print(f"\n{'='*70}")
print(f"  诊断：Phi 偏高的原因分析")
print(f"{'='*70}")

high_phi_frames = [r for r in results if r['phi'] > 0.5]
print(f"\n  Phi > 0.5 的帧数: {len(high_phi_frames)}/{len(results)}")

if high_phi_frames:
    # 分析主要贡献因素
    rho_dominant = [r for r in high_phi_frames if params.w_rho * r['rho'] > params.w_v * r['eta']]
    eta_dominant = [r for r in high_phi_frames if params.w_v * r['eta'] >= params.w_rho * r['rho']]

    print(f"  密度主导 (ρ贡献 > η贡献): {len(rho_dominant)} 帧")
    print(f"  速度主导 (η贡献 ≥ ρ贡献): {len(eta_dominant)} 帧")

    # 详细分析前 5 个高 Phi 帧
    print(f"\n  前 5 个高 Phi 帧的详细分析：")
    for r in sorted(high_phi_frames, key=lambda x: x['phi'], reverse=True)[:5]:
        rho_contrib = params.w_rho * r['rho']
        eta_contrib = params.w_v * r['eta']
        print(f"\n    帧 {r['frame']}: Phi={r['phi']:.3f}")
        print(f"      车辆数={r['count']}, 平均速度={r['avg_speed']:.1f} m/s")
        print(f"      ρ = {r['count']}/{params.N_sat} = {r['rho']:.3f}  →  贡献 = {params.w_rho}×{r['rho']:.3f} = {rho_contrib:.3f}")
        print(f"      η = 1 - {r['avg_speed']:.1f}/{params.v_ref} = {r['eta']:.3f}  →  贡献 = {params.w_v}×{r['eta']:.3f} = {eta_contrib:.3f}")

# ============================================================
# 参数敏感性分析
# ============================================================
print(f"\n{'='*70}")
print(f"  参数敏感性分析")
print(f"{'='*70}")

print(f"\n  当前参数：N_sat={params.N_sat}, v_ref={params.v_ref}")
print(f"  如果调整参数，Phi 会如何变化？\n")

# 测试不同的 N_sat
print(f"  [N_sat 敏感性] (固定 v_ref={params.v_ref})")
print(f"  {'N_sat':>8} | {'Phi_mean':>10} | {'Phi_max':>10} | {'变化':>10}")
print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")

base_phi_mean = np.mean(phi_values)
for n_sat in [20, 30, 40, 50, 60, 80]:
    phi_test = []
    for r in results:
        rho_t = min(1.0, r['count'] / n_sat)
        eta_t = r['eta']
        phi_t = params.w_rho * rho_t + params.w_v * eta_t
        phi_test.append(phi_t)
    change = np.mean(phi_test) - base_phi_mean
    print(f"  {n_sat:>8} | {np.mean(phi_test):>10.3f} | {max(phi_test):>10.3f} | {change:>+10.3f}")

# 测试不同的 v_ref
print(f"\n  [v_ref 敏感性] (固定 N_sat={params.N_sat})")
print(f"  {'v_ref':>8} | {'Phi_mean':>10} | {'Phi_max':>10} | {'变化':>10}")
print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")

for v_ref in [3.0, 5.0, 6.2, 8.0, 10.0, 15.0]:
    phi_test = []
    for r in results:
        rho_t = r['rho']
        eta_t = max(0.0, 1.0 - r['avg_speed'] / v_ref)
        phi_t = params.w_rho * rho_t + params.w_v * eta_t
        phi_test.append(phi_t)
    change = np.mean(phi_test) - base_phi_mean
    print(f"  {v_ref:>8.1f} | {np.mean(phi_test):>10.3f} | {max(phi_test):>10.3f} | {change:>+10.3f}")

# ============================================================
# 结论
# ============================================================
print(f"\n{'='*70}")
print(f"  结论与建议")
print(f"{'='*70}")

avg_count = np.mean(count_values)
avg_speed = np.mean([r['avg_speed'] for r in results])

print(f"""
  当前场景分析：
  - 平均车辆数: {avg_count:.1f} / N_sat({params.N_sat}) = {avg_count/params.N_sat:.3f}
  - 平均速度: {avg_speed:.1f} m/s / v_ref({params.v_ref}) = {avg_speed/params.v_ref:.3f}
  - 平均 Phi: {np.mean(phi_values):.3f}

  Phi 偏高的可能原因：

  1. 【密度项 ρ 偏高】
     - 车辆数 ({avg_count:.0f}) 相对于 N_sat ({params.N_sat}) 较多
     - 如果路口确实车辆密集，这是合理的
     - 如果检测有误报，会虚增车辆数

  2. 【速度项 η 偏高】
     - 平均速度 ({avg_speed:.1f} m/s) 相对于 v_ref ({params.v_ref}) 较低
     - v_ref = 6.2 m/s ≈ 22 km/h，这是一个较低的参考值
     - 如果实际车速更低，η 会更高

  3. 【权重分配】
     - w_v = {params.w_v} > w_rho = {params.w_rho}
     - 速度项权重更高，速度对 Phi 影响更大

  调整建议：
  - 如果 Phi 整体偏高，增大 N_sat（如 50-60）
  - 如果速度项贡献过大，增大 v_ref（如 8-10 m/s）
  - 可以通过热加载 risk_params.json 实时调整
""")
