"""
主循环引擎 (Pipeline Engine)

完整流程：
1. 读取斜视监控帧
2. YOLO 检测 + ByteTrack 跟踪（像素空间）
3. cv2.warpPerspective 生成 BEV 鸟瞰图像
4. 坐标变换到世界空间 + 世界坐标航向
5. 运动状态分析
6. 卷积冲突分析 + 归因
7. BEV 叠加可视化（车辆框按归因分数着色）
8. 实时输出：BEV + Phi 曲线 + 指标
"""

import cv2
import gc
import json
import csv
import time as time_mod
import numpy as np
from pathlib import Path
from math import atan2, hypot, pi
from collections import deque

from core.bev_transform import load_homography, pixel_to_world
from core.detector import VehicleDetector, cleanup_track_state
from core.motion import summarize_motion_stats
from core.phi import compute_phi, PhiEventTracker, RiskParamsReloader
from core.conflict import ConflictAnalyzer
from analysis.attribution import CongestionAttributor
from analysis.ablation import AblationStudy
from visualization.phi_chart import render_phi_chart_panel
from visualization.conflict_debug import render_conflict_debug_window
from utils.drawing import fit_for_display, draw_text_with_bg, letterbox_to
from utils.config_loader import load_vehicle_size_map
from utils.async_writer import AsyncLivePreviewWriter

# 默认 BEV 四边形坐标
_DEFAULT_QUAD = [
    [0.38, 0.24],   # 左上
    [0.72, 0.30],   # 右上
    [0.64, 0.80],   # 右下
    [0.16, 0.62],   # 左下
]


# ============================================================
# 引擎配置
# ============================================================

class EngineConfig:
    """引擎配置"""
    def __init__(self, args):
        self.model_path = args.model
        self.source = args.source
        self.homography_path = args.homography
        self.risk_params_path = args.risk_params
        self.imgsz = args.imgsz
        self.conf = args.conf
        self.iou = args.iou
        self.tracker = args.tracker

        # 输出路径
        self.save_path = getattr(args, 'save_path', None)
        self.save_third = getattr(args, 'save_third', None)
        self.live_dir = getattr(args, 'live_dir', None)
        self.events_dir = getattr(args, 'events_dir', None)
        self.charts_dir = getattr(args, 'charts_dir', None)

        # 显示
        self.show_windows = getattr(args, 'show_windows', True)
        self.display_scale = getattr(args, 'display_scale', 1.0)

        # 分析
        self.ablation_enable = getattr(args, 'ablation_enable', True)
        self.ablation_levels = getattr(args, 'ablation_levels', 3)
        self.realtime_congestion_interval = getattr(args, 'realtime_congestion_interval', 2)
        self.live_write_interval = getattr(args, 'live_write_interval', 2)
        self.async_writer = getattr(args, 'async_writer', True)
        self.async_writer_queue = getattr(args, 'async_writer_queue', 4)
        self.max_frames = getattr(args, 'max_frames', 0)

        # 卷积参数
        self.grid_size = getattr(args, 'grid_size', 64)
        self.world_width_m = getattr(args, 'world_width_m', 40.0)
        self.world_height_m = getattr(args, 'world_height_m', 40.0)

        # BEV 输出尺寸
        self.bev_width = getattr(args, 'bev_width', 1600)
        self.bev_height = getattr(args, 'bev_height', 1100)

        # 轨迹
        self.trail_seconds = getattr(args, 'trail_seconds', 10.0)
        self.frame_stride = getattr(args, 'frame_stride', 3)
        self.target_fps = getattr(args, 'target_fps', 20.0)

        # 车辆配置路径
        self.vehicle_size_path = getattr(args, 'vehicle_size_path',
                                         'data/vehicle/vehicle_size_m.json')

        # BEV 四边形坐标
        self.bev_quad = getattr(args, 'bev_quad', None)

        # Backtest 调试模式
        self.backtest = getattr(args, 'backtest', False)


# ============================================================
# 主引擎
# ============================================================

class PipelineEngine:
    """
    流水线引擎：斜视帧 → BEV → 卷积分析 → 可视化输出

    Usage
    -----
    >>> engine = PipelineEngine(config)
    >>> engine.run()
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self._running = False

    def run(self):
        """运行主循环"""
        cfg = self.cfg

        # ── 加载配置 ──────────────────────────────────────────
        vehicle_size_m = load_vehicle_size_map(cfg.vehicle_size_path)
        risk_reloader = RiskParamsReloader(cfg.risk_params_path)
        risk_params = risk_reloader.params

        # 单应性矩阵
        h_mat, img_pts, world_pts = load_homography(cfg.homography_path)

        # 从标定文件自动推算世界坐标范围（不再硬编码）
        wp = np.array(world_pts, dtype=np.float64)
        # 网格覆盖标定区域 + 余量，原点取标定最小值
        world_origin_x = float(wp[:, 0].min())
        world_origin_y = float(wp[:, 1].min())
        auto_world_w = float(wp[:, 0].max() - wp[:, 0].min())
        auto_world_h = float(wp[:, 1].max() - wp[:, 1].min())
        # 保留余量，确保边缘车辆也在网格内
        world_w = max(auto_world_w * 1.2, cfg.world_width_m)
        world_h = max(auto_world_h * 1.2, cfg.world_height_m)
        print(f"[INFO] 世界坐标范围: {world_w:.1f}m x {world_h:.1f}m, "
              f"原点: ({world_origin_x:.1f}, {world_origin_y:.1f})")

        # 检测器
        detector = VehicleDetector(
            model_path=cfg.model_path, tracker_config=cfg.tracker,
            memory_enabled=True,
        )

        # 冲突分析器（延迟初始化：等第一帧拿到实际车辆坐标后再构建）
        conflict_analyzer = None
        attributor = None
        ablation_study = None
        _analyzer_initialized = False

        # 事件跟踪
        event_tracker = PhiEventTracker(threshold=risk_params.phi_plot_threshold)

        # 异步写入器
        async_writer = None
        if cfg.async_writer and cfg.live_dir:
            async_writer = AsyncLivePreviewWriter(cfg.async_writer_queue)
            async_writer.start()

        # ── 打开视频源 ────────────────────────────────────────
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频源: {cfg.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.target_fps if hasattr(cfg, 'target_fps') else 20.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── 构建 BEV 变换矩阵 ────────────────────────────────
        # 用四边形区域做透视变换，只映射路面区域，避免非道路像素干扰
        bev_w, bev_h = cfg.bev_width, cfg.bev_height

        # 归一化四边形坐标
        if cfg.bev_quad:
            parts = cfg.bev_quad.replace(';', ',').split(',')
            coords = [float(p) for p in parts if p.strip()]
            if len(coords) == 8:
                quad_norm = np.array(coords, dtype=np.float32).reshape(4, 2)
            else:
                print(f"[WARN] bev_quad 格式错误，使用默认值")
                quad_norm = np.array(_DEFAULT_QUAD, dtype=np.float32)
        else:
            quad_norm = np.array(_DEFAULT_QUAD, dtype=np.float32)
        src_pts = quad_norm * np.array([frame_w, frame_h], dtype=np.float32)
        dst_pts = np.array([
            [0, 0], [bev_w - 1, 0],
            [bev_w - 1, bev_h - 1], [0, bev_h - 1]
        ], dtype=np.float32)
        H_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 像素/米 标定
        world_w = world_pts[:, 0].max() - world_pts[:, 0].min()
        world_h = world_pts[:, 1].max() - world_pts[:, 1].min()
        ppm_x = bev_w / max(world_w, 1e-6)
        ppm_y = bev_h / max(world_h, 1e-6)
        ppm = 0.5 * (ppm_x + ppm_y)  # pixels per meter

        # ── 状态变量 ──────────────────────────────────────────
        # 轨迹用 deque 自动淘汰旧数据，无需手动剪枝
        stride = max(1, cfg.frame_stride)
        trail_maxlen = max(int(cfg.trail_seconds * fps / stride), 30)
        trajectories: dict[int, deque] = {}  # tid -> deque[(t, x, y), ...]
        track_motion_state = {}
        smoothed_centers = {}   # tid -> (cx_ema, cy_ema) 仅用于显示平滑
        smoothed_headings = {}  # tid -> heading_deg_ema
        prev_raw_positions = {} # tid -> (bev_x, bev_y) 上一帧原始 BEV 位置（航向计算用）
        frame_index = 0
        phi = 0.0
        last_phi_time = 0
        last_congestion_frame = 0
        last_live_write_frame = 0
        conflict_result = None
        influences = None

        # ── 性能计时器 ──
        perf = {'detect': 0.0, 'traj': 0.0, 'motion': 0.0, 'phi': 0.0,
                'viz_ab': 0.0, 'viz_c': 0.0, 'heatmap': 0.0, 'write': 0.0, 'other': 0.0}
        _t_last = time_mod.perf_counter()

        target_fps = getattr(cfg, 'target_fps', 20.0)

        self._pending_event = None  # 待导出的事件（可视化后执行）
        # 冲突峰值追踪：记录交织点数最多的帧
        max_conflict_info = {
            'count': 0, 'phi': 0.0, 'timestamp': 0.0, 'frame_index': 0,
            'conflict_result': None, 'influences': None, 'vehicles': None,
            'combined': None,
        }
        self._running = True
        print("[INFO] 主循环开始")
        print(f"[INFO] BEV 输出: {bev_w}x{bev_h}, ppm={ppm:.1f}")
        print(f"[INFO] 目标 FPS: {target_fps}, 初始 stride: {stride}")

        while self._running:
            _tick = time_mod.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_index / fps

            # ── 固定跳帧：stride>1 时每隔 stride 帧处理一帧 ──
            is_key_frame = (stride <= 1) or ((frame_index % stride) == 0)

            if not is_key_frame:
                frame_index += 1
                continue
            _tick = time_mod.perf_counter()  # 仅关键帧计时
            _tick_frame_start = _tick

            # ── 步骤 1: 检测 + 跟踪（像素空间）─────────────────
            det_result = detector.detect_frame(
                frame, timestamp, cfg.imgsz, cfg.conf, cfg.iou,
                frame_index=frame_index,
            )
            perf['detect'] += time_mod.perf_counter() - _tick; _tick = time_mod.perf_counter()

            # ── 步骤 2: BEV 鸟瞰映射 ────────────────────────
            bev_frame = cv2.warpPerspective(frame, H_bev, (bev_w, bev_h))

            # ── 步骤 3: 更新轨迹（像素空间）──────────────────
            for v in det_result.vehicles:
                tid = v['track_id']
                if tid not in trajectories:
                    trajectories[tid] = deque(maxlen=trail_maxlen)
                trajectories[tid].append((timestamp, v['center'][0], v['center'][1]))

            # 清理过期轨迹：如果最近一个点的时间戳超过 trail_seconds，删除整个轨迹
            cutoff = timestamp - cfg.trail_seconds
            active_ids = {v['track_id'] for v in det_result.vehicles}
            for tid in list(trajectories.keys()):
                pts = trajectories[tid]
                if len(pts) == 0 or pts[-1][0] < cutoff:
                    del trajectories[tid]
                    track_motion_state.pop(tid, None)

            # ── 步骤 4: 构建 current_meta（全帧 BEV 坐标 + EMA 平滑航向）──
            # EMA 平滑因子：alpha 越小越平滑，alpha 越大响应越快
            # stride 自适应：跳帧时增大 alpha 以补偿帧间时间间隔增大
            # 公式：alpha_adj = 1 - (1 - alpha_base) ^ stride
            _alpha_base_pos = 0.4
            _alpha_base_heading = 0.15
            SMOOTH_ALPHA_POS = 1.0 - (1.0 - _alpha_base_pos) ** stride
            SMOOTH_ALPHA_HEADING = 1.0 - (1.0 - _alpha_base_heading) ** stride
            MIN_DISPLACEMENT = 2.0    # 最小位移阈值（BEV 像素）

            # ── 步骤 4a: 构建 current_meta（BEV坐标 + EMA平滑航向）──
            current_meta = {}
            all_pts = np.zeros((max(len(det_result.vehicles), 1) * 2, 2), dtype=np.float32)
            vi = 0
            for v in det_result.vehicles:
                tid = v['track_id']
                meta = dict(v)

                # 像素坐标 → 世界坐标（用于分析）
                cx_px, cy_px = v['center']
                w_coord = pixel_to_world(h_mat, (cx_px, cy_px))
                meta['world_x'] = float(w_coord[0])
                meta['world_y'] = float(w_coord[1])

                # 像素坐标 → BEV 像素坐标（原始值，用于航向计算）
                pt_bev_raw = cv2.perspectiveTransform(
                    np.array([[[cx_px, cy_px]]], dtype=np.float32), H_bev
                )[0][0]
                bev_raw_x, bev_raw_y = float(pt_bev_raw[0]), float(pt_bev_raw[1])

                # BEV 像素坐标（显示用，带 EMA 平滑，不参与航向计算）
                if tid in smoothed_centers:
                    sx = SMOOTH_ALPHA_POS * bev_raw_x + (1 - SMOOTH_ALPHA_POS) * smoothed_centers[tid][0]
                    sy = SMOOTH_ALPHA_POS * bev_raw_y + (1 - SMOOTH_ALPHA_POS) * smoothed_centers[tid][1]
                else:
                    sx, sy = bev_raw_x, bev_raw_y
                smoothed_centers[tid] = (sx, sy)
                meta['bev_x'] = float(sx)
                meta['bev_y'] = float(sy)

                # 航向：固定最近两帧（参考系不变）
                pts = trajectories.get(tid, [])
                if len(pts) >= 2:
                    _, px1, py1 = pts[-2]
                    _, px2, py2 = pts[-1]
                    all_pts[2 * vi] = (px1, py1)
                    all_pts[2 * vi + 1] = (px2, py2)
                    vi += 1

                meta['heading_deg'] = 0  # 默认值，批量投影后更新
                meta['bev_raw_x'] = bev_raw_x
                meta['bev_raw_y'] = bev_raw_y
                current_meta[tid] = meta

            # 批量 BEV 投影（一次性计算所有航向）
            if vi > 0:
                bev_pts = cv2.perspectiveTransform(
                    all_pts[:2 * vi].reshape(-1, 1, 2), H_bev
                ).reshape(-1, 2)
                idx = 0
                for v in det_result.vehicles:
                    tid = v['track_id']
                    meta = current_meta.get(tid)
                    if meta is None:
                        continue
                    pts_traj = trajectories.get(tid, [])
                    if len(pts_traj) < 2:
                        continue
                    p1, p2 = bev_pts[idx * 2], bev_pts[idx * 2 + 1]
                    idx += 1
                    dx_c = p2[0] - p1[0]
                    dy_c = p2[1] - p1[1]
                    if hypot(dx_c, dy_c) >= MIN_DISPLACEMENT:
                        raw_h = atan2(-dy_c, dx_c) * 180.0 / pi
                        if tid in smoothed_headings:
                            prev_h = smoothed_headings[tid]
                            diff = (raw_h - prev_h + 180) % 360 - 180
                            meta['heading_deg'] = prev_h + SMOOTH_ALPHA_HEADING * diff
                        else:
                            meta['heading_deg'] = raw_h
                    else:
                        meta['heading_deg'] = smoothed_headings.get(tid, 0)
                    smoothed_headings[tid] = meta['heading_deg']

            # 清理不存在的车辆
            active_ids = {v['track_id'] for v in det_result.vehicles}
            for tid in list(smoothed_centers.keys()):
                if tid not in active_ids:
                    del smoothed_centers[tid]
                    smoothed_headings.pop(tid, None)
            # 定期清理 detector 的标签历史，防止内存泄漏
            if frame_index % 100 == 0:
                cleanup_track_state(detector.track_state, active_ids)

            perf['traj'] += time_mod.perf_counter() - _tick; _tick = time_mod.perf_counter()

            # ── 步骤 5: 运动统计 ─────────────────────────────
            motion_stats = summarize_motion_stats(
                trajectories, current_meta, h_mat, track_motion_state,
                frame_shape=frame.shape,
            )
            perf['motion'] += time_mod.perf_counter() - _tick; _tick = time_mod.perf_counter()

            # ── 步骤 6: 计算 Phi ─────────────────────────────
            if timestamp - last_phi_time >= risk_params.refresh_seconds:
                phi = compute_phi(
                    motion_stats['active_count_for_rho'],
                    motion_stats['avg_speed_mps'],
                    risk_params,
                )
                last_phi_time = timestamp

                # 事件跟踪
                event_info = event_tracker.update(phi, timestamp, frame_index)

                # ── 步骤 7: 卷积冲突分析 + 归因（Phi 超阈值时，backtest 模式强制执行）──
                # 延迟初始化：用第一帧实际车辆坐标校正网格原点
                if not _analyzer_initialized and len(current_meta) > 0:
                    all_wx = [m['world_x'] for m in current_meta.values()]
                    all_wy = [m['world_y'] for m in current_meta.values()]
                    # 网格覆盖实际车辆范围 + 20%余量
                    wx_min, wx_max = min(all_wx), max(all_wx)
                    wy_min, wy_max = min(all_wy), max(all_wy)
                    actual_w = max(wx_max - wx_min, 10.0) * 1.2
                    actual_h = max(wy_max - wy_min, 10.0) * 1.2
                    # 原点偏移到车辆范围左下角
                    grid_ox = wx_min - (actual_w - (wx_max - wx_min)) / 2
                    grid_oy = wy_min - (actual_h - (wy_max - wy_min)) / 2
                    print(f"[INFO] 网格校正: 原点=({grid_ox:.1f}, {grid_oy:.1f}), "
                          f"范围={actual_w:.1f}m x {actual_h:.1f}m")

                    conflict_analyzer = ConflictAnalyzer(
                        grid_size=cfg.grid_size,
                        world_width_m=actual_w,
                        world_height_m=actual_h,
                        origin_x=grid_ox,
                        origin_y=grid_oy,
                        v_ref=risk_params.v_ref,
                    )
                    attributor = CongestionAttributor(
                        grid_size=cfg.grid_size,
                        world_width_m=actual_w,
                        world_height_m=actual_h,
                        origin_x=grid_ox,
                        origin_y=grid_oy,
                        v_ref=risk_params.v_ref,
                    )
                    ablation_study = AblationStudy(
                        grid_size=cfg.grid_size,
                        world_width_m=actual_w,
                        world_height_m=actual_h,
                        origin_x=grid_ox,
                        origin_y=grid_oy,
                        v_ref=risk_params.v_ref,
                    )
                    _analyzer_initialized = True
                should_analyze = (
                    cfg.backtest or
                    (phi > risk_params.phi_plot_threshold and
                     frame_index - last_congestion_frame >= cfg.realtime_congestion_interval)
                )
                if should_analyze:
                    last_congestion_frame = frame_index

                    vehicles_for_analysis = []
                    for tid, meta in current_meta.items():
                        vehicles_for_analysis.append({
                            'track_id': tid,
                            'cx': meta['world_x'],
                            'cy': meta['world_y'],
                            'speed_mps': meta.get('speed_mps', 0),
                            'heading_deg': meta.get('heading_deg', 0),
                            'label': meta.get('label', 'car'),
                        })

                    if vehicles_for_analysis:
                        # DEBUG: 打印冲突分析输入数据
                        if frame_index % 30 == 0:
                            headings = [v['heading_deg'] for v in vehicles_for_analysis]
                            world_xs = [v['cx'] for v in vehicles_for_analysis]
                            world_ys = [v['cy'] for v in vehicles_for_analysis]
                            print(f"  [DEBUG] vehicles={len(vehicles_for_analysis)}, "
                                  f"heading range=[{min(headings):.0f}, {max(headings):.0f}], "
                                  f"heading unique={len(set(round(h,0) for h in headings))}, "
                                  f"world_x=[{min(world_xs):.1f}, {max(world_xs):.1f}], "
                                  f"world_y=[{min(world_ys):.1f}, {max(world_ys):.1f}]")

                        conflict_result = conflict_analyzer.analyze(vehicles_for_analysis)
                        influences = conflict_result.influences

                        # DEBUG: 打印归因结果
                        if frame_index % 30 == 0:
                            pos_inf = [i for i in influences if i > 0]
                            print(f"  [DEBUG] influences: non-zero={len(pos_inf)}/{len(influences)}, "
                                  f"max={max(influences) if influences else 0:.6f}, "
                                  f"conflict_max={conflict_result.conflict_max:.6f}")

                        # 追踪冲突点数最多的帧
                        n_conflict_now = sum(1 for inf in influences if inf > 0)
                        if n_conflict_now > max_conflict_info['count']:
                            max_conflict_info['count'] = n_conflict_now
                            max_conflict_info['phi'] = phi
                            max_conflict_info['timestamp'] = timestamp
                            max_conflict_info['frame_index'] = frame_index
                            max_conflict_info['conflict_result'] = conflict_result
                            max_conflict_info['influences'] = list(influences)
                            max_conflict_info['vehicles'] = list(vehicles_for_analysis)
                            # combined 在可视化之后赋值，这里先标记需要保存

                # 记录事件结束信息（实际导出在可视化之后执行）
                self._pending_event = event_info if event_info and event_info.get('type') == 'end' else None
            perf['phi'] += time_mod.perf_counter() - _tick; _tick = time_mod.perf_counter()

            # ── Backtest 调试窗口（冲突检测内部可视化）─────────
            if cfg.backtest and cfg.show_windows:
                if conflict_result is not None:
                    debug_img = render_conflict_debug_window(
                        bev_frame=bev_frame,
                        conflict_result=conflict_result,
                        current_meta=current_meta,
                        grid_cfg=conflict_analyzer.grid_cfg,
                        cell_size_m=conflict_analyzer.grid_cfg.cell_size_m,
                        panel_size=(1600, 900),
                    )
                    cv2.imshow('Conflict Debug (Backtest)', debug_img)
                # 无冲突结果时也显示 BEV，保持窗口存在
                elif frame_index % 30 == 0:
                    debug_blank = np.full((900, 1600, 3), (30, 30, 30), dtype=np.uint8)
                    cv2.putText(debug_blank, "Waiting for Phi > threshold to trigger conflict analysis...",
                               (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
                    cv2.imshow('Conflict Debug (Backtest)', debug_blank)

            # ── 步骤 8: 可视化（直接构建 1920x1080，省去最终 resize）──
            # ── 步骤 8a: 可视化合成（1920x1080 三行布局）──
            # Row1: 视频+BEV小图+数据 | Row2: 冲突分析全宽 | Row3: Phi时间线

            OUT_W, OUT_H = 1920, 1080
            # 布局重构：视频+BEV+数据(上) | 冲突分析全宽(中) | Phi(下)
            ROW1_H = 400     # 上：视频+冲突分析+数据（三列并排）
            ROW2_H = 500     # 中：BEV全宽（大幅提升）
            ROW3_H = OUT_H - ROW1_H - ROW2_H  # = 180：Phi时间线

            max_inf = max(influences) if influences and max(influences) > 0 else 1.0

            tid_to_inf = {}
            if influences and conflict_result:
                for i, v in enumerate(conflict_result.vehicles):
                    tid_to_inf[v.get('track_id', i)] = influences[i]

            # ── 新布局：Row1(视频+BEV+数据) | Row2(冲突分析全宽) | Row3(Phi) ──
            # Row1 三等分
            ROW1_COL_W = OUT_W // 3  # = 640
            panel_a = letterbox_to(det_result.annotated_frame, (ROW1_COL_W, ROW1_H), bg_color=(30, 30, 30))

            # 中间列：小BEV鸟瞰图
            bev_small = letterbox_to(bev_frame, (ROW1_COL_W, ROW1_H), bg_color=(30, 30, 30))

            # ── C：冲突热力 + 框内填充 + 箭头（Row2 全宽）──
            panel_c = np.full((ROW2_H, OUT_W, 3), (20, 20, 20), dtype=np.uint8)
            # 按最小比例缩放，确保BEV内容完整显示在面板内
            scale_c = min(OUT_W / max(bev_w, 1), ROW2_H / max(bev_h, 1))

            # 底层：冲突热力图（映射到 BEV 像素坐标）
            if conflict_result and conflict_result.conflict_max > 0:
                try:
                    inv_h = np.linalg.inv(h_mat)
                except np.linalg.LinAlgError:
                    inv_h = None
                if inv_h is not None:
                    C = conflict_result.conflict_field
                    grid_size = C.shape[0]
                    # 使用冲突分析器的实际网格参数（可能与配置值不同）
                    cell_m = conflict_analyzer.grid_cfg.cell_size_m if conflict_analyzer else cfg.world_width_m / grid_size
                    heat_layer = np.zeros((ROW2_H, OUT_W, 3), dtype=np.uint8)

                    # 批量网格坐标 → 批量 BEV 投影（一次调用，性能提升 50x）
                    step = max(2, grid_size // 32)
                    gi_arr = np.arange(0, grid_size, step, dtype=np.float32)
                    gj_arr = np.arange(0, grid_size, step, dtype=np.float32)
                    gv, gu = np.meshgrid(gj_arr, gi_arr, indexing='xy')
                    world_pts = np.stack([
                        (gu + 0.5) * cell_m, (gv + 0.5) * cell_m
                    ], axis=-1).reshape(-1, 1, 2).astype(np.float32)

                    # 世界 → 图像 → BEV（各一次批量调用）
                    img_pts = cv2.perspectiveTransform(world_pts, inv_h).reshape(-1, 2)
                    bev_pts = cv2.perspectiveTransform(
                        img_pts.reshape(-1, 1, 2).astype(np.float32), H_bev
                    ).reshape(-1, 2)

                    # 投影到 C 面板
                    bx = (bev_pts[:, 0] * scale_c).astype(np.int32)
                    by = (bev_pts[:, 1] * scale_c).astype(np.int32)
                    vals = C[gv.astype(np.int32), gu.astype(np.int32)].ravel()
                    cmax = conflict_result.conflict_max

                    # 向量化热力图渲染（替代逐 cell cv2.rectangle 循环）
                    sz = max(2, int(step * scale_c * 0.7))
                    valid = (vals > 0) & (bx >= 0) & (bx < OUT_W) & (by >= 0) & (by < ROW2_H)
                    bx_v, by_v, vals_v = bx[valid], by[valid], vals[valid]
                    if len(bx_v) > 0:
                        intensities = np.clip((vals_v / cmax * 255).astype(np.int32), 0, 255)
                        for dx in range(-sz, sz + 1):
                            for dy in range(-sz, sz + 1):
                                px = np.clip(bx_v + dx, 0, OUT_W - 1)
                                py = np.clip(by_v + dy, 0, ROW2_H - 1)
                                heat_layer[py, px, 0] = np.maximum(heat_layer[py, px, 0], 0)
                                heat_layer[py, px, 1] = np.maximum(heat_layer[py, px, 1], 0)
                                heat_layer[py, px, 2] = np.maximum(heat_layer[py, px, 2], intensities)
                    panel_c = cv2.addWeighted(heat_layer, 0.4, panel_c, 0.6, 0)

            # 单遍绘制：填充 + 边框 + 箭头 + ID（合并原双重循环）
            fill_layer = np.zeros_like(panel_c)
            for tid, meta in current_meta.items():
                bev_x = meta.get('bev_x', 0)
                bev_y = meta.get('bev_y', 0)
                cx_c = int(bev_x * scale_c)
                cy_c = int(bev_y * scale_c)
                if not (0 <= cx_c < OUT_W and 0 <= cy_c < ROW2_H):
                    continue

                label = meta.get('label', 'car')
                size_info = vehicle_size_m.get(label, {'length_m': 4.0, 'width_m': 1.6})
                half_l = size_info['length_m'] * ppm * 0.65 / 2 * scale_c
                half_w = size_info['width_m'] * ppm * 0.65 / 2 * scale_c

                heading = meta.get('heading_deg', 0)
                rad = np.radians(heading)
                cos_h, sin_h = np.cos(rad), np.sin(rad)

                corners = [(-half_l, -half_w), (half_l, -half_w),
                           (half_l, half_w), (-half_l, half_w)]
                rotated = [(int(cx_c + c[0]*cos_h - c[1]*sin_h),
                            int(cy_c - c[0]*sin_h - c[1]*cos_h)) for c in corners]
                pts = np.array(rotated, dtype=np.int32)

                # 填充色：红色深度=拥堵贡献度
                inf = tid_to_inf.get(tid, 0)
                if inf > 0 and max_inf > 0:
                    ratio = min(inf / max_inf, 1.0)
                    fill_color = (int(20 + 20 * (1 - ratio)),
                                  int(10 + 20 * (1 - ratio)),
                                  int(40 + 200 * ratio))
                    cv2.fillPoly(fill_layer, [pts], fill_color)

                # 边框（统一细白线）+ 方向箭头（灰色）
                cv2.polylines(panel_c, [pts], True, (180, 180, 180), 1, cv2.LINE_AA)
                long_side_m = max(size_info['length_m'], size_info['width_m'])
                arrow_len = max(int(long_side_m * ppm * 0.65 * 1.5 * scale_c), 12)
                head_x = int(cx_c + arrow_len * cos_h)
                head_y = int(cy_c - arrow_len * sin_h)
                cv2.arrowedLine(panel_c, (cx_c, cy_c), (head_x, head_y),
                                (180, 180, 180), 1, cv2.LINE_AA, tipLength=0.15)

                # ID + 归因分数
                label_text = f"#{tid}"
                if inf > 0:
                    label_text += f" {inf:.4f}"
                draw_text_with_bg(panel_c, label_text, (cx_c + 8, cy_c - 8),
                                  color=(220, 220, 220), scale=0.35)

            # 混合填充层到背景（一次完成）
            panel_c = cv2.addWeighted(fill_layer, 0.55, panel_c, 0.45, 0)

            # ── Row 1: 视频 | 冲突分析 | 数据（三列并排）──
            conflict_count = sum(1 for inf in (influences or []) if inf > 0)
            conflict_total = len(conflict_result.vehicles) if conflict_result and conflict_result.vehicles else 0
            data_panel = np.full((ROW1_H, ROW1_COL_W, 3), (30, 30, 30), dtype=np.uint8)
            y = 16
            draw_text_with_bg(data_panel, f"Phi: {phi:.3f}", (10, y), (0, 200, 255), 0.7, 2)
            draw_text_with_bg(data_panel, f"Speed: {motion_stats['avg_speed_mps']:.1f} m/s", (10, y+32), (200,200,200), 0.55, 1)
            draw_text_with_bg(data_panel, f"Vehicles: {motion_stats['total_count']}", (10, y+56), (200,200,200), 0.55, 1)
            draw_text_with_bg(data_panel, f"Moving: {motion_stats['moving_count']}  Stationary: {motion_stats['stationary_count']}  Parked: {motion_stats['parked_count']}", (10, y+80), (200,200,200), 0.5, 1)
            draw_text_with_bg(data_panel, f"Interwoven: {conflict_count}/{conflict_total}", (10, y+108), (255,120,120), 0.55, 1)
            draw_text_with_bg(data_panel, f"Avg Speed: {motion_stats['avg_speed_mps']:.1f} m/s", (10, y+132), (100,255,100), 0.5, 1)

            # Top-3 交织车辆
            if influences and conflict_result and conflict_count > 0:
                ranked = conflict_result.get_vehicles_ranked_by_influence()
                for ri in range(min(3, len(ranked))):
                    idx, inf, v = ranked[ri]
                    if inf > 0:
                        tid = v.get('track_id', idx)
                        lbl = v.get('label', 'car')
                        draw_text_with_bg(data_panel, f"  #{tid}({lbl}) {inf:.4f}",
                                        (10, y + ri*20 + 160), (255,180,180), 0.4, 1)

            row1 = np.hstack([panel_a, bev_small, data_panel])

            # ── Row 3: Phi 时间线图表（全宽）──────────────────
            phi_chart = render_phi_chart_panel(
                (ROW3_H, OUT_W),
                event_tracker.phi_history,
                event_tracker.threshold,
            )

            # ── 全屏拼接：视频+BEV+数据(上) | 冲突分析(中) | Phi(下) ──
            combined = np.vstack([row1, panel_c, phi_chart])

            # 如果是当前冲突峰值帧，保存截图
            if (max_conflict_info['conflict_result'] is not None and
                    max_conflict_info['frame_index'] == frame_index):
                max_conflict_info['combined'] = combined.copy()

            perf['viz_c'] += time_mod.perf_counter() - _tick; _tick = time_mod.perf_counter()

            # 事件导出（在可视化之后，可带截图）
            if self._pending_event:
                self._export_event(self._pending_event, conflict_result, attributor,
                                   ablation_study, risk_params, cfg, combined, current_meta)
                self._pending_event = None

            # ── 步骤 10: 输出 ────────────────────────────────
            # 写入实时预览
            if (cfg.live_dir and
                    frame_index - last_live_write_frame >= cfg.live_write_interval):
                last_live_write_frame = frame_index
                self._write_live_preview(
                    combined, phi, motion_stats, event_tracker, cfg, async_writer,
                )

            # 显示
            if cfg.show_windows:
                display = fit_for_display(combined, display_scale=cfg.display_scale)
                cv2.imshow('Visual System - BEV Analysis', display)
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break

            perf['write'] += time_mod.perf_counter() - _tick

            frame_index += 1
            if cfg.max_frames > 0 and frame_index >= cfg.max_frames:
                break

        # ── 性能报告 ──
        total_t = sum(perf.values())
        print(f"\n[PERF] 共 {frame_index} 帧，总耗时 {total_t:.2f}s")
        if total_t > 0:
            for k, v in sorted(perf.items(), key=lambda x: -x[1]):
                pct = v / total_t * 100
                avg_ms = v / max(frame_index, 1) * 1000
                print(f"  {k:>8}: {v:.2f}s ({pct:4.1f}%)  avg {avg_ms:.1f}ms/frame")
            fps = frame_index / max(total_t, 1e-6)
            print(f"  处理FPS: {fps:.1f}")
            print(f"  固定stride: {stride}")
        print()

        # ── 导出冲突峰值帧 ──────────────────────────────────────
        if max_conflict_info['count'] > 0 and cfg.events_dir:
            peak_dir = Path(cfg.events_dir) / "peak_conflict"
            peak_dir.mkdir(parents=True, exist_ok=True)

            # 截图
            if max_conflict_info.get('combined') is not None:
                cv2.imwrite(str(peak_dir / 'peak_screenshot.jpg'),
                           max_conflict_info['combined'])

            # 摘要
            summary = {
                'max_conflict_count': max_conflict_info['count'],
                'phi_at_peak': max_conflict_info['phi'],
                'timestamp_s': max_conflict_info['timestamp'],
                'frame_index': max_conflict_info['frame_index'],
                'total_vehicles': len(max_conflict_info['vehicles']) if max_conflict_info['vehicles'] else 0,
            }
            (peak_dir / 'peak_summary.json').write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',
            )

            # 车辆归因排名
            cr = max_conflict_info['conflict_result']
            if cr and cr.vehicles:
                ranked = cr.get_vehicles_ranked_by_influence()
                with open(peak_dir / 'vehicle_influence_ranking.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['rank', 'track_id', 'influence_score', 'direction_bin', 'label'])
                    for idx, inf, v in ranked:
                        writer.writerow([idx + 1, v.get('track_id', ''), f"{inf:.6f}",
                                         v.get('heading_deg', 0), v.get('label', '')])

                # 渲染排名表格图片
                from core.conflict import BIN_NAMES
                row_h = 36
                header_h = 50
                img_h = header_h + len(ranked) * row_h + 20
                img_w = 700
                rank_img = np.full((img_h, img_w, 3), (30, 30, 30), dtype=np.uint8)

                # 标题
                cv2.putText(rank_img, "Vehicle Influence Ranking (Peak Conflict Frame)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)

                # 表头
                y = header_h
                cv2.line(rank_img, (0, y), (img_w, y), (80, 80, 80), 1)
                headers = [('Rank', 10), ('ID', 60), ('Influence', 130), ('Bin', 310), ('Dir', 380), ('Label', 470)]
                for text, x in headers:
                    cv2.putText(rank_img, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                y += 5

                # 数据行
                for idx, inf, v in ranked:
                    y += row_h
                    cv2.line(rank_img, (0, y), (img_w, y), (50, 50, 50), 1)

                    # 归因分数越高越红
                    ratio = min(inf / max(max_conflict_info['influences']) if max_conflict_info['influences'] else 1, 1.0) if inf > 0 else 0
                    score_color = (int(40*(1-ratio)), int(40*(1-ratio)), int(200*ratio+55)) if inf > 0 else (120, 120, 120)

                    cv2.putText(rank_img, f"{idx+1}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(rank_img, f"#{v.get('track_id', '')}", (60, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(rank_img, f"{inf:.6f}", (130, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, score_color, 1, cv2.LINE_AA)
                    bin_idx = int(v.get('heading_deg', 0) % 360 / 30) % 12
                    bin_name = BIN_NAMES[bin_idx] if bin_idx < len(BIN_NAMES) else str(bin_idx)
                    cv2.putText(rank_img, f"bin{bin_idx}", (310, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(rank_img, bin_name, (380, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
                    cv2.putText(rank_img, v.get('label', 'car'), (470, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

                cv2.imwrite(str(peak_dir / 'vehicle_influence_ranking.jpg'), rank_img)

                # 消融实验
                if cfg.ablation_enable and max_conflict_info['vehicles']:
                    ablation_results = ablation_study.run(
                        max_conflict_info['vehicles'], risk_params, cfg.ablation_levels,
                    )
                    ablation_study.export_csv(ablation_results, peak_dir / 'ablation_results.csv')
                    ablation_study.export_json(ablation_results, peak_dir / 'ablation_results.json')

            print(f"[INFO] 冲突峰值帧已导出到: {peak_dir}")
            print(f"  最大交织点数: {max_conflict_info['count']}")
            print(f"  对应 Phi: {max_conflict_info['phi']:.3f}")
            print(f"  时间戳: {max_conflict_info['timestamp']:.1f}s")

        # ── 清理 ──────────────────────────────────────────────
        cap.release()
        if async_writer:
            async_writer.close()
        if cfg.show_windows:
            cv2.destroyAllWindows()
        gc.collect()
        print(f"[INFO] 主循环结束，共处理 {frame_index} 帧")

    def _export_event(self, event_info, conflict_result, attributor,
                      ablation_study, risk_params, cfg,
                      combined=None, current_meta=None):
        """导出事件分析结果"""
        if not cfg.events_dir:
            return

        event_dir = Path(cfg.events_dir) / f"event_{int(event_info['start_time']):06d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        # ── 截图 ──
        if combined is not None:
            cv2.imwrite(str(event_dir / 'event_screenshot.jpg'), combined)

        # ── 事件摘要 ──
        summary = {
            'start_time': event_info['start_time'],
            'end_time': event_info['end_time'],
            'peak_phi': event_info['peak_phi'],
            'duration_s': event_info['duration_s'],
        }
        (event_dir / 'event_summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',
        )

        if conflict_result and conflict_result.vehicles:
            # ── 车辆归因排名 ──
            ranked = conflict_result.get_vehicles_ranked_by_influence()
            with open(event_dir / 'vehicle_influence_ranking.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['rank', 'track_id', 'influence_score', 'direction_bin', 'label'])
                for idx, inf, v in ranked:
                    writer.writerow([idx + 1, v.get('track_id', ''), f"{inf:.6f}",
                                     v.get('heading_deg', 0), v.get('label', '')])

            # ── 车辆速度 CSV ──
            with open(event_dir / 'vehicle_speed.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['track_id', 'label', 'speed_mps', 'world_x', 'world_y'])
                for v in conflict_result.vehicles:
                    writer.writerow([
                        v.get('track_id', ''), v.get('label', ''),
                        f"{v.get('speed_mps', 0):.2f}",
                        f"{v.get('cx', 0):.2f}", f"{v.get('cy', 0):.2f}",
                    ])

            # ── 交织点-车辆匹配 ──
            with open(event_dir / 'interweaving_pairs.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['track_id_a', 'label_a', 'heading_a', 'influence_a',
                                 'track_id_b', 'label_b', 'heading_b', 'influence_b'])
                vehicles_list = conflict_result.vehicles
                influences_list = conflict_result.influences
                for i in range(len(vehicles_list)):
                    for j in range(i + 1, len(vehicles_list)):
                        if influences_list[i] > 0 and influences_list[j] > 0:
                            va, vb = vehicles_list[i], vehicles_list[j]
                            writer.writerow([
                                va.get('track_id', i), va.get('label', ''),
                                f"{va.get('heading_deg', 0):.0f}", f"{influences_list[i]:.6f}",
                                vb.get('track_id', j), vb.get('label', ''),
                                f"{vb.get('heading_deg', 0):.0f}", f"{influences_list[j]:.6f}",
                            ])

            # ── 消融实验 ──
            if cfg.ablation_enable:
                ablation_results = ablation_study.run(
                    conflict_result.vehicles, risk_params, cfg.ablation_levels,
                )
                ablation_study.export_csv(ablation_results, event_dir / 'ablation_results.csv')
                ablation_study.export_json(ablation_results, event_dir / 'ablation_results.json')

    def _write_live_preview(self, combined, phi, motion_stats, event_tracker,
                            cfg, async_writer):
        """写入实时预览文件"""
        if not cfg.live_dir:
            return

        live_dir = Path(cfg.live_dir)
        live_dir.mkdir(parents=True, exist_ok=True)

        # 写入合成图
        preview = cv2.resize(combined, (960, 540))
        if async_writer:
            async_writer.enqueue(str(live_dir / 'window_3_third_phi.jpg'), preview)
        else:
            cv2.imwrite(str(live_dir / 'window_3_third_phi.jpg'), preview)

        # 指标 JSON
        metrics = {
            'phi_t': round(phi, 4),
            'vehicle_total': motion_stats['total_count'],
            'parked_count': motion_stats['parked_count'],
            'avg_speed_mps': round(motion_stats['avg_speed_mps'], 2),
            'frame_index': event_tracker.frame_count,
        }
        (live_dir / 'live_metrics.json').write_text(
            json.dumps(metrics), encoding='utf-8',
        )

        # Phi 时间线 CSV
        with open(live_dir / 'live_phi_timeline.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s', 'phi_t'])
            for t, p in list(event_tracker.phi_history)[-200:]:
                writer.writerow([f"{t:.2f}", f"{p:.4f}"])

    def stop(self):
        """停止主循环"""
        self._running = False
