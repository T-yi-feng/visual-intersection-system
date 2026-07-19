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
from analysis.root_cause import compute_root_cause, root_cause_to_pct
from visualization.phi_chart import render_phi_chart_panel
from visualization.conflict_debug import render_conflict_debug_window
from visualization.data_panel import render_data_panel
from visualization.effects import draw_glow_box
from utils.drawing import fit_for_display, draw_text_with_bg, letterbox_to
from utils.config_loader import load_vehicle_size_map
from utils.theme import THEME, phi_color, attr_color, phi_label_en
from utils.async_writer import AsyncLivePreviewWriter

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

        # 站点
        self.site_name = getattr(args, 'site_key', 'default')

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
        h_mat, raw_img_pts, world_pts = load_homography(cfg.homography_path)
        img_pts = raw_img_pts.copy()

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

        # 归因平滑状态（方案 B + C）
        self._attr_history: dict[int, deque] = {}  # track_id → 最近 N 帧归因
        self._hot_counter: dict[int, int] = {}      # track_id → 持续高亮计数
        ATTR_SMOOTH_WINDOW = 5                      # 滑动窗口大小

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

        # ── 自动缩放标定点以匹配实际视频分辨率（circle 视频 852x480 但标定点到 2525x1377） ──
        if frame_w > 0 and frame_h > 0:
            scale_x = frame_w / max(raw_img_pts[:, 0].max(), 1)
            scale_y = frame_h / max(raw_img_pts[:, 1].max(), 1)
            if scale_x < 0.8 or scale_y < 0.8:
                scale = min(scale_x, scale_y)
                img_pts = (raw_img_pts * scale).astype(np.float64)
                h_mat, _ = cv2.findHomography(
                    img_pts.reshape(-1, 1, 2).astype(np.float32),
                    world_pts.reshape(-1, 1, 2).astype(np.float32),
                )
            else:
                img_pts = raw_img_pts.copy()

        # ── 构建 BEV 变换矩阵 ────────────────────────────────
        # 使用标定文件中的 image_points 作为 warp 源四边形
        # 取代旧的硬编码 _DEFAULT_QUAD，确保 BEV 与世界坐标对齐
        world_w = world_pts[:, 0].max() - world_pts[:, 0].min()
        world_h = world_pts[:, 1].max() - world_pts[:, 1].min()

        # BEV 输出尺寸：固定长边，短边按世界比例适配，避免变形
        BEV_LONG = max(cfg.bev_width, cfg.bev_height)
        if world_w >= world_h:
            bev_w = BEV_LONG
            bev_h = max(int(BEV_LONG * world_h / max(world_w, 1e-6)), 200)
        else:
            bev_h = BEV_LONG
            bev_w = max(int(BEV_LONG * world_w / max(world_h, 1e-6)), 200)

        # 用标定文件的 4 个角点作为 H_bev 的源（与 h_mat 的坐标系一致）
        src_pts = img_pts.astype(np.float32).reshape(4, 2)
        dst_pts = np.array([
            [0, 0],
            [bev_w - 1, 0],
            [bev_w - 1, bev_h - 1],
            [0, bev_h - 1],
        ], dtype=np.float32)
        H_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 像素/米（世界坐标宽度方向的一致性比例）
        ppm = bev_w / max(world_w, 1e-6)

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
            MIN_DISPLACEMENT = 3.0    # 最小位移阈值（BEV 像素，≈9cm @ ppm=32）

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

            # ── 归因平滑：方案 B (中位数窗口) + C (驻留时间) ──
            raw_inf = {}
            if influences and conflict_result:
                for i, v in enumerate(conflict_result.vehicles):
                    tid = v.get('track_id', i)
                    raw_inf[tid] = influences[i]

            # 方案 B: 5帧中位数窗口
            if raw_inf:
                for tid, val in raw_inf.items():
                    if tid not in self._attr_history:
                        self._attr_history[tid] = deque(maxlen=5)
                    self._attr_history[tid].append(val)
                    if len(self._attr_history[tid]) >= 3:
                        window = sorted(self._attr_history[tid])
                        raw_inf[tid] = window[len(window)//2]  # median

            # 方案 C: 驻留时间 —— 更新计数器（相对于 max_inf）
            _max_r = max(raw_inf.values()) if raw_inf else 1.0
            tid_to_inf = {}
            highlighted_tids = set()
            for tid, val in raw_inf.items():
                inf_pct = (val / max(_max_r, 1e-8)) * 100.0
                if tid not in self._hot_counter:
                    self._hot_counter[tid] = 0
                if inf_pct > 15.0:
                    self._hot_counter[tid] = min(self._hot_counter[tid] + 1, 10)
                else:
                    self._hot_counter[tid] = max(self._hot_counter[tid] - 1, -5)
                if self._hot_counter[tid] > 2:   # 连续3帧>阈值才高亮
                    highlighted_tids.add(tid)
                    # 用窗口最大值稳定显示
                    if self._attr_history.get(tid):
                        val = max(self._attr_history[tid])
                tid_to_inf[tid] = val

            max_inf = max(tid_to_inf.values()) if tid_to_inf and max(tid_to_inf.values()) > 0 else 1.0

            # ── 因果溯源：必须先于渲染执行 ──
            root_cause_tids = set()
            tid_to_rc = {}
            root_cause_pct = None
            if conflict_result and conflict_result.vehicles:
                rc_vehicles = []
                for v in conflict_result.vehicles:
                    tid = v.get('track_id', 0)
                    meta = current_meta.get(tid, {})
                    rc_vehicles.append({
                        'track_id': tid,
                        'world_x': meta.get('world_x', v.get('cx', 0)),
                        'world_y': meta.get('world_y', v.get('cy', 0)),
                        'speed_mps': v.get('speed_mps', 0),
                        'heading_deg': v.get('heading_deg', 0),
                    })
                root_cause_scores = compute_root_cause(
                    rc_vehicles, influences, conflict_result.conflict_field,
                    conflict_analyzer.grid_cfg,
                )
                root_cause_pct = root_cause_to_pct(root_cause_scores)
                # 标记根因最高的 Top-2 车辆
                rc_indices = np.argsort(root_cause_scores)[::-1]
                for rank in range(min(2, len(rc_indices))):
                    idx = rc_indices[rank]
                    if root_cause_scores[idx] > root_cause_scores.min():
                        tid = conflict_result.vehicles[idx].get('track_id', idx)
                        tid_to_rc[tid] = root_cause_pct[idx] if idx < len(root_cause_pct) else 0
                        root_cause_tids.add(tid)
                if frame_index % 20 == 0:
                    ranked = conflict_result.get_vehicles_ranked_by_influence()
                    n_red = len(root_cause_tids)
                    max_rc = max(root_cause_pct) if len(root_cause_pct) > 0 else 0
                    print(f"  [ROOT CAUSE] n_red={n_red} max_cause={max_rc:.1f}% tids={sorted(root_cause_tids)[:5]}")
                    for ri in range(min(3, len(ranked))):
                        idx, iv, v = ranked[ri]
                        inf_pct = (iv / max(max(influences), 1e-8)) * 100.0
                        rc_pct_v = root_cause_pct[idx] if idx < len(root_cause_pct) else 0
                        tid = v.get('track_id', idx)
                        lbl = v.get('label', 'car')
                        star = " <<<" if tid in root_cause_tids else ""
                        print(f"    #{tid} {lbl}: Inf={inf_pct:.1f}%  Cause={rc_pct_v:.1f}%{star}")

            # ── 在原始视频帧上高亮标记 ──
            # 橙色 = 高参与度 | 亮红 = 因果根因
            HIGHLIGHT_ORANGE = (50, 140, 230)
            HIGHLIGHT_RED = (50, 50, 255)
            if (highlighted_tids or root_cause_tids) and det_result.vehicles:
                for v in det_result.vehicles:
                    tid = v.get('track_id', -1)
                    if tid in root_cause_tids:
                        color = HIGHLIGHT_RED
                        tag = "ROOT CAUSE"
                    elif tid in highlighted_tids:
                        color = HIGHLIGHT_ORANGE
                        tag = "CONGESTION"
                    else:
                        continue
                    bb = v.get('bbox', (0,0,0,0))
                    x1, y1, x2, y2 = map(int, bb)
                    if x2 - x1 > 0:
                        cv2.rectangle(det_result.annotated_frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(det_result.annotated_frame, f"#{tid} {tag}",
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # ── Row1(视频+BEV+数据) | Row2(冲突分析全宽) | Row3(Phi) ──
            ROW1_COL_W = OUT_W // 3
            panel_a = letterbox_to(det_result.annotated_frame, (ROW1_COL_W, ROW1_H),
                                   bg_color=THEME["bg_canvas"])

            bev_small = letterbox_to(bev_frame, (ROW1_COL_W, ROW1_H),
                                     bg_color=THEME["bg_canvas"])

            # ── C：车辆框 + 箭头（Row2 全宽，BEV 底图）──
            panel_c = np.full((ROW2_H, OUT_W, 3), THEME["bg_canvas"], dtype=np.uint8)
            scale_c = min(OUT_W / max(bev_w, 1), ROW2_H / max(bev_h, 1))

            fill_layer = np.zeros_like(panel_c)
            for tid, meta in current_meta.items():
                bev_x = meta.get('bev_x', 0)
                bev_y = meta.get('bev_y', 0)
                cx_c = int(bev_x * scale_c)
                cy_c = int(bev_y * scale_c)
                if not (0 <= cx_c < OUT_W and 0 <= cy_c < ROW2_H):
                    continue

                vlabel = meta.get('label', 'car')
                size_info = vehicle_size_m.get(vlabel, {'length_m': 4.0, 'width_m': 1.6})
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

                inf = tid_to_inf.get(tid, 0)
                inf_pct = (inf / max(max_inf, 1e-8)) * 100.0

                if tid in root_cause_tids:
                    # 因果根因：亮红填充 + 发光 + 粗红边框
                    cv2.fillPoly(fill_layer, [pts], HIGHLIGHT_RED)
                    draw_glow_box(panel_c, pts, HIGHLIGHT_RED, glow_radius=4)
                    cv2.polylines(panel_c, [pts], True, HIGHLIGHT_RED, 2, cv2.LINE_AA)
                    draw_text_with_bg(panel_c, f"#{tid} CAUSE",
                                      (cx_c + 8, cy_c - 10),
                                      color=HIGHLIGHT_RED, scale=0.45)
                elif inf_pct > 15.0:
                    # 高参与度：橙色填充 + 发光 + 粗橙边框 + 大字标签
                    cv2.fillPoly(fill_layer, [pts], HIGHLIGHT_ORANGE)
                    draw_glow_box(panel_c, pts, HIGHLIGHT_ORANGE, glow_radius=4)
                    cv2.polylines(panel_c, [pts], True, HIGHLIGHT_ORANGE, 2, cv2.LINE_AA)
                    draw_text_with_bg(panel_c, f"#{tid} {inf_pct:.0f}%",
                                      (cx_c + 8, cy_c - 10),
                                      color=HIGHLIGHT_ORANGE, scale=0.45)
                elif inf_pct > 5.0:
                    # 中归因：橙色填充 + 标准边框
                    mid_color = (80, 150, 220)
                    cv2.fillPoly(fill_layer, [pts], mid_color)
                    cv2.polylines(panel_c, [pts], True, THEME["text_secondary"], 1, cv2.LINE_AA)
                    draw_text_with_bg(panel_c, f"#{tid}",
                                      (cx_c + 6, cy_c - 6),
                                      color=THEME["text_secondary"], scale=0.32)
                else:
                    # 低归因：淡色填充 + 细线
                    if inf > 0 and max_inf > 0:
                        cv2.fillPoly(fill_layer, [pts], (60, 80, 100))
                    cv2.polylines(panel_c, [pts], True, THEME["text_dim"], 1, cv2.LINE_AA)

                # 方向箭头
                speed = meta.get('speed_mps', 0)
                long_side_m = max(size_info['length_m'], size_info['width_m'])
                base_arrow = max(int(long_side_m * ppm * 0.65 * 1.2 * scale_c), 10)
                speed_factor = 1.0 + 0.5 * min(speed / 5.0, 2.0)
                arrow_len = int(base_arrow * speed_factor)
                head_x = int(cx_c + arrow_len * cos_h)
                head_y = int(cy_c - arrow_len * sin_h)
                arrow_color = HIGHLIGHT_RED if tid in root_cause_tids else (HIGHLIGHT_ORANGE if tid in highlighted_tids else (180, 180, 180))
                cv2.arrowedLine(panel_c, (cx_c, cy_c), (head_x, head_y),
                                arrow_color, 1, cv2.LINE_AA, tipLength=0.12)

            panel_c = cv2.addWeighted(fill_layer, 0.50, panel_c, 0.50, 0)

            # ── Row 1: 视频 | 小BEV | 数据面板（三列并排）──
            conflict_count = sum(1 for inf in (influences or []) if inf > 0)
            conflict_total = len(conflict_result.vehicles) if conflict_result and conflict_result.vehicles else 0
            data_panel = render_data_panel(
                (ROW1_H, ROW1_COL_W), phi, motion_stats,
                conflict_count, conflict_total,
                influences, conflict_result, tid_to_inf, max_inf,
                cfg.site_name, root_cause_pct,
            )

            row1 = np.hstack([panel_a, bev_small, data_panel])

            # ── Row 3: Phi 时间线图表（全宽）──────────────────
            ev_info = None
            if event_tracker.in_event:
                ev_info = {
                    'active': True,
                    'start_t': event_tracker.event_start_time,
                    'peak_phi': event_tracker.peak_phi,
                    'peak_t': event_tracker.peak_frame,
                    'duration_s': timestamp - event_tracker.event_start_time,
                }
            phi_chart = render_phi_chart_panel(
                (ROW3_H, OUT_W),
                event_tracker.phi_history,
                event_tracker.threshold,
                event_info=ev_info,
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
