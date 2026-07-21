"""
Full Pipeline Demo — scatter, bin, convolve, conflict field, attribute, ablate

Shows every step from vehicle scatter to conflict field computation and ablation.

Usage:
    python demo_test/convolution_pipeline_demo.py

Controls:
    Tab: switch between 6 pipeline steps
    Drag: move vehicle  Right-click: select  Scroll: heading
    Q: add  E: remove  S: save  L: load
"""

import json
import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from math import exp, cos, sin, pi, atan2
from core.conflict import (
    GridConfig, KernelConfig, DIRECTION_BINS, BIN_NAMES,
    DEFAULT_CONFLICT_PAIRS,
    scatter_vehicles_to_grid, build_all_directional_kernels,
    compute_conflict_field, compute_vehicle_influence,
)
from analysis.root_cause import compute_root_cause, root_cause_to_pct

WINDOW_W, WINDOW_H = 1400, 900
WORLD_SIZE_M = 60.0
GRID_SIZE = 64
CELL_SIZE_M = WORLD_SIZE_M / GRID_SIZE  # 0.9375m

# 颜色
BG = (18, 18, 22)
PANEL_BG = (26, 28, 34)
TEXT = (200, 200, 200)
DIM = (100, 100, 100)
ACCENT = (80, 200, 255)
WHITE = (255, 255, 255)

STEP_NAMES = [
    "1. Scatter: Vehicles -> Grid",
    "2. Direction Binning (12 bins)",
    "3. Kernel Convolution -> R_k",
    "4. Conflict Field C(x,y)",
    "5. Conflict Pairs (16 pairs)",
    "6. Ablation: Remove Top-K",
    "7. Root Cause: Water Drop Propagation",
]


def main():
    grid_cfg = GridConfig(GRID_SIZE, CELL_SIZE_M, 0, 0)
    kernel_cfg = KernelConfig(15, 4, 5.0, 0.8)
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)

    # 车辆 — 排队场景（全部向东car，前车越来越慢，队首已停 → 水滴逆流汇聚到队首）
    vehicles = [
        {'cx': 20, 'cy': 30, 'speed_mps': 4.0, 'heading_deg': 90, 'label': 'car', 'track_id': 0},   # 队尾（最快）
        {'cx': 26, 'cy': 30, 'speed_mps': 3.0, 'heading_deg': 90, 'label': 'car', 'track_id': 1},
        {'cx': 32, 'cy': 30, 'speed_mps': 2.0, 'heading_deg': 90, 'label': 'car', 'track_id': 2},
        {'cx': 38, 'cy': 30, 'speed_mps': 0.5, 'heading_deg': 90, 'label': 'car', 'track_id': 3},
        {'cx': 44, 'cy': 30, 'speed_mps': 0.0, 'heading_deg': 90, 'label': 'car', 'track_id': 4},   # 队首（停止→根因）
    ]
    drag_idx = -1
    sel_idx = 0
    current_step = 3  # 默认显示冲突场
    show_labels = True  # 车辆标签显示开关
    prev_O = None

    cv2.namedWindow("Convolution Pipeline Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Convolution Pipeline Demo", WINDOW_W, WINDOW_H)

    def mouse_cb(event, x, y, flags, param):
        nonlocal drag_idx, sel_idx
        # world coords from mouse
        p_x0, p_y0 = 20, 50
        p_size = 600
        wx = (x - p_x0) / p_size * WORLD_SIZE_M
        wy = (y - p_y0) / p_size * WORLD_SIZE_M

        if event == cv2.EVENT_LBUTTONDOWN:
            best, best_d = -1, 100
            for i, v in enumerate(vehicles):
                d = np.hypot(wx - v['cx'], wy - v['cy'])
                if d < best_d: best_d, best = d, i
            if best >= 0: drag_idx = best

        elif event == cv2.EVENT_MOUSEMOVE and drag_idx >= 0:
            vehicles[drag_idx]['cx'] = max(1, min(WORLD_SIZE_M-1, wx))
            vehicles[drag_idx]['cy'] = max(1, min(WORLD_SIZE_M-1, wy))

        elif event == cv2.EVENT_RBUTTONDOWN:
            best, best_d = -1, 100
            for i, v in enumerate(vehicles):
                d = np.hypot(wx - v['cx'], wy - v['cy'])
                if d < best_d: best_d, best = d, i
            if best >= 0: sel_idx = best

        elif event == cv2.EVENT_MOUSEWHEEL:
            if vehicles:
                delta = -1 if flags > 0 else 1
                idx = min(sel_idx, len(vehicles)-1)
                vehicles[idx]['heading_deg'] = (vehicles[idx]['heading_deg'] + delta * 10) % 360

        elif event == cv2.EVENT_LBUTTONUP:
            drag_idx = -1

    cv2.setMouseCallback("Convolution Pipeline Demo", mouse_cb)

    print("=== Convolution Pipeline Demo ===")
    print("1-7: switch step | Drag: move  Right-click: select  Scroll: heading")
    print("Drag: move  Right-click: select  Scroll: heading")
    print("Q: add  E: remove  S: save  L: load  T: toggle labels  ESC: exit")
    print()

    import traceback
    while True:
        canvas = np.full((WINDOW_H, WINDOW_W, 3), BG, dtype=np.uint8)

        # ── 左上主面板 ──
        p_x0, p_y0 = 20, 50
        p_size = 600
        main_panel = np.full((p_size, p_size, 3), PANEL_BG, dtype=np.uint8)

        # 网格线
        cv2.rectangle(main_panel, (0, 0), (p_size-1, p_size-1), (50, 50, 55), 1)
        step_g = max(GRID_SIZE // 16, 1)
        for i in range(0, GRID_SIZE, step_g * 2):
            x = int(i / GRID_SIZE * p_size)
            cv2.line(main_panel, (x, 0), (x, p_size), (34, 36, 42), 1)
        for i in range(0, GRID_SIZE, step_g * 2):
            y = int(i / GRID_SIZE * p_size)
            cv2.line(main_panel, (0, y), (p_size, y), (34, 36, 42), 1)

        # 执行冲突分析
        O, V, Theta, layers = scatter_vehicles_to_grid(vehicles, grid_cfg)
        layers_decomp = np.zeros((DIRECTION_BINS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        bin_size = 360.0 / DIRECTION_BINS
        for idx, v in enumerate(vehicles):
            h = v['heading_deg'] % 360
            for k in range(DIRECTION_BINS):
                center = k * bin_size
                diff = abs(h - center)
                if diff > 180: diff = 360 - diff
                if diff > 20: continue
                w = exp(-0.5 * (diff / 10.0) ** 2)
                gx = int(v['cx'] / grid_cfg.cell_size_m)
                gy = int(v['cy'] / grid_cfg.cell_size_m)
                if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                    layers_decomp[k][gy, gx] += w

        conflict_pairs = DEFAULT_CONFLICT_PAIRS
        C, pair_results, R = compute_conflict_field(layers_decomp, kernels, conflict_pairs)
        influences = compute_vehicle_influence(vehicles, R, grid_cfg, conflict_pairs)

        # ── 根因溯源：水滴传播 ──
        # 映射 vehicle dict 到 root_cause 需要的字段名
        rc_vehicles = []
        for v in vehicles:
            rc_vehicles.append({
                'track_id': v.get('track_id', 0),
                'world_x': v.get('cx', 0),
                'world_y': v.get('cy', 0),
                'speed_mps': v.get('speed_mps', 0),
                'heading_deg': v.get('heading_deg', 0),
            })
        root_cause_scores = compute_root_cause(rc_vehicles, influences, C, grid_cfg)
        root_cause_pct = root_cause_to_pct(root_cause_scores)
        max_inf = max(influences) if influences and max(influences) > 0 else 1.0
        # 高亮标记
        # 橙色 = 冲突参与度高 (influence > 15%)
        highlighted_tids = set()
        for i, inf in enumerate(influences):
            pct = inf / max_inf * 100 if max_inf > 0 else 0
            if pct > 15.0:
                highlighted_tids.add(vehicles[i].get('track_id', i))

        # 红色 = 根因 Top-2 + 绝对冲突参与 > 5% + 车辆位置有冲突场
        root_cause_tids = set()
        rc_indices = np.argsort(root_cause_scores)[::-1]
        for rank in range(min(2, len(rc_indices))):
            idx = rc_indices[rank]
            tid = vehicles[idx].get('track_id', idx)
            inf_pct = influences[idx] / max_inf * 100 if idx < len(influences) and max_inf > 0 else 0
            # 车辆位置冲突值
            gx = int(vehicles[idx]['cx'] / grid_cfg.cell_size_m)
            gy = int(vehicles[idx]['cy'] / grid_cfg.cell_size_m)
            pos_conf = 0.0
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                pos_conf = float(C[gy, gx])
            if root_cause_scores[idx] > root_cause_scores.min() and inf_pct > 5.0 and pos_conf > 0:
                root_cause_tids.add(tid)

        # ── 根据当前步骤在主面板上绘制 ──
        if current_step == 0:
            # Step 1: 占用场
            O_disp = cv2.resize(O, (p_size, p_size), interpolation=cv2.INTER_NEAREST)
            alpha = 0.4
            mask = O_disp > 0
            overlay = main_panel.copy()
            overlay[mask] = (60, 140, 200)
            cv2.addWeighted(overlay, alpha, main_panel, 1-alpha, 0, dst=main_panel)
            # 画车辆编号
            for i, v in enumerate(vehicles):
                px = int(v['cx'] / WORLD_SIZE_M * p_size)
                py = int(v['cy'] / WORLD_SIZE_M * p_size)
                rr = int(CELL_SIZE_M / WORLD_SIZE_M * p_size * 0.6)
                cv2.circle(main_panel, (px, py), rr, ACCENT, -1)
                cv2.putText(main_panel, f"#{i}", (px-6, py+4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

        elif current_step == 1:
            # Step 2: 12 个方向箱（画一个小扇形图）
            for k in range(DIRECTION_BINS):
                angle = k * bin_size
                rad = np.radians(angle)
                cx_s, cy_s = p_size // 3, p_size // 2
                r = 80
                # 只有该方向有车才画
                if layers_decomp[k].sum() > 0:
                    c_val = int(float(min(layers_decomp[k].sum() * 80, 255)))
                    # 扇形
                    for a in range(int(angle - bin_size / 2), int(angle + bin_size / 2)):
                        rad_a = np.radians(a)
                        px = int(cx_s + r * cos(rad_a))
                        py = int(cy_s - r * sin(rad_a))
                        cv2.line(main_panel, (cx_s, cy_s), (px, py), (int(c_val*0.5), int(c_val*0.7), c_val), 1)
                    px_e = int(cx_s + (r+10) * cos(rad))
                    py_e = int(cy_s - (r+10) * sin(rad))
                    cv2.arrowedLine(main_panel, (cx_s, cy_s), (px_e, py_e), (int(c_val*0.5), int(c_val*0.7), c_val), 2, tipLength=0.15)
                    cv2.putText(main_panel, f"{BIN_NAMES[k]}", (px_e+2, py_e+2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, DIM, 1)

            # 在网格上叠加方向着色
            h_all = [v['heading_deg'] for v in vehicles]
            if h_all:
                colors = [(80, 140, 200), (200, 160, 80), (80, 200, 160), (200, 100, 100)]
                for i, v in enumerate(vehicles):
                    px = int(v['cx'] / WORLD_SIZE_M * p_size)
                    py = int(v['cy'] / WORLD_SIZE_M * p_size)
                    c = colors[i % len(colors)]
                    rr = int(CELL_SIZE_M / WORLD_SIZE_M * p_size * 0.5)
                    cv2.circle(main_panel, (px, py), rr, c, -1)
                    bin_k = int(v['heading_deg'] % 360 / bin_size) % DIRECTION_BINS
                    cv2.putText(main_panel, f"#{i} bin={bin_k}", (px+rr+2, py+2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)

        elif current_step == 2:
            # Step 3: 核卷积 → R_k（显示前 6 个方向的影响场）
            cols = min(6, DIRECTION_BINS)
            rows_c = int(np.ceil(DIRECTION_BINS / cols))
            thumb_sz = p_size // max(cols, rows_c) - 8
            for k in range(DIRECTION_BINS):
                col = k % cols
                row = k // cols
                tx = col * (thumb_sz + 6) + 8
                ty = row * (thumb_sz + 24) + 12
                # 渲染 R_k 热力图
                Rk = R[k]
                if Rk.max() > 0:
                    Rn = (Rk / Rk.max() * 255).astype(np.uint8)
                    Rc = cv2.applyColorMap(Rn, cv2.COLORMAP_VIRIDIS)
                else:
                    Rc = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
                R_rs = cv2.resize(Rc, (thumb_sz, thumb_sz), interpolation=cv2.INTER_LINEAR)
                cv2.rectangle(R_rs, (0, 0), (thumb_sz-1, thumb_sz-1), (60, 60, 60), 1)
                t_end_y = min(ty + thumb_sz, p_size)
                t_end_x = min(tx + thumb_sz, p_size)
                R_crop = R_rs[:t_end_y-ty, :t_end_x-tx]
                if R_crop.shape[0] > 0 and R_crop.shape[1] > 0:
                    main_panel[ty:t_end_y, tx:t_end_x] = R_crop
                cv2.putText(main_panel, f"R_{k} ({BIN_NAMES[k]})", (tx, ty+thumb_sz+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)

        elif current_step == 3:
            # Step 4: 冲突场 C(x,y)
            C_disp = (C / max(C.max(), 1e-8) * 255).astype(np.uint8)
            C_color = cv2.applyColorMap(C_disp, cv2.COLORMAP_INFERNO)
            C_rs = cv2.resize(C_color, (p_size, p_size), interpolation=cv2.INTER_LINEAR)
            alpha = 0.55
            cv2.addWeighted(C_rs, alpha, main_panel, 1-alpha, 0, dst=main_panel)

            # 车辆位置标记 + 归因分数
            for i, v in enumerate(vehicles):
                px = int(v['cx'] / WORLD_SIZE_M * p_size)
                py = int(v['cy'] / WORLD_SIZE_M * p_size)
                cv2.drawMarker(main_panel, (px, py), (50, 200, 255), cv2.MARKER_CROSS, 10, 1)
                if i < len(influences):
                    pct = influences[i] / max(max(influences, default=0), 1e-8) * 100
                    cv2.putText(main_panel, f"#{i} {pct:.0f}%", (px+8, py-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)

        elif current_step == 4:
            # Step 5: 归因 → 冲突对分解显示
            cols = 4
            thumb_sz = p_size // cols - 10
            for ki, (k1, k2) in enumerate(conflict_pairs[:16]):
                col = ki % cols
                row = ki // cols
                tx = col * (thumb_sz + 6) + 8
                ty = row * (thumb_sz + 28) + 12
                # 这对冲突的影响
                C_pair = pair_results.get((k1, k2), np.zeros((GRID_SIZE, GRID_SIZE)))
                if C_pair.max() > 0:
                    Cn = (C_pair / C_pair.max() * 255).astype(np.uint8)
                    Cc = cv2.applyColorMap(Cn, cv2.COLORMAP_INFERNO)
                else:
                    Cc = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
                Cr = cv2.resize(Cc, (thumb_sz, thumb_sz), interpolation=cv2.INTER_LINEAR)
                cv2.rectangle(Cr, (0, 0), (thumb_sz-1, thumb_sz-1), (60, 60, 60), 1)
                # 裁剪到面板边界内
                t_end_y = min(ty + thumb_sz, p_size)
                t_end_x = min(tx + thumb_sz, p_size)
                Cr_crop = Cr[:t_end_y-ty, :t_end_x-tx]
                if Cr_crop.shape[0] > 0 and Cr_crop.shape[1] > 0:
                    main_panel[ty:t_end_y, tx:t_end_x] = Cr_crop
                name1, name2 = BIN_NAMES[k1].split('/')[0], BIN_NAMES[k2].split('/')[0]
                cv2.putText(main_panel, f"{name1}↔{name2}", (tx, ty+thumb_sz+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, DIM, 1)

        elif current_step == 5:
            # Step 6: 消融 — 逐级移除 Top 车辆
            if influences:
                ranked = sorted(enumerate(influences), key=lambda x: x[1], reverse=True)
                n_show = min(len(ranked), 5)
                # 显示 Top-K 移除效果
                for ki in range(n_show):
                    k_idx = ki
                    ty = 12 + ki * 50
                    _, val = ranked[ki]
                    pct = val / max(max(influences), 1e-8) * 100
                    cv2.putText(main_panel, f"K={ki+1}  #{vehicles[ranked[ki][0]]['track_id']}  inf={pct:.0f}%",
                                (12, ty+24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
                    # 进度条
                    bar_w = int(pct / 30.0 * (p_size - 40))
                    cv2.rectangle(main_panel, (12, ty+30), (12+bar_w, ty+44), (50, 200, 255), -1)
                    cv2.rectangle(main_panel, (12, ty+30), (p_size-28, ty+44), (60, 60, 60), 1)

        elif current_step == 6:
            # Step 7: 根因传播 — 水滴算法
            for i, v in enumerate(vehicles):
                tid = v.get('track_id', i)
                px = int(v['cx'] / WORLD_SIZE_M * p_size)
                py = int(v['cy'] / WORLD_SIZE_M * p_size)
                inf_pct = influences[i] / max_inf * 100 if i < len(influences) else 0
                rc_pct = root_cause_pct[i] if i < len(root_cause_pct) else 0

                # 颜色：橙色=拥堵参与  红色=因果根因  灰色=普通
                if tid in root_cause_tids:
                    color = (50, 50, 255)
                    label = f"#{tid} ROOT CAUSE {rc_pct:.0f}%"
                elif tid in highlighted_tids:
                    color = (50, 140, 230)
                    label = f"#{tid} CONGESTION {inf_pct:.0f}%"
                else:
                    color = (100, 100, 100)
                    label = f"#{tid} inf={inf_pct:.0f}%"

                _draw_vehicle(main_panel, v, px, py, color, p_size)
                if show_labels:
                    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    lx, ly = px - tw//2, py - 22
                    if lx < 0: lx = 2
                    if ly < 10: ly = py + 10
                    cv2.rectangle(main_panel, (lx-2, ly-14), (lx+tw+2, ly+2), (20, 20, 24), -1)
                    cv2.putText(main_panel, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            # 排名面板（右上角）
            if influences:
                ranked_rc = sorted(enumerate(root_cause_pct), key=lambda x: x[1], reverse=True)
                n_show = min(len(ranked_rc), 8)
                px0, py0 = p_size - 210, 40
                cv2.rectangle(main_panel, (px0-6, py0-14), (px0+204, py0+n_show*22+10), (30, 30, 36), -1)
                cv2.rectangle(main_panel, (px0-6, py0-14), (px0+204, py0+n_show*22+10), (50, 50, 55), 1)
                cv2.putText(main_panel, "  Vehicle   Inf%   Cause%", (px0, py0), cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1)
                for ri in range(n_show):
                    idx, rc_val = ranked_rc[ri]
                    tid = vehicles[idx].get('track_id', idx)
                    inf_pct = influences[idx] / max_inf * 100 if idx < len(influences) and max_inf > 0 else 0
                    rc_pct = rc_val
                    iy = py0 + 18 + ri * 22
                    # 颜色
                    if idx < len(vehicles) and vehicles[idx].get('track_id', idx) in root_cause_tids:
                        row_color = (50, 50, 255)
                    elif inf_pct > 15.0:
                        row_color = (50, 140, 230)
                    else:
                        row_color = DIM
                    cv2.putText(main_panel, f"  #{tid:<4d}  {inf_pct:>4.0f}%  {rc_pct:>4.0f}%",
                                (px0, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, row_color, 1)

            cv2.putText(main_panel, "Orange = High Congestion | Red = Root Cause (water accumulates at source)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1)

        # 车辆框+方向箭头（所有步骤都画，Step 7 已单独处理框+箭头+标签）
        if current_step != 6:
            for i, v in enumerate(vehicles):
                px = int(v['cx'] / WORLD_SIZE_M * p_size)
                py = int(v['cy'] / WORLD_SIZE_M * p_size)
                color = (50, 255, 100) if i == sel_idx else (180, 180, 180)
                _draw_vehicle(main_panel, v, px, py, color, p_size)
                if i == sel_idx:
                    cv2.putText(main_panel, f"#{v.get('track_id', i)}", (px+8, py-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 255, 100), 1)

        cv2.rectangle(main_panel, (0, 0), (p_size-1, p_size-1), (50, 50, 55), 1)
        canvas[p_y0:p_y0+p_size, p_x0:p_x0+p_size] = main_panel

        # ── 标题 ──
        cv2.putText(canvas, f"Step {current_step+1}/7:  {STEP_NAMES[current_step]}",
                    (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 2)

        # ── 右侧车辆面板 ──
        rx0, ry0 = p_x0 + p_size + 20, 50
        rw = WINDOW_W - rx0 - 20

        cv2.putText(canvas, "Vehicles", (rx0, ry0+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 2)
        for i, v in enumerate(vehicles):
            vy = ry0 + 30 + i * 60
            sel = " <<" if i == sel_idx else ""
            cv2.putText(canvas, f"#{v['track_id']} {v['label']}{sel}", (rx0+6, vy+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT, 1)
            cv2.putText(canvas, f"h={v['heading_deg']:.0f}°  pos=({v['cx']:.1f},{v['cy']:.1f})",
                        (rx0+6, vy+28), cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)
            bin_k = int(v['heading_deg'] % 360 / bin_size) % DIRECTION_BINS
            if i < len(influences):
                inf_pct = influences[i] / max(max(influences, default=0), 1e-8) * 100
                cv2.putText(canvas, f"inf={inf_pct:.0f}%  bin={bin_k}({BIN_NAMES[bin_k]})",
                            (rx0+6, vy+44), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 180, 200), 1)

        # ── 底部说明 ──
        descs = [
            "Each vehicle occupies one grid cell (Occupancy Field O)",
            f"Directions split into {DIRECTION_BINS} bins at {bin_size:.0f}deg intervals",
            "Anisotropic Gaussian convolution -> 12 influence fields R_k",
            "Conflict field C = sum(R_a x R_b) over 24 conflict pairs",
            "Conflict pairs: opposite 6 + orthogonal 6 + same-direction 12",
            "Ablation: remove Top-K vehicles -> conflict field decay",
            "Root Cause: water drops propagate upstream via adjacency matrix -> source is RED",
        ]
        if current_step < len(descs):
            cv2.putText(canvas, descs[current_step],
                        (20, WINDOW_H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1)

        # ── 颜色图例 ──
        leg_y = WINDOW_H - 60
        for i in range(10):
            t = i / 9.0
            r = int(200 * t)
            g = int(50 + 150 * (1 - abs(t - 0.5) * 2))
            b = int(30 + 100 * (1 - t))
            cv2.rectangle(canvas, (rx0 + i*24, leg_y), (rx0 + i*24 + 22, leg_y+18), (b, g, r), -1)
        cv2.putText(canvas, "Low", (rx0, leg_y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.28, DIM, 1)
        cv2.putText(canvas, "High", (rx0+216, leg_y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.28, DIM, 1)

        try:
            cv2.imshow("Convolution Pipeline Demo", canvas)
        except Exception as e:
            traceback.print_exc()
            print(f"[RENDER ERROR in step {current_step+1}] {e}")
            # 不要 break，继续循环
        key = cv2.waitKey(100) & 0xFF

        if key == 27:
            break
        elif key in (49, 50, 51, 52, 53, 54, 55):  # '1'~'7'
            current_step = key - 49
            if current_step >= len(STEP_NAMES):
                current_step = len(STEP_NAMES) - 1
            print(f"  Step {current_step+1}")
        elif key in (ord('t'), ord('T')):
            show_labels = not show_labels
            print(f"  Vehicle labels: {'ON' if show_labels else 'OFF'}")
        elif key in (ord('q'), ord('Q')):
            import random
            vehicles.append({
                'cx': random.uniform(5, WORLD_SIZE_M-5),
                'cy': random.uniform(5, WORLD_SIZE_M-5),
                'speed_mps': random.uniform(0, 5),
                'heading_deg': random.uniform(0, 360),
                'label': 'car',
                'track_id': max(v['track_id'] for v in vehicles) + 1 if vehicles else 0,
            })
            print(f"  Added vehicle #{len(vehicles)-1}")
        elif key == ord('e') and len(vehicles) > 1:
            vehicles.pop()
            print(f"  Removed vehicle, now {len(vehicles)}")
        elif key in (ord('s'), ord('S')):
            # 保存布局
            save_dir = Path(__file__).resolve().parent / "layouts"
            save_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"layout_{ts}.json"
            data = {
                'vehicles': [{
                    'track_id': v['track_id'],
                    'cx': round(v['cx'], 2),
                    'cy': round(v['cy'], 2),
                    'heading_deg': round(v['heading_deg'] % 360, 1),
                    'speed_mps': round(v.get('speed_mps', 3.0), 2),
                    'label': v.get('label', 'car'),
                } for v in vehicles],
            }
            save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"  📂Saved {len(vehicles)} vehicles to {save_path.name}")
        elif key in (ord('l'), ord('L')):
            try:
                save_dir = Path(__file__).resolve().parent / "layouts"
                if not save_dir.exists():
                    print("  No saved layouts found.")
                    continue
                files = sorted(save_dir.glob("*.json"))
                if not files:
                    print("  No saved layouts found.")
                    continue
                latest = files[-1]
                data = json.loads(latest.read_text(encoding='utf-8'))
                if 'vehicles' in data:
                    vehicles.clear()
                    for vd in data['vehicles']:
                        vehicles.append({
                            'track_id': vd.get('track_id', len(vehicles)),
                            'cx': vd['cx'],
                            'cy': vd['cy'],
                            'heading_deg': vd['heading_deg'],
                            'speed_mps': vd.get('speed_mps', 3.0),
                            'label': 'car',
                        })
                    sel_idx = 0
                    print(f"  📂Loaded {len(vehicles)} vehicles from {latest.name}")
            except Exception as e:
                print(f"  📂Load failed: {e}")

    cv2.destroyAllWindows()


def _draw_vehicle(panel, v, px, py, color, p_size):
    """画车辆矩形框+方向箭头（与主项目尺寸一致）"""
    vlen = {'car':4.0,'truck':10.0,'van':4.2,'bus':9.0,'motorcycle':2.1,'bicycle':1.6}
    vwid = {'car':1.6,'truck':2.6,'van':1.6,'bus':2.2,'motorcycle':0.8,'bicycle':0.6}
    lbl = v.get('label', 'car')
    ppm = p_size / WORLD_SIZE_M
    half_l = vlen.get(lbl, 4.0) * 0.65 / 2 * ppm
    half_w = vwid.get(lbl, 1.6) * 0.65 / 2 * ppm

    rad = np.radians(v['heading_deg'])
    cos_h, sin_h = cos(rad), sin(rad)
    corners = [(-half_l, -half_w), (half_l, -half_w),
               (half_l, half_w), (-half_l, half_w)]
    rotated = [(int(px + c[0]*cos_h - c[1]*sin_h),
                int(py - c[0]*sin_h - c[1]*cos_h)) for c in corners]
    pts = np.array(rotated, dtype=np.int32)

    cv2.fillPoly(panel, [pts], color)
    cv2.polylines(panel, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    long_m = max(vlen.get(lbl, 4.0), vwid.get(lbl, 1.6))
    arrow_len = int(long_m * 0.65 * 1.2 * ppm)
    hx = int(px + arrow_len * cos_h)
    hy = int(py - arrow_len * sin_h)
    cv2.arrowedLine(panel, (px, py), (hx, hy), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.15)


if __name__ == "__main__":
    main()
