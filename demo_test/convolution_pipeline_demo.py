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
    "1. Scatter: Vehicles → Grid",
    "2. Direction Binning (12 bins)",
    "3. Kernel Convolution → R_k",
    "4. Conflict Field C(x,y)",
    "5. Per-Vehicle Attribution",
    "6. Ablation: Remove Top-K",
]


def main():
    grid_cfg = GridConfig(GRID_SIZE, CELL_SIZE_M, 0, 0)
    kernel_cfg = KernelConfig(15, 4, 5.0, 1.5)
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)

    # 车辆 — 默认构造一个十字交叉冲突场景
    vehicles = [
        {'cx': 28.0, 'cy': 40.0, 'speed_mps': 4.0, 'heading_deg': 0,   'label': 'car',   'track_id': 0},   # → 向东
        {'cx': 20.0, 'cy': 35.0, 'speed_mps': 3.0, 'heading_deg': 0,   'label': 'truck', 'track_id': 1},   # → 向东
        {'cx': 38.0, 'cy': 20.0, 'speed_mps': 2.0, 'heading_deg': 180,'label': 'car',   'track_id': 2},   # ← 向西
        {'cx': 32.0, 'cy': 28.0, 'speed_mps': 3.0, 'heading_deg': 90, 'label': 'van',   'track_id': 3},   # ↓ 向南(90°=South)
        {'cx': 40.0, 'cy': 42.0, 'speed_mps': 2.0, 'heading_deg': 270,'label': 'car',   'track_id': 4},   # ↑ 向北(270°=North)
    ]
    drag_idx = -1
    sel_idx = 0
    current_step = 3  # 默认显示冲突场
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
    print("Tab: switch step (1-6)")
    print("Drag: move  Right-click: select  Scroll: heading")
    print("Q: add  E: remove  S: save  L: load  ESC: exit")
    print()

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
                    intensity = min(layers_decomp[k].sum() * 80, 255)
                    # 扇形
                    for a in range(int(angle - bin_size / 2), int(angle + bin_size / 2)):
                        rad_a = np.radians(a)
                        px = int(cx_s + r * cos(rad_a))
                        py = int(cy_s - r * sin(rad_a))
                        cv2.line(main_panel, (cx_s, cy_s), (px, py), (int(intensity*0.5), int(intensity*0.7), intensity), 1)
                    px_e = int(cx_s + (r+10) * cos(rad))
                    py_e = int(cy_s - (r+10) * sin(rad))
                    cv2.arrowedLine(main_panel, (cx_s, cy_s), (px_e, py_e), (int(intensity*0.5), int(intensity*0.7), intensity), 2, tipLength=0.15)
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
                main_panel[ty:ty+thumb_sz, tx:tx+thumb_sz] = R_rs
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
                main_panel[ty:ty+thumb_sz, tx:tx+thumb_sz] = Cr
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

        # 车辆方向箭头（所有步骤都画）
        for i, v in enumerate(vehicles):
            px = int(v['cx'] / WORLD_SIZE_M * p_size)
            py = int(v['cy'] / WORLD_SIZE_M * p_size)
            rad = np.radians(v['heading_deg'])
            al = int(CELL_SIZE_M * 3 / WORLD_SIZE_M * p_size)
            hx = int(px + al * cos(rad))
            hy = int(py - al * sin(rad))
            # 选中的车用绿色标记
            color = (50, 255, 100) if i == sel_idx else (200, 200, 200)
            cv2.arrowedLine(main_panel, (px, py), (hx, hy), color, 1, tipLength=0.2)

        cv2.rectangle(main_panel, (0, 0), (p_size-1, p_size-1), (50, 50, 55), 1)
        canvas[p_y0:p_y0+p_size, p_x0:p_x0+p_size] = main_panel

        # ── 标题 ──
        cv2.putText(canvas, f"Step {current_step+1}/6:  {STEP_NAMES[current_step]}",
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
            "Conflict field C = sum(R_a x R_b) over 12 conflict pairs",
            "Per-vehicle: Influence_i = R_{ki}(P_i) x sum(R_{k'})(P_i)",
            "Ablation: remove Top-K vehicles -> conflict field decay",
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
            print(f"[RENDER ERROR] {e}")
            break
        key = cv2.waitKey(100) & 0xFF

        if key == 27:
            break
        elif key in (49, 50, 51, 52, 53, 54):  # '1'~'6'
            current_step = key - 49
            print(f"  Step {current_step+1}")
        elif key == ord('q'):
            import random
            vehicles.append({
                'cx': random.uniform(5, WORLD_SIZE_M-5),
                'cy': random.uniform(5, WORLD_SIZE_M-5),
                'speed_mps': random.uniform(0, 5),
                'heading_deg': random.uniform(0, 360),
                'label': random.choice(['car', 'truck', 'van']),
                'track_id': max(v['track_id'] for v in vehicles) + 1 if vehicles else 0,
            })
            print(f"  Added vehicle #{len(vehicles)-1}")
        elif key == ord('e') and len(vehicles) > 1:
            vehicles.pop()
            print(f"  Removed vehicle, now {len(vehicles)}")
        elif key == ord('s'):
            # 保存布局
            import json, datetime
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
            print(f"  💾 Saved {len(vehicles)} vehicles to {save_path.name}")
        elif key == ord('l'):
            # 加载布局
            save_dir = Path(__file__).resolve().parent / "layouts"
            if not save_dir.exists():
                print("  No saved layouts found.")
                continue
            files = sorted(save_dir.glob("*.json"))
            if not files:
                print("  No saved layouts found.")
                continue
            # 加载最新的
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
                        'label': vd.get('label', 'car'),
                    })
                sel_idx = 0
                print(f"  📂 Loaded {len(vehicles)} vehicles from {latest.name}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
