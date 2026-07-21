"""
Anisotropic Kernel Demo — interactively explore the directional influence field.

Usage:
    python demo_test/conflict_field_demo.py

Controls:
    - Drag vehicles to move them
    - Scroll wheel: change heading of selected vehicle
    - Right-click: select vehicle
    - A/D: adjust sigma_along
    - W/S: adjust sigma_perp
    - Q: add vehicle  E: remove  S: save  L: load
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from math import exp, cos, sin, pi, atan2

# ── 默认参数 ──
GRID_SIZE = 64
WORLD_SIZE_M = 50.0
CELL_SIZE_M = WORLD_SIZE_M / GRID_SIZE

# 核参数
SIGMA_ALONG = 3.0
SIGMA_PERP = 0.6
KERNEL_HALF_LEN = 10
KERNEL_HALF_WIDTH = 6

# 颜色
BG_COLOR = (24, 24, 28)
GRID_COLOR = (42, 42, 48)
VEHICLE_COLOR = (80, 200, 255)
INFLUENCE_COLORS = [
    (40, 30, 30),   # 0% 深灰
    (60, 40, 20),   # 低
    (30, 80, 60),   # 中低
    (40, 140, 100), # 中
    (60, 180, 170), # 中高
    (50, 220, 230), # 高
]
WINDOW_W, WINDOW_H = 1200, 850


def build_kernel(heading_deg: float) -> np.ndarray:
    """构建单个方向核"""
    k_size = 2 * KERNEL_HALF_LEN + 1
    k_h = k_size
    k_w = k_size
    K = np.zeros((k_h, k_w), dtype=np.float32)
    cx, cy = KERNEL_HALF_LEN, KERNEL_HALF_LEN
    theta = heading_deg * pi / 180.0
    ux, uy = cos(theta), -sin(theta)      # uy取反（numpy行坐标y向下为正）
    nx, ny = sin(theta), cos(theta)       # 法向量
    for i in range(k_h):
        for j in range(k_w):
            dx, dy = j - cx, i - cy
            along = dx * ux + dy * uy
            perp = dx * nx + dy * ny
            if along >= 0:
                eff_sigma = SIGMA_ALONG
                eff_half = KERNEL_HALF_LEN
            else:
                eff_sigma = SIGMA_ALONG * 0.33
                eff_half = KERNEL_HALF_LEN * 0.33
            f_along = exp(-0.5 * (along / eff_sigma) ** 2)
            if abs(along) > eff_half:
                f_along = 0.0
            if along >= 0:
                fan_sigma = SIGMA_PERP * (1.0 + 0.6 * along / max(SIGMA_ALONG, 1e-6))
            else:
                fan_sigma = SIGMA_PERP
            f_perp = exp(-0.5 * (perp / max(fan_sigma, 0.1)) ** 2)
            K[i, j] = f_along * f_perp
    total = K.sum()
    if total > 0:
        K /= total
    return K


def scatter_vehicle(ox, oy, heading_deg, grid_size=GRID_SIZE):
    """单辆车散布到网格"""
    O = np.zeros((grid_size, grid_size), dtype=np.float32)
    V = np.zeros((grid_size, grid_size), dtype=np.float32)
    Theta = np.zeros((grid_size, grid_size), dtype=np.float32)
    gx = int(ox / CELL_SIZE_M)
    gy = int(oy / CELL_SIZE_M)
    if 0 <= gx < grid_size and 0 <= gy < grid_size:
        O[gy, gx] = 1.0
        V[gy, gx] = 5.0  # 模拟速度
        Theta[gy, gx] = heading_deg
    return O, V, Theta


def render_influence_field(R, canvas_x, canvas_y, cw, ch):
    """渲染单个影响场到 canvas 区域"""
    if R.max() <= 0:
        return
    R_norm = (R / R.max() * 255).astype(np.uint8)
    colored = cv2.applyColorMap(R_norm, cv2.COLORMAP_VIRIDIS)
    resized = cv2.resize(colored, (cw, ch), interpolation=cv2.INTER_LINEAR)
    canvas[canvas_y:canvas_y+ch, canvas_x:canvas_x+cw] = resized


def draw_grid(panel, grid_size, ox, oy, cell_m, panel_w, panel_h):
    """在面板上画网格 + 坐标轴"""
    # 画网格线
    step = max(grid_size // 16, 1)
    for i in range(0, grid_size, step):
        x = int(i * cell_m)
        cv2.line(panel, (x, 0), (x, panel_h), GRID_COLOR, 1)
    for i in range(0, grid_size, step):
        y = int(i * cell_m)
        cv2.line(panel, (0, y), (panel_w, y), GRID_COLOR, 1)
    # 坐标标注
    for i in range(0, grid_size, step * 4):
        cv2.putText(panel, f"{i*CELL_SIZE_M:.0f}m", (int(i*cell_m)-10, panel_h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)
        cv2.putText(panel, f"{i*CELL_SIZE_M:.0f}m", (2, int(i*cell_m)+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)


def heading_color(deg):
    """朝向角度 → BGR 颜色"""
    h = deg / 360.0
    r = int(255 * (1 - h))
    g = int(128 * (1 - abs(h - 0.5) * 2))
    b = int(255 * h)
    return (b, g, r)


def main():
    global SIGMA_ALONG, SIGMA_PERP, GRID_SIZE, KERNEL_HALF_LEN, KERNEL_HALF_WIDTH

    # 车辆状态
    vehicles = [{'x': 25.0, 'y': 25.0, 'heading': 45.0}]  # 默认一辆车
    drag_idx = -1
    drag_offset = (0, 0)

    cv2.namedWindow("Conflict Field Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Conflict Field Demo", WINDOW_W, WINDOW_H)

    # 当前选中的车辆索引（用于滚轮调朝向）
    selected_idx = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal drag_idx, drag_offset, selected_idx
        # 鼠标坐标映射到世界坐标
        panel_x0, panel_y0 = 20, 40
        panel_w = WINDOW_W // 2 - 30
        panel_h = WINDOW_H - 80
        cell_m = min(panel_w, panel_h) / GRID_SIZE
        offset_x = int(panel_x0 + (panel_w - cell_m * GRID_SIZE) // 2)
        offset_y = int(panel_y0 + (panel_h - cell_m * GRID_SIZE) // 2)

        wx = (x - offset_x) / cell_m * CELL_SIZE_M
        wy = (y - offset_y) / cell_m * CELL_SIZE_M

        if event == cv2.EVENT_LBUTTONDOWN:
            # 找最近的车
            best, best_d = -1, 50
            for i, v in enumerate(vehicles):
                d = np.hypot(wx - v['x'], wy - v['y'])
                if d < best_d:
                    best_d = d
                    best = i
            if best >= 0:
                drag_idx = best
                drag_offset = (vehicles[best]['x'] - wx, vehicles[best]['y'] - wy)

        elif event == cv2.EVENT_MOUSEMOVE and drag_idx >= 0:
            vehicles[drag_idx]['x'] = max(0, min(WORLD_SIZE_M, wx + drag_offset[0]))
            vehicles[drag_idx]['y'] = max(0, min(WORLD_SIZE_M, wy + drag_offset[1]))

        elif event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮调整朝向
            if vehicles:
                delta = -1 if flags > 0 else 1  # 上滚+5°，下滚-5°
                idx = selected_idx if selected_idx < len(vehicles) else 0
                vehicles[idx]['heading'] = (vehicles[idx]['heading'] + delta * 5) % 360

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键选择车辆
            best, best_d = -1, 50
            for i, v in enumerate(vehicles):
                d = np.hypot(wx - v['x'], wy - v['y'])
                if d < best_d:
                    best_d = d
                    best = i
            if best >= 0:
                selected_idx = best
                print(f"  Selected vehicle #{best} (heading={vehicles[best]['heading']:.0f}°)")

        elif event == cv2.EVENT_LBUTTONUP:
            drag_idx = -1

    cv2.setMouseCallback("Conflict Field Demo", mouse_cb)

    print("=== Conflict Field Demo ===")
    print("Drag: move vehicle | Scroll: change heading")
    print("Right-click: select | A/D: sigma_along | W/S: sigma_perp")
    print("Q: add vehicle  E: remove  S: save  L: load  ESC: exit")
    print()

    while True:
        canvas = np.full((WINDOW_H, WINDOW_W, 3), BG_COLOR, dtype=np.uint8)

        # ── 标题 ──
        cv2.putText(canvas, "Conflict Field Demo — Direction-Field Convolution",
                    (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # ── 主面板：占用场 + 影响场（左半） ──
        panel_x0, panel_y0 = 20, 40
        panel_w = WINDOW_W // 2 - 30
        panel_h = WINDOW_H - 80
        cell_m = min(panel_w, panel_h) / GRID_SIZE
        offset_x = int(panel_x0 + (panel_w - cell_m * GRID_SIZE) // 2)
        offset_y = int(panel_y0 + (panel_h - cell_m * GRID_SIZE) // 2)

        # 底图
        main_panel = np.full((panel_h, panel_w, 3), BG_COLOR, dtype=np.uint8)
        draw_grid(main_panel, GRID_SIZE, 0, 0, cell_m, panel_w, panel_h)

        if vehicles:
            # 散布和卷积
            O = None
            for v in vehicles:
                Ov, _, _ = scatter_vehicle(v['x'], v['y'], v['heading'])
                if O is None:
                    O = Ov.copy()
                else:
                    O += Ov

            # 每个方向的核 → 影响场
            bins = 12
            bin_size = 360.0 / bins
            R_total = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
            for k in range(bins):
                heading_k = k * bin_size
                K = build_kernel(heading_k)
                # 该方向箱的占用（软分配）
                O_k = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
                for v in vehicles:
                    bin_idx = int(v['heading'] % 360 / bin_size) % bins
                    # 软分配：高斯权重到相邻箱
                    for bk in [bin_idx, (bin_idx + 1) % bins, (bin_idx - 1) % bins]:
                        diff = abs(heading_k - v['heading'])
                        if diff > 180: diff = 360 - diff
                        if diff > 30: continue
                        w = exp(-0.5 * (diff / 10.0) ** 2)
                        gx = int(v['x'] / CELL_SIZE_M)
                        gy = int(v['y'] / CELL_SIZE_M)
                        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                            O_k[gy, gx] += w
                # 卷积（flip 抵消 cv2.filter2D 翻转，与 core/conflict.py 一致）
                K = cv2.flip(K, -1)
                k_h, k_w = K.shape
                pad = (k_h // 2, k_w // 2)
                O_pad = np.pad(O_k, ((pad[0], pad[0]), (pad[1], pad[1])), mode='constant')
                R_k = np.zeros_like(O_k)
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        R_k[i, j] = (O_pad[i:i+k_h, j:j+k_w] * K).sum()
                R_total += R_k

            # 渲染影响场
            if R_total.max() > 0:
                R_norm = (R_total / R_total.max() * 255).astype(np.uint8)
                heat = cv2.applyColorMap(R_norm, cv2.COLORMAP_VIRIDIS)
                heat_resized = cv2.resize(heat, (int(cell_m*GRID_SIZE), int(cell_m*GRID_SIZE)),
                                          interpolation=cv2.INTER_LINEAR)
                # 半透明叠加
                h_sz = int(cell_m * GRID_SIZE)
                ry0 = int(offset_y - panel_y0)
                rx0 = int(offset_x - panel_x0)
                roi = main_panel[ry0:ry0+h_sz, rx0:rx0+h_sz]
                alpha = 0.55
                cv2.addWeighted(heat_resized, alpha, roi, 1-alpha, 0, dst=roi)
                draw_grid(main_panel, GRID_SIZE, 0, 0, cell_m, panel_w, panel_h)

            # 画车辆
            for i, v in enumerate(vehicles):
                px = int(offset_x + v['x'] / CELL_SIZE_M * cell_m)
                py = int(offset_y + v['y'] / CELL_SIZE_M * cell_m)
                # 车辆矩形（与主项目一致: 车 4m×1.6m, 视觉缩放 0.65）
                ppm_disp = panel_w / WORLD_SIZE_M
                vlen = {'car':4.0,'truck':10.0,'van':4.2,'bus':9.0,'motorcycle':2.1,'bicycle':1.6}
                vwid = {'car':1.6,'truck':2.6,'van':1.6,'bus':2.2,'motorcycle':0.8,'bicycle':0.6}
                lbl = v.get('label', 'car')
                half_l = vlen.get(lbl, 4.0) * 0.65 / 2 * ppm_disp
                half_w = vwid.get(lbl, 1.6) * 0.65 / 2 * ppm_disp
                rad = np.radians(v['heading'])
                cos_h, sin_h = cos(rad), sin(rad)
                corners = [(-half_l, -half_w), (half_l, -half_w),
                           (half_l, half_w), (-half_l, half_w)]
                rotated = [(int(px + c[0]*cos_h - c[1]*sin_h),
                            int(py - c[0]*sin_h - c[1]*cos_h)) for c in corners]
                pts = np.array(rotated, dtype=np.int32)
                color = VEHICLE_COLOR if i != 0 else (80, 200, 255)
                cv2.fillPoly(main_panel, [pts], color)
                cv2.polylines(main_panel, [pts], True, (255, 255, 255), 1)
                # 方向线（与主项目一致）
                long_m = max(vlen.get(lbl, 4.0), vwid.get(lbl, 1.6))
                arrow_len = int(long_m * 0.65 * 1.2 * ppm_disp)
                hx = int(px + arrow_len * cos_h)
                hy = int(py - arrow_len * sin_h)
                cv2.arrowedLine(main_panel, (px, py), (hx, hy), (255, 255, 255), 1, tipLength=0.15)
                # 选中标记
                if i == selected_idx:
                    cv2.circle(main_panel, (px, py), int(cell_m * 0.8), (50, 255, 50), 2)
                # 标签
                sel = " <<" if i == selected_idx else ""
                label = f"#{i} {v['heading']:.0f}°{sel}"
                cv2.putText(main_panel, label, (px+5, py-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        # 面板边框
        cv2.rectangle(main_panel, (0, 0), (panel_w-1, panel_h-1), (80, 80, 80), 1)
        canvas[panel_y0:panel_y0+panel_h, panel_x0:panel_x0+panel_w] = main_panel

        # ── 右侧面板：参数 + 核形状 + 说明 ──
        rx0 = WINDOW_W // 2 + 10
        ry0 = 40
        rw = WINDOW_W // 2 - 30
        rh = WINDOW_H - 80

        # 参数面板
        py0 = ry0
        cv2.putText(canvas, "Parameters", (rx0, py0+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        params = [
            f"Grid: {GRID_SIZE}x{GRID_SIZE}  ({CELL_SIZE_M:.1f}m/cell)",
            f"sigma_along: {SIGMA_ALONG:.1f} cells  ({SIGMA_ALONG*CELL_SIZE_M:.1f}m)",
            f"sigma_perp:  {SIGMA_PERP:.1f} cells  ({SIGMA_PERP*CELL_SIZE_M:.1f}m)",
            f"Kernel: {2*KERNEL_HALF_LEN+1}x{2*KERNEL_HALF_LEN+1} (square)",
            f"Search dist: {KERNEL_HALF_LEN*CELL_SIZE_M:.1f}m forward",
            f"Vehicles: {len(vehicles)}",
        ]
        for i, p in enumerate(params):
            cv2.putText(canvas, p, (rx0+10, py0+42+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # 核形状可视化
        ky0 = py0 + 42 + len(params) * 22 + 20
        cv2.putText(canvas, "Anisotropic Kernel (12 directions)", (rx0, ky0+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        # 画 12 个方向核缩略图 (2行×6列) + 方向箭头
        k_cols = 6
        k_avail_w = (rw - 30) // k_cols
        k_sz = min(k_avail_w - 8, 80)
        # 生成核的白色边框透明底显示
        for k in range(12):
            h_k = k * 30.0
            K = build_kernel(h_k)
            # 核 → 彩色显示（保持长宽比，放入正方形底板）
            kh, kw = K.shape  # (9, 31)
            K_blob = np.zeros((k_sz, k_sz, 3), dtype=np.uint8)
            if K.max() > 0:
                K_disp = (K / K.max() * 255).astype(np.uint8)
                K_color = cv2.applyColorMap(K_disp, cv2.COLORMAP_VIRIDIS)
                # 保持长宽比缩放到正方形
                scale_k = min((k_sz-4) / kw, (k_sz-4) / kh)
                new_w, new_h = int(kw * scale_k), int(kh * scale_k)
                K_small = cv2.resize(K_color, (new_w, new_h), interpolation=cv2.INTER_AREA)
                # 居中放置
                x_off = (k_sz - new_w) // 2
                y_off = (k_sz - new_h) // 2
                K_blob[y_off:y_off+new_h, x_off:x_off+new_w] = K_small
            # 加边框
            cv2.rectangle(K_blob, (0, 0), (k_sz-1, k_sz-1), (80, 80, 80), 1)
            # 画方向箭头
            cx_k, cy_k = k_sz // 2, k_sz // 2
            arr_len = k_sz // 3
            rad_k = np.radians(h_k)
            ax = int(cx_k + arr_len * np.cos(rad_k))
            ay = int(cy_k - arr_len * np.sin(rad_k))
            cv2.arrowedLine(K_blob, (cx_k, cy_k), (ax, ay), (255, 255, 255), 1, tipLength=0.25)
            # 放到画布
            kx = rx0 + 6 + (k % k_cols) * (k_sz + 4)
            ky = ky0 + 30 + (k // k_cols) * (k_sz + 26)
            canvas[ky:ky+k_sz, kx:kx+k_sz] = K_blob
            # 方向标注
            label = f"{h_k:.0f}°"
            cv2.putText(canvas, label, (kx+2, ky+k_sz+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        # 当前核形状放大显示
        ay0 = ky0 + 30 + 2 * (k_sz + 30) + 20
        cv2.putText(canvas, "Current Vehicle Kernel", (rx0, ay0+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        if vehicles:
            v = vehicles[0] if drag_idx < 0 else vehicles[0]
            K = build_kernel(v['heading'])
            if K.max() > 0:
                K_disp = (K / K.max() * 255).astype(np.uint8)
            else:
                K_disp = np.zeros_like(K, dtype=np.uint8)
            K_color = cv2.applyColorMap(K_disp, cv2.COLORMAP_VIRIDIS)
            big_k = 150
            K_big = cv2.resize(K_color, (big_k, big_k), interpolation=cv2.INTER_NEAREST)
            # 加边框
            K_big = cv2.copyMakeBorder(K_big, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(80, 80, 80))
            canvas[ay0+24:ay0+24+big_k+4, rx0+10:rx0+10+big_k+4] = K_big

            # 核数值文本
            txt_x = rx0 + big_k + 30
            cv2.putText(canvas, f"Heading: {v['heading']:.0f}°", (txt_x, ay0+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            cv2.putText(canvas, f"Kernel sum: {K.sum():.4f}", (txt_x, ay0+52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(canvas, f"Eff. length (3σ): {3*SIGMA_ALONG*CELL_SIZE_M:.1f}m", (txt_x, ay0+74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(canvas, f"Eff. width (3σ): {3*SIGMA_PERP*CELL_SIZE_M:.1f}m", (txt_x, ay0+96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # 说明文字
        info_y = WINDOW_H - 120
        cv2.putText(canvas, "Controls:", (rx0, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        hints = [
            "Drag vehicle to move",
            "Scroll wheel: change selected heading",
            "Right-click: select vehicle",
            "A/D: sigma_along    W/S: sigma_perp",
            "Q: add vehicle    E: remove last",
            "ESC: exit",
        ]
        for i, h in enumerate(hints):
            cv2.putText(canvas, h, (rx0+10, info_y+20+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 120, 120), 1)

        # ── 颜色图例（左下） ──
        leg_x, leg_y = 20, WINDOW_H - 50
        for i in range(6):
            t = i / 5.0
            r = (int(50 * (1-t) + 230 * t), int(30 * (1-t) + 220 * t), int(30 * (1-t) + 50 * t))
            cv2.rectangle(canvas, (leg_x + i*30, leg_y), (leg_x + i*30 + 28, leg_y + 18), r, -1)
        cv2.putText(canvas, "Low", (leg_x, leg_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(canvas, "High", (leg_x + 150, leg_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        cv2.imshow("Conflict Field Demo", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('a'):
            SIGMA_ALONG = max(1.0, SIGMA_ALONG - 0.5)
        elif key == ord('d'):
            SIGMA_ALONG = min(20.0, SIGMA_ALONG + 0.5)
        elif key == ord('w'):
            SIGMA_PERP = max(0.5, SIGMA_PERP - 0.1)
        elif key == ord('s'):
            SIGMA_PERP = min(10.0, SIGMA_PERP + 0.1)
        elif key == ord('q'):
            # 在随机位置添加车辆
            import random
            nx = random.uniform(5, WORLD_SIZE_M - 5)
            ny = random.uniform(5, WORLD_SIZE_M - 5)
            nh = random.uniform(0, 360)
            vehicles.append({'x': nx, 'y': ny, 'heading': nh})
            print(f"  Added vehicle #{len(vehicles)-1} at ({nx:.1f}, {ny:.1f}), {nh:.0f}°")
        elif key == ord('e'):
            if len(vehicles) > 1:
                removed = vehicles.pop()
                print(f"  Removed vehicle #{len(vehicles)}")
        elif key == ord('s'):
            import json, datetime
            save_dir = Path(__file__).resolve().parent / "layouts"
            save_dir.mkdir(exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"field_layout_{ts}.json"
            data = {'vehicles': [{'cx': round(v['x'],2), 'cy': round(v['y'],2), 'heading': round(v['heading']%360,1)} for v in vehicles]}
            save_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            print(f"  Saved {len(vehicles)} vehicles to {save_path.name}")
        elif key == ord('l'):
            save_dir = Path(__file__).resolve().parent / "layouts"
            if not save_dir.exists(): continue
            files = sorted(save_dir.glob("field_layout_*.json"))
            if not files: continue
            data = json.loads(files[-1].read_text(encoding='utf-8'))
            if 'vehicles' in data:
                vehicles.clear()
                for vd in data['vehicles']:
                    vehicles.append({'x': vd['cx'], 'y': vd['cy'], 'heading': vd['heading']})
                print(f"  Loaded {len(vehicles)} vehicles from {files[-1].name}")

        # 滚轮调整航向（最后处理的车的）
        # 用特殊方式处理：实际上 OpenCV 没有直接的滚轮回调
        # 使用鼠标滚轮值：flags 包含 cv2.EVENT_FLAG_ALT 组合

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
