"""
Grid & Direction Encoding Demo — 车辆散布到栅格 + 方向分解可视化

展示：
  1. 车辆在 G×G 栅格上的占用场 O(x,y)
  2. 方向场 Θ(x,y) — 每辆车的朝向
  3. 软分配 → 12 个方向箱 O_k
  4. 速度场 V(x,y)

操作: 拖拽车辆移动 | 滚轮调朝向 | Q/E 增删 | 空格键切换显示模式

python demo_test/grid_direction_demo.py
"""

import cv2
import numpy as np
from math import exp, cos, sin, pi
import random

W, H = 1300, 850
GRID = 32           # G×G 网格
CELL = 16           # 每格像素
PANEL_SZ = GRID * CELL  # 512
C_BG = (18, 18, 22)
C_SURFACE = (26, 28, 34)
C_TEXT = (200, 200, 200)
C_DIM = (100, 100, 100)
C_GRID = (42, 44, 50)
DIR_COLORS = [
    (200, 120, 60), (180, 200, 80),  (80, 200, 140), (60, 200, 220),
    (80, 140, 220), (160, 100, 220), (220, 200, 40), (220, 140, 120),
    (160, 100, 200), (100, 160, 200), (200, 160, 80), (100, 200, 140),
]
BIN_NAMES = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW']
BINS = 12
BIN_DEG = 360.0 / BINS


def make_base_grid():
    """创建深色网格底图"""
    img = np.full((PANEL_SZ, PANEL_SZ, 3), C_BG, dtype=np.uint8)
    for i in range(GRID + 1):
        x = i * CELL
        cv2.line(img, (x, 0), (x, PANEL_SZ), C_GRID, 1)
        cv2.line(img, (0, x), (PANEL_SZ, x), C_GRID, 1)
    # 坐标标注
    for i in range(0, GRID, 4):
        cv2.putText(img, f"{i}", (i*CELL+2, PANEL_SZ-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_DIM, 1)
    return img


def draw_vehicle(panel, gx, gy, heading, color, size=6, label=""):
    """在网格上画车辆（带方向箭头）"""
    cx, cy = gx * CELL + CELL//2, gy * CELL + CELL//2
    rad = np.radians(heading)
    # 车身
    cv2.circle(panel, (cx, cy), size, color, -1, cv2.LINE_AA)
    cv2.circle(panel, (cx, cy), size, (255, 255, 255), 1, cv2.LINE_AA)
    # 箭头
    al = size * 2
    hx = int(cx + al * cos(rad))
    hy = int(cy - al * sin(rad))
    cv2.arrowedLine(panel, (cx, cy), (hx, hy), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
    if label:
        cv2.putText(panel, label, (cx-10, cy+size+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_TEXT, 1)


class GridDemo:
    def __init__(self):
        self.vehicles = [
            {'gx': 6, 'gy': 8,  'heading': 0,   'v': 3.0, 'label': 'car'},
            {'gx': 12,'gy': 12, 'heading': 45,  'v': 2.5, 'label': 'truck'},
            {'gx': 10,'gy': 20, 'heading': 90,  'v': 4.0, 'label': 'car'},
            {'gx': 20,'gy': 10, 'heading': 180, 'v': 2.0, 'label': 'van'},
            {'gx': 22,'gy': 20, 'heading': 270, 'v': 1.5, 'label': 'car'},
            {'gx': 16,'gy': 16, 'heading': 135, 'v': 3.0, 'label': 'car'},
        ]
        self.drag_idx = -1
        self.mode = 0       # 0=O, 1=Θ, 2=O_k (12 panels), 3=V
        self.mode_names = [
            "Occupancy Field O(x,y)",
            "Direction Field [Theta](x,y)",
            "Direction Binning O_k (soft assignment)",
            "Speed Field V(x,y)"
        ]

        cv2.namedWindow("Grid & Direction Encoding Demo")
        cv2.resizeWindow("Grid & Direction Encoding Demo", W, H)
        cv2.setMouseCallback("Grid & Direction Encoding Demo", self._mouse)

    def _mouse(self, event, x, y, flags, param):
        # 左面板区域
        mx0, my0 = 30, 50
        in_left = mx0 <= x <= mx0 + PANEL_SZ and my0 <= y <= my0 + PANEL_SZ
        gx = (x - mx0) // CELL if in_left else -1
        gy = (y - my0) // CELL if in_left else -1
        in_grid = 0 <= gx < GRID and 0 <= gy < GRID

        if event == cv2.EVENT_LBUTTONDOWN and in_grid:
            best, bd = -1, 3
            for i, v in enumerate(self.vehicles):
                d = np.hypot(gx - v['gx'], gy - v['gy'])
                if d < bd: bd, best = d, i
            if best >= 0:
                self.drag_idx = best
            else:
                # 点击空白处新增车辆
                self.vehicles.append({'gx': gx, 'gy': gy, 'heading': random.randint(0, 359),
                                      'v': random.uniform(0.5, 5.0), 'label': 'car'})
                self.drag_idx = len(self.vehicles) - 1

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx >= 0 and in_grid:
            self.vehicles[self.drag_idx]['gx'] = max(0, min(GRID-1, gx))
            self.vehicles[self.drag_idx]['gy'] = max(0, min(GRID-1, gy))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = -1

    def run(self):
        print("=== Grid & Direction Encoding Demo ===")
        print("  Drag: move vehicle  |  Scroll: change heading (hover over vehicle)")
        print("  Q: add vehicle  |  E: remove last  |  SPACE: switch mode")
        print("  Modes: O -> Theta -> O_k -> V")
        print()

        while True:
            canvas = np.full((H, W, 3), C_BG, dtype=np.uint8)

            # ── 左面板: 主网格 ──
            panel = make_base_grid()
            mx0, my0 = 30, 50

            # 计算各场
            O = np.zeros((GRID, GRID), dtype=np.float32)
            Theta = np.zeros((GRID, GRID), dtype=np.float32)
            V = np.zeros((GRID, GRID), dtype=np.float32)
            cos_s = np.zeros((GRID, GRID), dtype=np.float32)
            sin_s = np.zeros((GRID, GRID), dtype=np.float32)

            for v in self.vehicles:
                gx, gy = int(v['gx']), int(v['gy'])
                if 0 <= gx < GRID and 0 <= gy < GRID:
                    O[gy, gx] += 1.0
                    V[gy, gx] += v['v']
                    rad = np.radians(v['heading'])
                    cos_s[gy, gx] += cos(rad)
                    sin_s[gy, gx] += sin(rad)

            valid = O > 0
            Theta[valid] = np.arctan2(sin_s[valid], cos_s[valid])
            V[valid] /= O[valid]

            # 方向分解（软分配）
            layers = []
            for k in range(BINS):
                center = k * BIN_DEG
                deg = np.degrees(Theta)
                diff = np.abs(deg - center)
                diff = np.minimum(diff, 360.0 - diff)
                weight = np.exp(-0.5 * (diff / (BIN_DEG/3)) ** 2)
                layers.append(weight * valid.astype(np.float32))

            mode = self.mode % 4

            # 渲染当前模式
            if mode == 0:
                # O: 占用场
                O_disp = cv2.resize(O, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_NEAREST)
                O_color = cv2.applyColorMap((O_disp * 255 / max(O.max(), 0.01)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                alpha = 0.5
                mask = O_disp > 0
                panel[mask] = cv2.addWeighted(O_color[mask], alpha, panel[mask], 1-alpha, 0)

            elif mode == 1:
                # Θ: 方向场（每个格子的方向用色相表示）
                theta_disp = (np.degrees(Theta) % 360 / 360 * 180).astype(np.uint8)
                theta_color = cv2.applyColorMap(theta_disp, cv2.COLORMAP_HSV)
                for gy in range(GRID):
                    for gx in range(GRID):
                        if valid[gy, gx]:
                            x0, y0 = gx*CELL, gy*CELL
                            panel[y0:y0+CELL, x0:x0+CELL] = cv2.addWeighted(
                                theta_color[gy:gy+1, gx:gx+1], 0.6,
                                panel[y0:y0+CELL, x0:x0+CELL], 0.4, 0)

            elif mode == 2:
                # O_k: 12 个方向箱（在右侧显示）
                pass  # 在右侧渲染

            elif mode == 3:
                # V: 速度场
                V_disp = V.copy()
                V_max = max(V.max(), 0.01)
                V_u8 = (V_disp / V_max * 255).astype(np.uint8)
                V_color = cv2.applyColorMap(V_u8, cv2.COLORMAP_PLASMA)
                valid_mask = valid.astype(np.uint8) * 255
                V_color_resized = cv2.resize(V_color, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_NEAREST)
                valid_resized = cv2.resize(valid_mask, (PANEL_SZ, PANEL_SZ), interpolation=cv2.INTER_NEAREST) > 0
                panel[valid_resized] = cv2.addWeighted(V_color_resized[valid_resized], 0.5,
                                                       panel[valid_resized], 0.5, 0)

            # 画车辆
            for i, v in enumerate(self.vehicles):
                gx, gy = int(v['gx']), int(v['gy'])
                if not (0 <= gx < GRID and 0 <= gy < GRID):
                    continue
                color = (80, 200, 255) if i != self.drag_idx else (60, 255, 100)
                draw_vehicle(panel, gx, gy, v['heading'], color, size=7,
                            label=f"#{i} {v['heading']}deg")

            # 模式标题
            cv2.putText(panel, self.mode_names[mode], (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)
            cv2.putText(panel, f"Vehicles: {len(self.vehicles)}  Grid: {GRID}x{GRID}",
                        (8, PANEL_SZ-6), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_DIM, 1)

            canvas[my0:my0+PANEL_SZ, mx0:mx0+PANEL_SZ] = panel

            # ── 右面板: 方向箱缩略图或速度图 ──
            rx0, ry0 = mx0 + PANEL_SZ + 20, 50
            rw = W - rx0 - 20

            if mode == 2:
                # 12 个方向箱的 O_k
                n_cols = 3
                n_rows = 4
                thumb_sz = min((rw - 20) // n_cols, (PANEL_SZ - 40) // n_rows)
                cv2.putText(canvas, "Direction Decomposition: O_k (soft assignment)", (rx0, ry0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT, 1)
                for k in range(BINS):
                    col = k % n_cols
                    row = k // n_cols
                    tx = rx0 + col * (thumb_sz + 8) + 10
                    ty = ry0 + row * (thumb_sz + 42) + 22

                    Ok = layers[k]
                    if Ok.max() > 0:
                        Ok_u8 = (Ok / Ok.max() * 255).astype(np.uint8)
                        Ok_color = cv2.applyColorMap(Ok_u8, cv2.COLORMAP_VIRIDIS)
                    else:
                        Ok_color = np.zeros((GRID, GRID, 3), dtype=np.uint8)
                    disp = cv2.resize(Ok_color, (thumb_sz, thumb_sz), interpolation=cv2.INTER_NEAREST)
                    cv2.rectangle(disp, (0, 0), (thumb_sz-1, thumb_sz-1), (60, 60, 60), 1)
                    canvas[ty:ty+thumb_sz, tx:tx+thumb_sz] = disp
                    # 方向色条 + 名称
                    color_bar = np.full((6, thumb_sz, 3), DIR_COLORS[k], dtype=np.uint8)
                    cv2.putText(color_bar, f'bin{k}:{BIN_NAMES[k]}', (2, 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                    canvas[ty+thumb_sz+2:ty+thumb_sz+8, tx:tx+thumb_sz] = color_bar

            elif mode == 3:
                # 速度场图例
                leg = np.full((200, 60, 3), C_SURFACE, dtype=np.uint8)
                for i in range(60):
                    t = i / 59.0
                    color = cv2.applyColorMap(np.array([[int(t*255)]], dtype=np.uint8), cv2.COLORMAP_PLASMA)[0, 0]
                    leg[10:190, i:i+1] = color
                canvas[ry0:ry0+200, rx0:rx0+60] = leg
                cv2.putText(canvas, f"min: 0 m/s", (rx0+65, ry0+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)
                cv2.putText(canvas, f"max: {V_max:.1f} m/s", (rx0+65, ry0+180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)
                cv2.putText(canvas, "Speed Field V(x,y)", (rx0, ry0+220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT, 1)
                # 速度值显示
                for v in self.vehicles:
                    gx, gy = int(v['gx']), int(v['gy'])
                    if 0 <= gx < GRID and 0 <= gy < GRID:
                        val = V[gy, gx]
                        cv2.putText(canvas, f"v({gx},{gy})={val:.1f}", (rx0, ry0+260+len(self.vehicles)*18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)
            else:
                # 占用场/方向场模式显示信息
                info_lines = [
                    f"Grid: {GRID}x{GRID}",
                    f"Cell size: simulated",
                    f"Vehicles: {len(self.vehicles)}",
                    f"",
                    "Soft assignment sigma:",
                    f"  sigma = bin_size/3 = {BIN_DEG/3:.1f}deg",
                    f"",
                    "12 direction bins:",
                ] + [f"  bin{k:2d}: {0+k*30:3d}deg ({BIN_NAMES[k]})" for k in range(0, 12)]
                for li, line in enumerate(info_lines[:30]):
                    cv2.putText(canvas, line, (rx0, ry0+20+li*18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                C_TEXT if ':' in line else C_DIM, 1)

            # ── 底部控制提示 ──
            cv2.putText(canvas, "SPACE: switch mode  |  Drag: move vehicle  |  Scroll over vehicle: change heading",
                        (30, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)
            cv2.putText(canvas, "Q: add vehicle  |  E: remove  |  ESC: exit",
                        (30, H-35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)

            cv2.imshow("Grid & Direction Encoding Demo", canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == ord(' '):
                self.mode = (self.mode + 1) % 4
            elif key in (ord('q'), ord('Q')):
                import random
                self.vehicles.append({'gx': random.randint(0, GRID-1), 'gy': random.randint(0, GRID-1),
                                      'heading': random.randint(0, 359), 'v': random.uniform(0.5, 5.0), 'label': 'car'})
            elif key in (ord('e'), ord('E')) and len(self.vehicles) > 1:
                self.vehicles.pop()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    GridDemo().run()
