"""
水滴传播算法演示 — 逐帧展示水如何汇聚到拥堵源头

按 1-9 手动步进迭代 | 按 空格自动播放 | 按 R 重置

python demo_test/water_drop_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import time
from math import cos, sin, pi, exp

# ── 颜色 ──
C_BG = (18, 18, 22)
C_SURFACE = (26, 28, 34)
C_TEXT = (200, 200, 200)
C_DIM = (100, 100, 100)
C_ORANGE = (50, 140, 230)
C_RED = (50, 50, 255)
C_GREEN = (50, 200, 100)
C_BLUE = (200, 140, 70)
C_CYAN = (200, 220, 60)
C_WHITE = (255, 255, 255)

W, H = 1200, 800
# 模拟参数
ALPHA = 0.4
N_QUEUE = 8          # 队列长度
SPACING = 50         # 车距(pixels)
QUEUE_X0 = 120       # 队列起始 x
QUEUE_Y = 350        # 队列 y
CAR_W, CAR_H = 40, 20

# 9 个队列 (0=tail, N-1=head)
# 速度: 尾部最快, 头部最慢(已停)
SPEEDS = [5.0, 4.3, 3.5, 2.5, 1.5, 0.5, 0.0, 0.0]


class WaterDropDemo:
    def __init__(self):
        self.iteration = 0
        self.max_iters = 15
        self.auto_play = False
        self.last_toggle = 0

        # 车辆状态
        self.N = N_QUEUE
        self.vehicles = []
        for i in range(self.N):
            self.vehicles.append({
                'id': i,
                'x': QUEUE_X0 + i * SPACING,
                'y': QUEUE_Y,
                'speed': SPEEDS[i],
                'water': 1.0,           # 初始水滴=1
                'prev_water': 1.0,
                'is_root_cause': False,
                'color': C_WHITE,
            })

        self.A = np.zeros((self.N, self.N), dtype=np.float64)  # 邻接矩阵
        self.compute_adjacency()
        self._history = [[1.0] * self.N]  # 迭代历史

        cv2.namedWindow("Water Drop Propagation Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Water Drop Propagation Demo", W, H)

    def compute_adjacency(self):
        """构建邻接矩阵: 前车更慢且距离近 → 强连接"""
        self.A = np.zeros((self.N, self.N), dtype=np.float64)
        self.connections = []
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                # j 在 i 前方？
                if j <= i:
                    continue
                dist = abs(self.vehicles[j]['x'] - self.vehicles[i]['x'])
                if dist > 200 or dist < 5:
                    continue
                # 速度差: 前车更慢 → 水从 i 流向 j
                sd = self.vehicles[j]['speed'] - self.vehicles[i]['speed']
                vf = max(0, min(-sd / 5.0, 1.0)) + 0.05
                df = exp(-0.5 * (dist / 80) ** 2)
                w = vf * df
                if w > 0.01:
                    self.A[i, j] = w
                    self.connections.append((i, j, w))

    def propagate(self):
        """执行一轮迭代传播"""
        if self.iteration >= self.max_iters:
            return

        x = np.array([v['water'] for v in self.vehicles], dtype=np.float64)
        row_sums = self.A.sum(axis=1, keepdims=True)
        A_norm = self.A / np.maximum(row_sums, 1e-10)
        x_new = x + ALPHA * (A_norm.T @ x)
        x_new = np.clip(x_new, 0, 1e6)

        for i in range(self.N):
            self.vehicles[i]['prev_water'] = self.vehicles[i]['water']
            self.vehicles[i]['water'] = x_new[i]

        self.iteration += 1
        self._history.append(x_new.tolist())
        self.update_root_cause()

    def update_root_cause(self):
        """标记根因: 水滴最多的 Top-2 + 速度慢"""
        waters = np.array([v['water'] for v in self.vehicles])
        total = waters.sum() or 1.0
        pcts = waters / total * 100
        top2 = np.argsort(waters)[-2:]
        for i in range(self.N):
            self.vehicles[i]['pct'] = pcts[i]
            self.vehicles[i]['is_root_cause'] = i in top2 and self.vehicles[i]['speed'] < 1.0

    def reset(self):
        """重置到初始状态"""
        self.iteration = 0
        self._history = [[1.0] * self.N]
        for i in range(self.N):
            self.vehicles[i]['water'] = 1.0
            self.vehicles[i]['prev_water'] = 1.0
            self.vehicles[i]['is_root_cause'] = False
            self.vehicles[i]['pct'] = 100.0 / self.N

    def draw_vehicle(self, canvas, v, is_sel=False):
        """画一辆车 + 水滴指示"""
        x, y = int(v['x']), int(v['y'])
        wb, hb = CAR_W, CAR_H

        # 车体
        pts = np.array([[x - wb, y - hb], [x + wb, y - hb],
                        [x + wb, y + hb], [x - wb, y + hb]], dtype=np.int32)

        # 水滴值映射到颜色和大小
        base_waters = np.array([v['water'] for v in self.vehicles])
        max_w = max(base_waters) or 1.0
        ratio = v['water'] / max_w

        if v['is_root_cause']:
            color = C_RED
            glow = 4
        elif ratio > 0.6:
            color = C_ORANGE
            glow = 2
        else:
            color = (180, 180, 180)
            glow = 1

        # 发光效果
        for g in range(glow, 0, -1):
            cv2.polylines(canvas, [pts], True,
                         (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3)),
                         g * 3 + 2, cv2.LINE_AA)

        cv2.fillPoly(canvas, [pts], color)
        cv2.polylines(canvas, [pts], True, C_WHITE, 1, cv2.LINE_AA)

        # 水滴值显示
        water_str = f"{v['water']:.2f}"
        cv2.putText(canvas, water_str, (x - 20, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1, cv2.LINE_AA)

        # ID
        cv2.putText(canvas, f"#{v['id']}", (x - 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT, 1)

        if v['is_root_cause']:
            cv2.putText(canvas, "ROOT CAUSE", (x - 35, y - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_RED, 2, cv2.LINE_AA)

        # 速度标注
        speed_str = f"{v['speed']:.1f}m/s"
        cv2.putText(canvas, speed_str, (x - 18, y + hb + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)

    def draw_arrow(self, canvas, fr, to, weight, alpha=0.3):
        """画水滴流动箭头"""
        x1, y1 = int(fr[0]), int(fr[1])
        x2, y2 = int(to[0]), int(to[1])
        intensity = min(weight * 2, 1.0)
        color = (int(200 * intensity), int(200 * intensity * 0.3), int(60 * intensity))
        cv2.arrowedLine(canvas, (x1, y1 - 20), (x2, y2 - 20), color,
                        max(1, int(weight * 2)), cv2.LINE_AA, tipLength=0.15)

    def run(self):
        print("=== Water Drop Propagation Demo ===")
        print("  空格: 播放/暂停 | 1-9: 步进到第N轮 | R: 重置 | ESC: 退出")
        print()

        while True:
            canvas = np.full((H, W, 3), C_BG, dtype=np.uint8)

            # ── 标题 ──
            cv2.putText(canvas, "Water Drop Propagation: Finding the Root Cause",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 2)
            status = f"Iteration {self.iteration}/{self.max_iters}  |  {'PLAYING' if self.auto_play else 'PAUSED'}"
            cv2.putText(canvas, status, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_CYAN if self.auto_play else C_DIM, 1)

            # ── 画车辆 ──
            for v in self.vehicles:
                self.draw_vehicle(canvas, v)

            # ── 画水流连接 ──
            base_w = np.array([v['water'] for v in self.vehicles])
            max_w = max(base_w) or 1.0
            for i, j, w in self.connections:
                fr = (self.vehicles[i]['x'], self.vehicles[i]['y'])
                to = (self.vehicles[j]['x'], self.vehicles[j]['y'])
                flow_strength = w * (self.vehicles[i]['water'] / max_w)
                if flow_strength > 0.02:
                    self.draw_arrow(canvas, fr, to, flow_strength)

            # ── 信息面板 ──
            px, py = 20, H - 180
            cv2.rectangle(canvas, (px, py), (W - 20, py + 170), C_SURFACE, -1)
            cv2.rectangle(canvas, (px, py), (W - 20, py + 170), C_DIM, 1)

            # 排名表
            ranked = sorted(enumerate(self.vehicles), key=lambda x: x[1]['water'], reverse=True)
            cv2.putText(canvas, " Rank | ID | Speed | Water  | Root Cause %", (px + 10, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT, 1)
            total_w = sum(v['water'] for v in self.vehicles) or 1.0
            for ri, (idx, v) in enumerate(ranked[:8]):
                pct = v['water'] / total_w * 100
                row_color = C_RED if v['is_root_cause'] else (C_ORANGE if pct > 10 else C_DIM)
                label = f"  T={ri+1:<2d}   #{v['id']:<2d}   {v['speed']:.1f}    {v['water']:>6.2f}     {pct:>4.1f}%"
                if v['is_root_cause']:
                    label += "  <<< ROOT CAUSE"
                cv2.putText(canvas, label, (px + 10, py + 38 + ri * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, row_color, 1)

            # 邻接矩阵热力图
            mx0, my0 = W - 260, py + 5
            cv2.putText(canvas, "Adjacency Matrix A", (mx0, my0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_TEXT, 1)
            cell = min(180 // self.N, 20)
            for i in range(self.N):
                for j in range(self.N):
                    val = self.A[i, j]
                    if val > 0:
                        gr = int(255 * (1 - min(val, 1)))
                        cv2.rectangle(canvas, (mx0 + j * cell, my0 + 12 + i * cell),
                                      (mx0 + j * cell + cell - 1, my0 + 12 + i * cell + cell - 1),
                                      (gr, gr, 200), -1)
                    else:
                        cv2.rectangle(canvas, (mx0 + j * cell, my0 + 12 + i * cell),
                                      (mx0 + j * cell + cell - 1, my0 + 12 + i * cell + cell - 1),
                                      (40, 40, 45), -1)

            # 邻接矩阵标注
            cv2.putText(canvas, "i\\j  (i sends water to j)", (mx0, my0 + 12 + self.N * cell + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_DIM, 1)
            for i in range(self.N):
                cv2.putText(canvas, str(i), (mx0 + i * cell + 3, my0 + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_DIM, 1)

            # ── 控制提示 ──
            cv2.putText(canvas, "SPACE: play/pause  1-9: jump to iteration  R: reset  ESC: exit",
                        (20, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_DIM, 1)

            cv2.imshow("Water Drop Propagation Demo", canvas)

            if self.auto_play:
                key = cv2.waitKey(400) & 0xFF
            else:
                key = cv2.waitKey(50) & 0xFF

            if key == 27:
                break
            elif key == ord(' '):
                self.auto_play = not self.auto_play
            elif key == ord('r'):
                self.reset()
            elif ord('1') <= key <= ord('9'):
                target = key - ord('0')
                while self.iteration < target and self.iteration < self.max_iters:
                    self.propagate()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    WaterDropDemo().run()
