"""
BEV 透视变换演示 — 棋盘格标定 + 实时变换

用棋盘格模拟路面，拖拽 4 个角点观察透视变换效果。

python demo_test/bev_transform_demo.py
操作: 拖拽 BEV 面板的 4 个角点调整映射  |  R 重置  |  ESC 退出
"""

import cv2
import numpy as np

W, H = 1300, 800
PANEL_SZ = 580
BEV_SZ = 500

C_BG = (18, 18, 22)
C_SURFACE = (26, 28, 34)
C_TEXT = (200, 200, 200)
C_DIM = (100, 100, 100)
C_WHITE = (255, 255, 255)
C_CYAN = (60, 220, 230)
PCOLORS = [(80, 200, 255), (50, 200, 100), (50, 50, 255), (230, 220, 50)]
PNAMES = ["TL", "TR", "BR", "BL"]


def make_checkerboard(rows=8, cols=12, cell=40):
    """生成棋盘格图像 (模拟路面)"""
    h, w = rows * cell, cols * cell
    board = np.ones((h, w, 3), dtype=np.uint8) * 220
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                board[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = (70, 120, 180)
    # 添加中心标记
    cv2.line(board, (w//2-15, h//2), (w//2+15, h//2), (50, 50, 255), 2)
    cv2.line(board, (w//2, h//2-15), (w//2, h//2+15), (50, 50, 255), 2)
    # 添加文字标记
    cv2.putText(board, "INTERSECTION", (w//2-80, h//2+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 1)
    return board


def make_bev_checkerboard(bev_sz=500, cell=50):
    """生成 BEV 棋盘格目标图"""
    board = np.ones((bev_sz, bev_sz, 3), dtype=np.uint8) * 200
    n_cells = bev_sz // cell
    for r in range(n_cells):
        for c in range(n_cells):
            if (r + c) % 2 == 0:
                board[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = (60, 110, 170)
    # 网格线
    for i in range(n_cells + 1):
        cv2.line(board, (0, i*cell), (bev_sz, i*cell), (140, 150, 160), 1)
        cv2.line(board, (i*cell, 0), (i*cell, bev_sz), (140, 150, 160), 1)
    return board


class BEVDemo:
    def __init__(self):
        self.src_img = make_checkerboard()
        self.tgt_img = make_bev_checkerboard(BEV_SZ)
        self.h_src, self.w_src = self.src_img.shape[:2]

        # 源四点 (默认在棋盘上取4个角)
        margin = 40
        self.src_pts = np.array([
            [margin, margin],
            [self.w_src - margin, margin],
            [self.w_src - margin, self.h_src - margin],
            [margin, self.h_src - margin]
        ], dtype=np.float32)

        self.tgt_pts = np.array([
            [0, 0], [BEV_SZ-1, 0], [BEV_SZ-1, BEV_SZ-1], [0, BEV_SZ-1]
        ], dtype=np.float32)

        # 缩放原始图到面板
        s = min((PANEL_SZ - 20) / self.w_src, (PANEL_SZ - 20) / self.h_src)
        self.disp_w = int(self.w_src * s)
        self.disp_h = int(self.h_src * s)
        self.disp_src = cv2.resize(self.src_img, (self.disp_w, self.disp_h))
        self.scale = s
        self.off_x = (PANEL_SZ - self.disp_w) // 2
        self.off_y = (PANEL_SZ - self.disp_h) // 2 + 50

        # 源点缩放到显示坐标
        self.disp_pts = self.src_pts * s

        self.H = None
        self.bev_out = None
        self._drag = -1
        self._recompute()

        cv2.namedWindow("BEV Transform Demo")
        cv2.resizeWindow("BEV Transform Demo", W, H)
        cv2.setMouseCallback("BEV Transform Demo", self._mouse)

    def _recompute(self):
        """计算 H 并生成 BEV"""
        src = self.disp_pts / self.scale
        self.H, _ = cv2.findHomography(src, self.tgt_pts)
        if self.H is not None:
            self.bev_out = cv2.warpPerspective(self.src_img, self.H, (BEV_SZ, BEV_SZ))
            # 在 BEV 上叠加目标棋盘格（半透明）
            overlay = cv2.addWeighted(self.bev_out, 0.7, self.tgt_img, 0.3, 0)
            self.bev_out = overlay

    def _mouse(self, event, x, y, flags, param):
        # 判断哪个面板
        in_left = 30 <= x <= 30+PANEL_SZ and 50 <= y <= 50+PANEL_SZ
        in_right = 30+PANEL_SZ+20 <= x <= 30+PANEL_SZ+20+PANEL_SZ and 50 <= y <= 50+PANEL_SZ

        if event == cv2.EVENT_LBUTTONDOWN:
            if in_left and self.disp_pts is not None:
                best, bd = -1, 50
                for i in range(4):
                    px = int(self.disp_pts[i][0] + self.off_x)
                    py = int(self.disp_pts[i][1] + self.off_y)
                    d = np.hypot(x - px, y - py)
                    if d < bd: bd, best = d, i
                self._drag = best if best >= 0 else -1
            elif in_right:
                ox2 = 30 + PANEL_SZ + 20 + 10
                oy2 = 50 + 50
                bsz = PANEL_SZ - 70
                s2 = bsz / BEV_SZ
                best, bd = -1, 50
                for i in range(4):
                    px = int(self.tgt_pts[i][0] * s2 + ox2)
                    py = int(self.tgt_pts[i][1] * s2 + oy2)
                    d = np.hypot(x - px, y - py)
                    if d < bd: bd, best = d, i
                self._drag = best if best >= 0 else -1
            else:
                self._drag = -1

        elif event == cv2.EVENT_MOUSEMOVE and self._drag >= 0:
            if in_left:
                # 限制在显示区域内
                ix = np.clip(x - self.off_x, 0, self.disp_w - 1)
                iy = np.clip(y - self.off_y, 0, self.disp_h - 1)
                self.disp_pts[self._drag] = [ix, iy]
                self._recompute()
            elif in_right:
                ox2 = 30 + PANEL_SZ + 20 + 10
                oy2 = 50 + 50
                bsz = PANEL_SZ - 70
                s2 = bsz / BEV_SZ
                tx = np.clip((x - ox2) / s2, 0, BEV_SZ - 1)
                ty = np.clip((y - oy2) / s2, 0, BEV_SZ - 1)
                self.tgt_pts[self._drag] = [tx, ty]
                self._recompute()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = -1

    def run(self):
        print("=== BEV Transform Demo ===")
        print("  Drag color points (left panel = source, right panel = target)")
        print("  R: reset  |  ESC: exit")
        print()

        while True:
            canvas = np.full((H, W, 3), C_BG, dtype=np.uint8)

            # ── 左面板: 原始图像 ──
            panel1 = np.full((PANEL_SZ, PANEL_SZ, 3), C_SURFACE, dtype=np.uint8)
            cv2.putText(panel1, "Original (Perspective View)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)
            cv2.putText(panel1, "Drag source corner points", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_DIM, 1)

            panel1[self.off_y:self.off_y+self.disp_h,
                   self.off_x:self.off_x+self.disp_w] = self.disp_src

            # 画源点四边形
            for i in range(4):
                x = int(self.disp_pts[i][0] + self.off_x)
                y = int(self.disp_pts[i][1] + self.off_y)
                cv2.circle(panel1, (x, y), 8, PCOLORS[i], -1, cv2.LINE_AA)
                cv2.circle(panel1, (x, y), 8, C_WHITE, 1, cv2.LINE_AA)
                cv2.putText(panel1, PNAMES[i], (x+10, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, PCOLORS[i], 1, cv2.LINE_AA)
            for i in range(4):
                j = (i+1) % 4
                x1 = int(self.disp_pts[i][0]+self.off_x); y1 = int(self.disp_pts[i][1]+self.off_y)
                x2 = int(self.disp_pts[j][0]+self.off_x); y2 = int(self.disp_pts[j][1]+self.off_y)
                cv2.line(panel1, (x1, y1), (x2, y2), C_WHITE, 1, cv2.LINE_AA)

            canvas[50:50+PANEL_SZ, 30:30+PANEL_SZ] = panel1

            # ── 右面板: BEV ──
            panel2 = np.full((PANEL_SZ, PANEL_SZ, 3), C_SURFACE, dtype=np.uint8)
            cv2.putText(panel2, "Bird's Eye View (BEV)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)
            cv2.putText(panel2, "Drag target corner points", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_DIM, 1)

            if self.bev_out is not None:
                ox2 = 10; oy2 = 50
                bsz = PANEL_SZ - 65
                s2 = bsz / BEV_SZ
                out_sz = int(BEV_SZ * s2)
                bev_disp = cv2.resize(self.bev_out, (out_sz, out_sz))
                panel2[oy2:oy2+out_sz, ox2:ox2+out_sz] = bev_disp

                # 画目标点
                for i in range(4):
                    x = int(self.tgt_pts[i][0] * s2 + ox2)
                    y = int(self.tgt_pts[i][1] * s2 + oy2)
                    cv2.circle(panel2, (x, y), 8, PCOLORS[i], -1, cv2.LINE_AA)
                    cv2.circle(panel2, (x, y), 8, C_WHITE, 1, cv2.LINE_AA)
                for i in range(4):
                    j = (i+1) % 4
                    x1 = int(self.tgt_pts[i][0]*s2+ox2); y1 = int(self.tgt_pts[i][1]*s2+oy2)
                    x2 = int(self.tgt_pts[j][0]*s2+ox2); y2 = int(self.tgt_pts[j][1]*s2+oy2)
                    cv2.line(panel2, (x1, y1), (x2, y2), C_WHITE, 1, cv2.LINE_AA)

            canvas[50:50+PANEL_SZ, 30+PANEL_SZ+20:30+PANEL_SZ+20+PANEL_SZ] = panel2

            # ── 底部信息 ──
            bot = np.full((120, W-60, 3), C_SURFACE, dtype=np.uint8)
            if self.H is not None:
                h_str = np.array2string(self.H, precision=2, suppress_small=True)
                for li, line in enumerate(h_str.split('\n')[:3]):
                    cv2.putText(bot, line, (20, 22+li*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_CYAN, 1)

            cv2.putText(bot, "R: reset  |  ESC: exit", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)
            cv2.putText(bot, f"BEV: {BEV_SZ}x{BEV_SZ}  |  Source: {self.w_src}x{self.h_src}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)

            canvas[50+PANEL_SZ+15:50+PANEL_SZ+15+120, 30:30+W-60] = bot

            cv2.imshow("BEV Transform Demo", canvas)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                self.disp_pts = self.src_pts.copy() * self.scale
                self.tgt_pts = np.array([[0,0],[BEV_SZ-1,0],[BEV_SZ-1,BEV_SZ-1],[0,BEV_SZ-1]], dtype=np.float32)
                self._recompute()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    BEVDemo().run()
