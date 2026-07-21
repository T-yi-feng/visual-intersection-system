"""
Velocity-Aware Kernel Convolution Demo — 速度感知核卷积演示

展示不同车速下各向异性核形态的变化 + 卷积后的影响场热力图。

操作: 拖拽调整车速滑块 | 空格切换视角

python demo_test/speed_kernel_demo.py
"""

import cv2
import numpy as np
from math import exp, cos, sin, pi

W, H = 1300, 820
CELL_M = 0.78  # m/cell

C_BG = (18, 18, 22)
C_SURFACE = (26, 28, 34)
C_TEXT = (200, 200, 200)
C_DIM = (100, 100, 100)
C_WHITE = (255, 255, 255)
C_CYAN = (60, 220, 230)

# 核默认参数
SIGMA_ALONG = 3.0
SIGMA_PERP = 0.6
HALF_LEN = 10
HALF_WIDTH = 6
ALPHA = 1.0      # 速度敏感系数
V_REF = 5.0      # 参考速度 (m/s)


def build_kernel(heading, speed, sigma_a=SIGMA_ALONG, sigma_p=SIGMA_PERP, alpha=ALPHA):
    """速度感知核: sigma_along(v) = sigma_0 * (1 + alpha * v / v_ref)"""
    sigma_v = sigma_a * (1 + alpha * speed / V_REF)
    k_size = 2 * HALF_LEN + 1
    K = np.zeros((k_size, k_size), dtype=np.float32)
    cx = cy = HALF_LEN
    theta = heading * pi / 180.0
    ux, uy = cos(theta), -sin(theta)
    nx, ny = sin(theta), cos(theta)
    for i in range(k_size):
        for j in range(k_size):
            dx, dy = j - cx, i - cy
            along = dx * ux + dy * uy
            perp = dx * nx + dy * ny
            if along >= 0:
                eff_s = sigma_v
                eff_h = HALF_LEN
            else:
                eff_s = sigma_v * 0.33
                eff_h = HALF_LEN * 0.33
            f_a = exp(-0.5 * (along / eff_s) ** 2) if abs(along) <= eff_h else 0.0
            if along >= 0:
                fan = sigma_p * (1 + 0.6 * along / max(sigma_v, 1e-6))
            else:
                fan = sigma_p
            f_p = exp(-0.5 * (perp / max(fan, 0.1)) ** 2)
            K[i, j] = f_a * f_p
    K /= K.sum()
    return K, sigma_v


def convolve_kernel(grid_val, K):
    """简易卷积: 单点源在网格中心"""
    k_h, k_w = K.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(grid_val, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(grid_val)
    g_h, g_w = grid_val.shape
    for i in range(g_h):
        for j in range(g_w):
            result[i, j] = (padded[i:i+k_h, j:j+k_w] * K).sum()
    return result


def make_grid_panel(h, w):
    """创建带网格线的面板"""
    panel = np.full((h, w, 3), C_BG, dtype=np.uint8)
    for i in range(0, w, 20):
        cv2.line(panel, (i, 0), (i, h), (32, 34, 38), 1)
    for i in range(0, h, 20):
        cv2.line(panel, (0, i), (w, i), (32, 34, 38), 1)
    return panel


class SpeedKernelDemo:
    def __init__(self):
        self.speeds = [0, 2, 5, 10, 15]  # m/s: 停止/慢行/市区/快速/高速
        self.sel_speed_idx = 2
        self.show_conv = False
        self.kernel_size = 2 * HALF_LEN + 1
        self.grid_size = 80
        self.alpha = ALPHA

        cv2.namedWindow("Velocity-Aware Kernel Demo")
        cv2.resizeWindow("Velocity-Aware Kernel Demo", W, H)

    def run(self):
        print("=== Velocity-Aware Kernel Convolution Demo ===")
        print("  LEFT/RIGHT: change selected speed  |  SPACE: toggle kernel/convolution view")
        print("  A/D: adjust alpha (speed sensitivity)  |  ESC: exit")
        print()

        while True:
            canvas = np.full((H, W, 3), C_BG, dtype=np.uint8)

            # ── 顶部参数栏 ──
            cv2.putText(canvas, "Velocity-Aware Anisotropic Kernel", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 2)
            cv2.putText(canvas,
                f"sigma_along(v) = sigma_0 x (1 + alpha x v / v_ref)"
                f"  |  sigma_0={SIGMA_ALONG}  alpha={self.alpha:.1f}  v_ref={V_REF}m/s  |  "
                f"SELECTED: v={self.speeds[self.sel_speed_idx]}m/s",
                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_CYAN, 1)

            # ── 5 个速度等级的核 ──
            n_cars = len(self.speeds)
            car_w = (W - 40) // n_cars
            car_h = 280

            kernels = []
            sigmas = []
            for s in self.speeds:
                K, sv = build_kernel(0, s, alpha=self.alpha)
                kernels.append(K)
                sigmas.append(sv)

            for ci, (speed, K, sv) in enumerate(zip(self.speeds, kernels, sigmas)):
                cx = 20 + ci * car_w
                cy = 75

                # 面板
                panel = np.full((car_h, car_w-10, 3), C_SURFACE, dtype=np.uint8)

                # 核热力图
                if K.max() > 0:
                    K_u8 = (K / K.max() * 255).astype(np.uint8)
                    K_color = cv2.applyColorMap(K_u8, cv2.COLORMAP_VIRIDIS)
                else:
                    K_color = np.zeros_like(panel, dtype=np.uint8)

                k_disp_sz = min(car_w - 40, car_h - 60)
                K_rs = cv2.resize(K_color, (k_disp_sz, k_disp_sz), interpolation=cv2.INTER_LINEAR)
                ox = (car_w - 10 - k_disp_sz) // 2
                oy = 30
                panel[oy:oy+k_disp_sz, ox:ox+k_disp_sz] = K_rs
                cv2.rectangle(panel, (ox, oy), (ox+k_disp_sz, oy+k_disp_sz), (60, 60, 60), 1)

                # 速度标注
                speed_label = f"v={speed} m/s"
                if speed == 0:
                    speed_label += " (stopped)"
                elif speed <= 2:
                    speed_label += " (crawl)"
                elif speed <= 5:
                    speed_label += " (city)"
                elif speed <= 10:
                    speed_label += " (fast)"
                else:
                    speed_label += " (highway)"

                cv2.putText(panel, speed_label, (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            C_CYAN if ci == self.sel_speed_idx else C_TEXT, 1)

                # sigma 值
                cv2.putText(panel, f"sigma_fwd={sv:.1f}cell ({sv*CELL_M:.1f}m)",
                            (6, oy+k_disp_sz+16), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_DIM, 1)

                # 选中标记
                if ci == self.sel_speed_idx:
                    cv2.rectangle(panel, (1, 1), (car_w-12, car_h-2), C_CYAN, 2)

                # 方向箭头（在核上画）
                arr_len = k_disp_sz // 3
                cx_k = ox + k_disp_sz // 2
                cy_k = oy + k_disp_sz // 2
                ax = cx_k + arr_len
                ay = cy_k
                cv2.arrowedLine(panel, (cx_k, cy_k), (ax, ay), C_WHITE, 1, cv2.LINE_AA, tipLength=0.2)
                cv2.putText(panel, "forward", (cx_k + arr_len//2, cy_k - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_DIM, 1)

                canvas[cy:cy+car_h, cx:cx+car_w-10] = panel

            # ── 中间：选中的速度等级的核 + 卷积结果 ──
            my0 = 75 + car_h + 15
            mh = 380

            sel_speed = self.speeds[self.sel_speed_idx]
            K_sel, sv_sel = build_kernel(0, sel_speed, alpha=self.alpha)
            k_disp_sz2 = 200

            panel_left = np.full((mh, 400, 3), C_SURFACE, dtype=np.uint8)

            # 选中核放大
            K_u8 = (K_sel / K_sel.max() * 255).astype(np.uint8)
            K_color = cv2.applyColorMap(K_u8, cv2.COLORMAP_VIRIDIS)
            K_big = cv2.resize(K_color, (k_disp_sz2, k_disp_sz2), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(K_big, (0, 0), (k_disp_sz2-1, k_disp_sz2-1), C_DIM, 1)

            # 截面曲线
            mid = K_sel.shape[0] // 2
            profile = K_sel[mid, :]
            profile_norm = profile / profile.max() * 80  # 缩放到80px

            # 空白区域放截面图
            cx_panel = 190
            cy_panel = 20
            panel_left[cy_panel:cy_panel+k_disp_sz2, cx_panel:cx_panel+k_disp_sz2] = K_big

            # 信息
            info_lines = [
                f"Speed: {sel_speed} m/s ({sel_speed*3.6:.0f} km/h)",
                f"sigma_along = {sv_sel:.1f} cells ({sv_sel*CELL_M:.1f} m)",
                f"sigma_perp = {SIGMA_PERP:.1f} cells ({SIGMA_PERP*CELL_M:.1f} m)",
                f"Effective forward (3sigma): {3*sv_sel*CELL_M:.1f} m",
                f"Kernel size: {self.kernel_size}x{self.kernel_size}",
                "",
                "Profile (center row):",
                f"  Forward peak at +{sv_sel:.1f}cell = {sv_sel*CELL_M:.1f}m",
                f"  Cutoff at +{HALF_LEN}cell = {HALF_LEN*CELL_M:.1f}m",
            ]
            for li, l in enumerate(info_lines):
                cv2.putText(panel_left, l, (15, cy_panel+li*18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_TEXT if ':' in l else C_DIM, 1)

            # 截面曲线绘制
            curve_y0 = cy_panel + k_disp_sz2 + 20
            curve_h = 80
            curve_w = k_disp_sz2
            for xi in range(1, len(profile_norm)):
                x1 = cx_panel + (xi-1) * curve_w // len(profile_norm)
                x2 = cx_panel + xi * curve_w // len(profile_norm)
                y1 = curve_y0 + curve_h - int(profile_norm[xi-1])
                y2 = curve_y0 + curve_h - int(profile_norm[xi])
                cv2.line(panel_left, (x1, y1), (x2, y2), C_CYAN, 1, cv2.LINE_AA)
            cv2.rectangle(panel_left, (cx_panel, curve_y0), (cx_panel+curve_w, curve_y0+curve_h), C_DIM, 1)
            cv2.putText(panel_left, "Forward Profile", (cx_panel, curve_y0-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_DIM, 1)
            # 标记 1sigma / 2sigma
            for si, label in [(1, '1s'), (2, '2s'), (3, '3s')]:
                x = cx_panel + int(si * sv_sel / (2*HALF_LEN+1) * curve_w)
                if x < cx_panel + curve_w:
                    cv2.line(panel_left, (x, curve_y0), (x, curve_y0+curve_h), C_DIM, 1, cv2.LINE_AA)
                    cv2.putText(panel_left, label, (x-4, curve_y0+curve_h+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, C_DIM, 1)

            canvas[my0:my0+mh, 20:20+400] = panel_left

            # ── 右侧: 影响场对比 ──
            rx0 = 440
            panel_right = np.full((mh, W - rx0 - 20, 3), C_SURFACE, dtype=np.uint8)

            # 多辆车不同速度的场景
            gs = 60
            grid = np.zeros((gs, gs), dtype=np.float32)

            # 多辆车：不同速度，同一方向
            test_speeds = [0, 0.5, 2.0, 5.0, 10.0]
            test_positions = [(20, 30), (26, 30), (32, 30), (38, 30), (44, 30)]
            test_names = ["Stop", "Crawl", "City", "Fast", "Highway"]

            for (gx, gy), sp, nm in zip(test_positions, test_speeds, test_names):
                if 0 <= gx < gs and 0 <= gy < gs:
                    grid[gy, gx] = 1.0
                    # 画车
                    cv2.circle(panel_right, (int(gx*5.5)+40, int(gy*5.5)+30), 5,
                               (80, 200, 255), -1)
                    cv2.putText(panel_right, nm, (int(gx*5.5)+40, int(gy*5.5)+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, C_DIM, 1)

            # 对每辆车用速度感知核做卷积
            R_total = np.zeros((gs, gs), dtype=np.float32)
            for (gx, gy), sp in zip(test_positions, test_speeds):
                K_v, sv = build_kernel(90, sp, alpha=self.alpha)  # 朝北
                grid_i = np.zeros((gs, gs), dtype=np.float32)
                if 0 <= gx < gs and 0 <= gy < gs:
                    grid_i[gy, gx] = 1.0
                R_i = convolve_kernel(grid_i, K_v)
                R_total += R_i

            # 渲染影响场
            if R_total.max() > 0:
                R_u8 = (R_total / R_total.max() * 255).astype(np.uint8)
                R_color = cv2.applyColorMap(R_u8, cv2.COLORMAP_INFERNO)
                R_disp = cv2.resize(R_color, (W - rx0 - 40, mh - 60), interpolation=cv2.INTER_LINEAR)
                panel_right[25:25+R_disp.shape[0], 15:15+R_disp.shape[1]] = R_disp

            cv2.putText(panel_right, "Influence Field: 5 vehicles at different speeds",
                        (15, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_TEXT, 1)
            cv2.putText(panel_right, "Faster vehicle = longer forward influence",
                        (15, mh-12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_DIM, 1)

            canvas[my0:my0+mh, rx0:rx0+panel_right.shape[1]] = panel_right

            # ── 底部提示 ──
            cv2.putText(canvas,
                "LEFT/RIGHT: change speed  |  A/D: adjust alpha (now {:.1f})  |  SPACE: toggle view  |  ESC: exit".format(self.alpha),
                (20, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_DIM, 1)

            cv2.imshow("Velocity-Aware Kernel Demo", canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == 81:  # LEFT
                self.sel_speed_idx = max(0, self.sel_speed_idx - 1)
            elif key == 83:  # RIGHT
                self.sel_speed_idx = min(len(self.speeds)-1, self.sel_speed_idx + 1)
            elif key == ord(' '):
                self.show_conv = not self.show_conv
            elif key == ord('a'):
                self.alpha = max(0.0, self.alpha - 0.1)
            elif key == ord('d'):
                self.alpha = min(3.0, self.alpha + 0.1)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    SpeedKernelDemo().run()
