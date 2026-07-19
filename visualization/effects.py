"""
可视化特效模块

提供车辆发光、热力图颜色图例等纯渲染函数。
与引擎/检测/跟踪完全解耦。
"""

import cv2
import numpy as np

from utils.theme import THEME


def draw_glow_box(canvas: np.ndarray, pts: np.ndarray, color: tuple,
                  glow_radius: int = 3):
    """发光效果旋转矩形框——仅对高归因车辆使用。

    先画粗半透明光晕，再叠加主体。直接写入 canvas。
    """
    glow_overlay = canvas.copy()
    cv2.polylines(glow_overlay, [pts], isClosed=True,
                  color=color, thickness=glow_radius + 3, lineType=cv2.LINE_AA)
    cv2.addWeighted(glow_overlay, 0.25, canvas, 0.75, 0, dst=canvas)


def draw_heatmap_colorbar(canvas: np.ndarray, x: int, y: int,
                          w: int, h: int, vmin: float = 0.0, vmax: float = 1.0):
    """在 canvas 上绘制垂直颜色图例条（VIRIDIS 色盲友好）。"""
    grad = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_VIRIDIS)
    bar_resized = cv2.resize(bar, (w, h), interpolation=cv2.INTER_LINEAR)
    canvas[y:y + h, x:x + w] = bar_resized

    cv2.putText(canvas, f"{vmax:.1f}", (x + w + 4, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, THEME["text_secondary"],
                1, cv2.LINE_AA)
    cv2.putText(canvas, "0.0", (x + w + 4, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, THEME["text_secondary"],
                1, cv2.LINE_AA)
    cv2.putText(canvas, "C", (x + w + 2, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, THEME["text_dim"],
                1, cv2.LINE_AA)
