"""
Phi 图表渲染模块

负责实时 Phi 时间线图表和论文级图表的渲染。
"""

import cv2
import numpy as np

from utils.drawing import draw_text_with_bg


def render_phi_chart_panel(
    size_hw: tuple[int, int],
    phi_series: list[tuple[float, float]],
    phi_threshold: float = 0.75,
    chart_window_seconds: float = 120.0,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    line_color: tuple[int, int, int] = (0, 200, 255),
    threshold_color: tuple[int, int, int] = (0, 0, 200),
) -> np.ndarray:
    """
    渲染实时 Phi 时间线图表。

    Parameters
    ----------
    size_hw : (H, W) 面板尺寸
    phi_series : list of (time_s, phi_t)
    phi_threshold : 阈值线

    Returns
    -------
    chart : np.ndarray
    """
    h, w = size_hw
    chart = np.full((h, w, 3), bg_color, dtype=np.uint8)

    if not phi_series:
        draw_text_with_bg(chart, "Phi: --", (10, 30), (200, 200, 200), 0.6)
        return chart

    # 时间窗口
    t_max = phi_series[-1][0]
    t_min = max(0, t_max - chart_window_seconds)

    # 过滤窗口内的数据
    window_data = [(t, p) for t, p in phi_series if t >= t_min]
    if len(window_data) < 2:
        return chart

    # 绘图区域
    margin_l, margin_r = 60, 20
    margin_t, margin_b = 40, 30
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    # 坐标映射
    def tx(t):
        return int(margin_l + (t - t_min) / max(t_max - t_min, 1) * plot_w)

    def py(phi):
        return int(margin_t + plot_h - phi * plot_h)

    # 阈值线
    y_thr = py(phi_threshold)
    cv2.line(chart, (margin_l, y_thr), (w - margin_r, y_thr), threshold_color, 1)

    # 绘制 Phi 曲线（冷暖色渐变）
    for i in range(1, len(window_data)):
        t0, p0 = window_data[i - 1]
        t1, p1 = window_data[i]

        # 颜色：蓝(低) → 红(高)
        ratio = min(p1, 1.0)
        b = int(255 * (1 - ratio))
        g = int(100 * (1 - abs(ratio - 0.5) * 2))
        r = int(255 * ratio)

        cv2.line(chart, (tx(t0), py(p0)), (tx(t1), py(p1)), (b, g, r), 2)

    # 当前值标注
    if window_data:
        t_cur, phi_cur = window_data[-1]
        cv2.circle(chart, (tx(t_cur), py(phi_cur)), 4, line_color, -1)
        draw_text_with_bg(
            chart, f"Phi: {phi_cur:.3f}",
            (10, 25), line_color, 0.6, 1,
        )

    # 坐标轴标签
    draw_text_with_bg(chart, "1.0", (5, margin_t + 5), (150, 150, 150), 0.3)
    draw_text_with_bg(chart, "0.0", (5, margin_t + plot_h), (150, 150, 150), 0.3)
    draw_text_with_bg(chart, f"{phi_threshold:.2f}", (5, y_thr + 5),
                     threshold_color, 0.3)

    return chart


def export_paper_phi_figure(
    out_path: str,
    width: int,
    height: int,
    phi_series: list[tuple[float, float]],
    phi_threshold: float = 0.75,
    title: str = "Phi Timeline",
):
    """导出论文级 Phi 图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not phi_series:
        return

    times = [t for t, _ in phi_series]
    phis = [p for _, p in phi_series]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # 冷暖色渐变线段
    points = np.array([times, phis]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=2)
    lc.set_array(np.array(phis[:-1]))
    ax.add_collection(lc)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.axhline(y=phi_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold={phi_threshold}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phi')
    ax.set_title(title)
    ax.legend()

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
