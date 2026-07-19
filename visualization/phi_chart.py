"""
Phi 图表渲染模块 (v2)

改进：
- 网格线 + 轴标签
- 曲线下半透明渐变填充
- 当前值大字 callout + 圆角背景
- cv2.polylines 批量绘制替代逐段 cv2.line
- 事件摘要条
- 统一主题色
"""

import cv2
import numpy as np

from utils.theme import THEME, phi_color, phi_label_en
from utils.drawing import draw_text_with_bg


def render_phi_chart_panel(
    size_hw: tuple[int, int],
    phi_series: list[tuple[float, float]],
    phi_threshold: float = 0.75,
    chart_window_seconds: float = 120.0,
    event_info: dict = None,
) -> np.ndarray:
    """
    渲染实时 Phi 时间线图表 (v2)。

    Parameters
    ----------
    size_hw : (H, W) 面板尺寸
    phi_series : list of (time_s, phi_t)
    phi_threshold : 事件触发阈值线
    chart_window_seconds : 时间窗口宽度
    event_info : 可选, 当前事件信息 dict
        {active: bool, start_t: float, peak_phi: float, peak_t: float,
         top_vehicle_id: int, top_vehicle_type: str, duration_s: float}

    Returns
    -------
    chart : np.ndarray (H, W, 3) BGR
    """
    panel_h, panel_w = size_hw
    canvas = np.full((panel_h, panel_w, 3), THEME["bg_canvas"], dtype=np.uint8)

    # ── 绘图区域 ──
    margin_l, margin_r = 62, 22
    margin_t, margin_b = 44, 36
    plot_w = panel_w - margin_l - margin_r
    plot_h = panel_h - margin_t - margin_b

    if not phi_series:
        draw_text_with_bg(canvas, "Phi: --", (10, 30),
                          THEME["text_secondary"], 0.6)
        return canvas

    # ── 时间窗口 ──
    t_max = phi_series[-1][0]
    t_min = max(0, t_max - chart_window_seconds)

    window_data = [(t, p) for t, p in phi_series if t >= t_min]
    if len(window_data) < 2:
        draw_text_with_bg(canvas, "Phi: --", (10, 30),
                          THEME["text_secondary"], 0.6)
        return canvas

    # ── 坐标映射 ──
    def tx(t_val):
        return int(margin_l + (t_val - t_min) / max(t_max - t_min, 1e-6) * plot_w)

    def py(phi_val):
        return int(margin_t + plot_h - phi_val * plot_h)

    # ── 1. 水平网格线 (Y 轴: 0.0, 0.3, 0.5, 0.7, 1.0) ──
    grid_levels = [0.0, 0.3, 0.5, 0.7, 1.0]
    for level in grid_levels:
        y = py(min(level, 1.0))
        cv2.line(canvas, (margin_l, y), (panel_w - margin_r, y),
                 THEME["border_panel"], 1, cv2.LINE_AA)

    # ── 2. 垂直网格线 (时间刻度, 目标 ~5 条) ──
    time_span = t_max - t_min
    tick_interval = _nice_tick_interval(time_span, target_ticks=5)
    tick_t = t_min - (t_min % tick_interval) + tick_interval
    while tick_t < t_max:
        x = tx(tick_t)
        cv2.line(canvas, (x, margin_t), (x, margin_t + plot_h),
                 THEME["border_panel"], 1, cv2.LINE_AA)
        # X 轴时间标签
        rel_s = tick_t - t_max
        label = f"{rel_s:.0f}s" if rel_s != 0 else "now"
        draw_text_with_bg(canvas, label, (x - 14, panel_h - margin_b + 16),
                          THEME["text_dim"], 0.28)
        tick_t += tick_interval

    # ── 3. 阈值线 ──
    y_thr = py(phi_threshold)
    cv2.line(canvas, (margin_l, y_thr), (panel_w - margin_r, y_thr),
             THEME["danger"], 1, cv2.LINE_AA)
    # 阈值标签
    draw_text_with_bg(canvas, f"T={phi_threshold:.2f}",
                      (panel_w - margin_r - 56, y_thr - 14),
                      THEME["danger"], 0.3)

    # ── 4. 曲线下半透明填充 ──
    t_vals = np.array([t for t, _ in window_data])
    p_vals = np.array([min(p, 1.0) for _, p in window_data])
    xs = np.array([tx(t) for t in t_vals])
    ys = np.array([py(p) for p in p_vals])

    # 构建填充多边形: 曲线点 + 底部水平线
    bottom = panel_h - margin_b
    poly_pts = np.column_stack([xs, ys]).astype(np.int32)
    poly = np.vstack([
        poly_pts,
        [[poly_pts[-1][0], bottom]],
        [[poly_pts[0][0], bottom]],
    ])
    # 在 overlay 上画填充然后混合
    fill_overlay = canvas.copy()
    cv2.fillPoly(fill_overlay, [poly], (60, 55, 50))
    cv2.addWeighted(fill_overlay, 0.22, canvas, 0.78, 0, dst=canvas)

    # ── 5. 曲线 (批量 polylines + 逐段着色) ──
    # 为了保持渐变着色，仍逐段绘制但用 LINE_AA
    for i in range(1, len(window_data)):
        _, p0 = window_data[i - 1]
        _, p1 = window_data[i]
        color = phi_color(min(p1, 1.0))
        cv2.line(canvas,
                 (xs[i - 1], ys[i - 1]), (xs[i], ys[i]),
                 color, 2, cv2.LINE_AA)

    # ── 6. 当前值 callout ──
    if window_data:
        t_cur, phi_cur = window_data[-1]
        cx, cy = xs[-1], ys[-1]

        # 圆点标记
        cv2.circle(canvas, (cx, cy), 5, THEME["accent"], -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 8, THEME["accent"], 1, cv2.LINE_AA)

        # 大字 Phi 值 (左上角浮动)
        level = phi_label_en(phi_cur)
        label = f"Phi  {phi_cur:.3f}  [{level}]"
        color = phi_color(phi_cur)

        # 圆角背景框
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx, by = 10, 6
        bw, bh = tw + 20, th + 14
        bg_rect = canvas[by:by + bh, bx:bx + bw].copy()
        bg_rect[:] = THEME["bg_panel"]
        cv2.addWeighted(bg_rect, 0.85, canvas[by:by + bh, bx:bx + bw], 0.15, 0,
                        dst=canvas[by:by + bh, bx:bx + bw])
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh),
                      THEME["border_panel"], 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (bx + 10, by + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # ── 7. Y 轴标签 ──
    for level in grid_levels:
        y = py(min(level, 1.0))
        draw_text_with_bg(canvas, f"{level:.1f}",
                          (4, y + 4), THEME["text_dim"], 0.28)

    # ── 8. 事件摘要条 (底部) ──
    if event_info and event_info.get("active"):
        _draw_event_bar(canvas, event_info, panel_w, panel_h)

    # ── 9. 面板边框 ──
    cv2.rectangle(canvas, (0, 0), (panel_w - 1, panel_h - 1),
                  THEME["border_panel"], 1)

    return canvas


def _draw_event_bar(canvas, info, panel_w, panel_h):
    """在图表底部绘制事件摘要条"""
    bar_h = 26
    bar_y = panel_h - bar_h - 2
    bar_color = THEME["danger"]

    # 半透明背景条
    bar_roi = canvas[bar_y:bar_y + bar_h, 10:panel_w - 10]
    overlay = bar_roi.copy()
    overlay[:] = (38, 32, 32)
    cv2.addWeighted(overlay, 0.8, bar_roi, 0.2, 0, dst=bar_roi)

    # 左边强调线
    cv2.line(bar_roi, (0, 4), (0, bar_h - 4), bar_color, 3, cv2.LINE_AA)

    # 事件文本
    dur = info.get("duration_s", 0)
    peak = info.get("peak_phi", 0)
    vid = info.get("top_vehicle_id", "?")
    vtype = info.get("top_vehicle_type", "?")
    msg = f"Event Active | Dur {dur:.1f}s | Peak Phi={peak:.3f} | Top Vehicle: ID={vid} ({vtype})"

    cv2.putText(bar_roi, msg, (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, THEME["text_primary"], 1, cv2.LINE_AA)


def _nice_tick_interval(span: float, target_ticks: int = 5) -> float:
    """计算"好看"的刻度间隔（如 10s, 20s, 30s, 60s）"""
    raw = span / target_ticks
    nice_steps = [5, 10, 15, 20, 30, 60, 120, 300]
    best = nice_steps[0]
    for step in nice_steps:
        if step >= raw:
            return step
        best = step
    return best


def export_paper_phi_figure(
    out_path: str,
    width: int,
    height: int,
    phi_series: list[tuple[float, float]],
    phi_threshold: float = 0.75,
    title: str = "Phi Timeline",
):
    """导出论文级 Phi 图表 (matplotlib, 无改动)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if not phi_series:
        return

    times = [t for t, _ in phi_series]
    phis = [p for _, p in phi_series]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    points = np.array([times, phis]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=2)
    lc.set_array(np.array(phis[:-1]))
    ax.add_collection(lc)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, 1)
    ax.axhline(y=phi_threshold, color='r', linestyle='--', alpha=0.5,
               label=f'Threshold={phi_threshold}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Phi')
    ax.set_title(title)
    ax.legend()

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
