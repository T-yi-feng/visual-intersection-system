"""
拥堵叠加可视化模块

负责实时拥堵热力图叠加：基于归因分数的车辆颜色渲染。
"""

import cv2
import numpy as np


def build_realtime_congestion_overlay(
    base_img: np.ndarray,
    tracked_meta: dict,
    influences: list[float],
    vehicle_indices: dict,  # track_id -> index in influences
    vehicle_size_m: dict,
    pixels_per_meter: float,
    alpha_min: float = 0.18,
    alpha_max: float = 0.73,
) -> np.ndarray:
    """
    在图像上叠加拥堵贡献热力图。

    归因分数越高的车辆，叠加颜色越深（越红）。

    Parameters
    ----------
    base_img : 底图
    tracked_meta : 跟踪元数据
    influences : 每辆车的归因分数
    vehicle_indices : track_id → influences 索引的映射
    alpha_min : 最小透明度
    alpha_max : 最大透明度

    Returns
    -------
    overlay : np.ndarray, 叠加后的图像
    """
    overlay = base_img.copy()

    if not influences:
        return overlay

    max_inf = max(influences)
    if max_inf <= 0:
        return overlay

    for tid, meta in tracked_meta.items():
        idx = vehicle_indices.get(tid)
        if idx is None or idx >= len(influences):
            continue

        inf = influences[idx]
        if inf <= 0:
            continue

        # 归一化归因分数 → 颜色（蓝→红）
        ratio = min(inf / max_inf, 1.0)
        b = int(255 * (1 - ratio))
        g = int(50 * (1 - abs(ratio - 0.5) * 2))
        r = int(255 * ratio)
        color = (b, g, r)

        # 透明度随归因分数增加
        alpha = alpha_min + (alpha_max - alpha_min) * ratio

        # 绘制半透明车辆框
        cx, cy = meta.get('center', (0, 0))
        label = meta.get('label', 'car')
        size_info = vehicle_size_m.get(label, {'length_m': 4.0, 'width_m': 1.6})

        half_l = size_info['length_m'] * pixels_per_meter * 0.65 / 2
        half_w = size_info['width_m'] * pixels_per_meter * 0.65 / 2

        # 旋转框
        heading = meta.get('heading_deg', 0)
        rad = np.radians(heading)
        cos_h, sin_h = np.cos(rad), np.sin(rad)

        corners = [
            (-half_l, -half_w), (half_l, -half_w),
            (half_l, half_w), (-half_l, half_w),
        ]
        rotated = [(int(cx + c[0]*cos_h - c[1]*sin_h),
                    int(cy + c[0]*sin_h + c[1]*cos_h)) for c in corners]

        pts = np.array(rotated, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)

    # 混合
    result = cv2.addWeighted(overlay, 0.4, base_img, 0.6, 0)
    return result


def build_conflict_heatmap(
    conflict_field: np.ndarray,
    output_size: tuple[int, int],
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    将冲突场渲染为热力图。

    Parameters
    ----------
    conflict_field : (H, W) float32 冲突场
    output_size : (W, H) 输出尺寸
    colormap : OpenCV colormap

    Returns
    -------
    heatmap : np.ndarray, BGR 热力图
    """
    c_max = conflict_field.max()
    if c_max > 0:
        normalized = (conflict_field / c_max * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(conflict_field, dtype=np.uint8)

    colored = cv2.applyColorMap(normalized, colormap)
    resized = cv2.resize(colored, output_size, interpolation=cv2.INTER_LINEAR)
    return resized
