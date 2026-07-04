"""
鸟瞰变换模块 (Bird's-Eye View Transform)

负责：
- 单应性矩阵加载与计算
- 像素坐标 → 世界坐标转换
- 全帧 BEV 透视变换
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Iterable


# ============================================================
# 单应性矩阵加载
# ============================================================

def load_homography(config_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 JSON 配置文件加载单应性矩阵。

    Parameters
    ----------
    config_path : str | Path
        JSON 文件路径，包含 image_points 和 world_points_m

    Returns
    -------
    h : (3, 3) float64 单应性矩阵
    img_pts : (N, 2) float64 图像点
    world_pts : (N, 2) float64 世界坐标点
    """
    config_path = Path(config_path)
    text = config_path.read_text(encoding='utf-8')

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text)

    img_pts = np.array(data['image_points'], dtype=np.float64)
    world_pts = np.array(data['world_points_m'], dtype=np.float64)

    if len(img_pts) < 4 or len(img_pts) != len(world_pts):
        raise ValueError(
            f"需要至少 4 对匹配点，实际: {len(img_pts)} 图像点, {len(world_pts)} 世界点"
        )

    # 检测重复或近重复点（会导致单应性矩阵退化）
    unique_img = set(map(tuple, img_pts.astype(int)))
    if len(unique_img) < len(img_pts):
        n_dup = len(img_pts) - len(unique_img)
        print(f"[WARN] 标定文件 {config_path.name} 存在 {n_dup} 个重复图像点，"
              f"单应性矩阵可能不稳定。请用 tools/calibrate_homography.py 重新标定。")
    else:
        # 检查近重复点（距离 < 20px）
        for i in range(len(img_pts)):
            for j in range(i + 1, len(img_pts)):
                dist = float(np.hypot(img_pts[i, 0] - img_pts[j, 0],
                                      img_pts[i, 1] - img_pts[j, 1]))
                if dist < 20:
                    print(f"[WARN] 标定文件 {config_path.name} 的点 {i+1} 和点 {j+1} "
                          f"距离仅 {dist:.0f}px，可能为误标。建议用 tools/calibrate_homography.py 重新标定。")

    h, status = cv2.findHomography(img_pts, world_pts, method=0)
    return h, img_pts, world_pts


# ============================================================
# 坐标转换
# ============================================================

def pixel_to_world(h: np.ndarray, xy: Iterable[float]) -> np.ndarray:
    """
    像素坐标 → 世界坐标（齐次变换）。

    Parameters
    ----------
    h : (3, 3) 单应性矩阵
    xy : (x, y) 像素坐标

    Returns
    -------
    (X, Y) 世界坐标 (m)
    """
    p = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    w = h @ p
    return w[:2] / w[2]


# ============================================================
# 全帧 BEV 变换
# ============================================================

def build_fullframe_view_homography(
    base_h: np.ndarray,
    frame_shape: tuple,
    out_wh: tuple[int, int],
    pad_ratio: float = 0.05,
) -> np.ndarray:
    """
    构建全帧 BEV 单应性矩阵。
    将整个帧的四角通过 base_h 变换后，缩放平移到输出画布。

    Returns
    -------
    h_full : (3, 3) 全帧变换矩阵
    """
    fh, fw = frame_shape[:2]
    ow, oh = out_wh

    src_corners = np.array([
        [0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    dst_raw = cv2.perspectiveTransform(src_corners, base_h).reshape(-1, 2)

    x_min, y_min = dst_raw.min(axis=0)
    x_max, y_max = dst_raw.max(axis=0)
    span_x = x_max - x_min
    span_y = y_max - y_min

    pad_x = int(ow * pad_ratio)
    pad_y = int(oh * pad_ratio)
    sx = (ow - 2 * pad_x) / max(span_x, 1e-6)
    sy = (oh - 2 * pad_y) / max(span_y, 1e-6)
    s = min(sx, sy)

    tx = pad_x - s * x_min
    ty = pad_y - s * y_min

    t_mat = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]], dtype=np.float64)
    return t_mat @ base_h
