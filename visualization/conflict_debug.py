"""
冲突检测调试可视化模块 (Backtest Debug Visualization)

布局 (2×4 网格): 所有面板都以 BEV 为底图叠加，确保空间对齐。
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ BEV + 占用场  │ BEV + 冲突场  │ BEV + 归因   │ BEV + 所有方向 │
│  半透明叠加   │  热力叠加    │ 框+分数叠加  │  N/E/S/W 箭头  │
└──────────────┴──────────────┴──────────────┴──────────────┘
"""

import cv2
import numpy as np
from core.conflict import BIN_NAMES, DIRECTION_BINS
from utils.theme import THEME, attr_color


def _field_to_heatmap(field: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """将 float 场归一化到 0-255 并映射为热力图"""
    fmax = field.max()
    if fmax <= 0:
        return np.zeros((*field.shape, 3), dtype=np.uint8)
    normalized = (field / fmax * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, colormap)


def _draw_label(img: np.ndarray, text: str, org: tuple, scale: float = 0.45):
    """在图像上绘制白色标签（带黑色背景）"""
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 1, cv2.LINE_AA)


def _resize_to(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """缩放到目标尺寸"""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _overlay_grid_on_bev(
    bev_resized: np.ndarray,
    grid_field: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """将 64×64 网格场叠加到 BEV 底图上（保持空间对齐）"""
    h, w = bev_resized.shape[:2]
    grid_h, grid_w = grid_field.shape

    # 归一化并上色
    fmax = grid_field.max()
    if fmax <= 0:
        return bev_resized.copy()
    norm = (grid_field / fmax * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)

    # 缩放到 BEV 尺寸
    colored_resized = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)

    # 叠加（只在有值的区域叠加）
    mask = (norm > 0).astype(np.float32)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_3ch = np.stack([mask_resized] * 3, axis=-1)

    overlay = bev_resized.copy().astype(np.float32)
    layer = colored_resized.astype(np.float32)
    result = overlay * (1 - mask_3ch * alpha) + layer * mask_3ch * alpha
    return result.astype(np.uint8)


def _draw_grid_lines(img: np.ndarray, grid_size: int, color=None):
    """在图像上画网格线辅助对齐"""
    if color is None:
        color = THEME["border_panel"]
    h, w = img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size
    for i in range(1, grid_size):
        y = i * cell_h
        cv2.line(img, (0, y), (w, y), color, 1)
        x = i * cell_w
        cv2.line(img, (x, 0), (x, h), color, 1)
    return img


def render_conflict_debug_window(
    bev_frame: np.ndarray,
    conflict_result,
    current_meta: dict,
    grid_cfg,
    cell_size_m: float,
    panel_size: tuple[int, int] = (1600, 900),
) -> np.ndarray:
    """
    渲染冲突检测调试窗口。

    所有面板都以 BEV 为底图叠加，确保空间位置对齐。

    布局 (2×4):
    ┌──────────┬──────────┬──────────┬──────────┐
    │ BEV+占用  │ BEV+冲突  │ BEV+归因  │ BEV+方向  │
    │ 场叠加    │ 场叠加    │ 框叠加    │ 箭头叠加  │
    └──────────┴──────────┴──────────┴──────────┘
    """
    pw, ph = panel_size
    cell_w = pw // 4
    cell_h = ph // 2

    canvas = np.full((ph, pw, 3), THEME["bg_canvas"], dtype=np.uint8)

    if conflict_result is None:
        _draw_label(canvas, "No conflict result", (pw // 3, ph // 2))
        return canvas

    grid_size = conflict_result.grid_cfg.grid_size

    # 统一缩放 BEV 到面板尺寸
    bev_small = _resize_to(bev_frame, cell_w, cell_h)

    # ═══ Row 1 ═══

    # 1. BEV + 占用场叠加
    panel1 = _overlay_grid_on_bev(bev_small, conflict_result.density_field,
                                   alpha=0.6, colormap=cv2.COLORMAP_BONE)
    _draw_grid_lines(panel1, 8)
    n_veh = len(conflict_result.vehicles)
    _draw_label(panel1, f"Occupancy ({n_veh} veh)", (8, 20))
    canvas[0:cell_h, 0:cell_w] = panel1

    # 2. BEV + 冲突场叠加
    C = conflict_result.conflict_field
    panel2 = _overlay_grid_on_bev(bev_small, C, alpha=0.7, colormap=cv2.COLORMAP_HOT)
    _draw_grid_lines(panel2, 8)
    c_max = C.max()
    n_hot = int((C > 0).sum())
    _draw_label(panel2, f"Conflict (max={c_max:.3f}, hot={n_hot})", (8, 20))
    canvas[0:cell_h, cell_w:cell_w*2] = panel2

    # 3. BEV + 归因叠加（车辆框 + 分数）
    panel3 = bev_small.copy()
    influences = conflict_result.influences
    max_inf = max(influences) if influences and max(influences) > 0 else 1.0
    sx = cell_w / max(bev_frame.shape[1], 1)
    sy = cell_h / max(bev_frame.shape[0], 1)

    # 用 track_id 做 key 匹配归因分数，避免索引错位
    tid_to_inf = {}
    if influences and conflict_result.vehicles:
        for idx, v in enumerate(conflict_result.vehicles):
            tid_to_inf[v.get('track_id', idx)] = influences[idx]

    for tid, meta in current_meta.items():
        bx = int(meta.get('bev_x', 0) * sx)
        by = int(meta.get('bev_y', 0) * sy)
        if not (0 <= bx < cell_w and 0 <= by < cell_h):
            continue
        inf = tid_to_inf.get(tid, 0)
        if inf > 0 and max_inf > 0:
            inf_pct = (inf / max(max_inf, 1e-8)) * 100.0
            color = attr_color(inf_pct)
            sz = max(int(18 * sx), 6)
            cv2.rectangle(panel3, (bx-sz, by-sz), (bx+sz, by+sz), color, 2)
            cv2.putText(panel3, f"#{tid} {inf_pct:.1f}%", (bx+sz+2, by-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
        else:
            sz = max(int(12 * sx), 4)
            cv2.circle(panel3, (bx, by), sz, THEME["text_dim"], 1)

    n_conflict = sum(1 for inf in influences if inf > 0)
    _draw_label(panel3, f"Attribution ({n_conflict}/{len(influences)} conflict)", (8, 20))
    canvas[0:cell_h, cell_w*2:cell_w*3] = panel3

    # 4. BEV + 所有方向箭头叠加
    panel4 = bev_small.copy()
    # 为每个方向 bin 画不同颜色的箭头
    bin_colors = [
        (255,180,100), (180,255,100), (100,255,180), (100,255,255),
        (100,100,255), (180,100,255), (255,255,100), (255,180,180),
        (200,150,255), (150,200,255), (255,200,150), (150,255,200),
    ]
    layers = conflict_result.directional_fields
    R_fields = conflict_result.influence_fields

    for i, (tid, meta) in enumerate(current_meta.items()):
        bx = int(meta.get('bev_x', 0) * sx)
        by = int(meta.get('bev_y', 0) * sy)
        if not (0 <= bx < cell_w and 0 <= by < cell_h):
            continue
        heading = meta.get('heading_deg', 0)
        bin_size = 360.0 / DIRECTION_BINS
        bin_idx = int(heading % 360.0 / bin_size) % DIRECTION_BINS
        color = bin_colors[bin_idx % len(bin_colors)]

        # 画方向箭头
        rad = np.radians(heading)
        arrow_len = max(int(25 * sx), 10)
        hx = int(bx + arrow_len * np.cos(rad))
        hy = int(by - arrow_len * np.sin(rad))
        cv2.arrowedLine(panel4, (bx, by), (hx, hy), color, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(panel4, f"#{tid} {BIN_NAMES[bin_idx]}", (bx+arrow_len+4, by-4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    _draw_label(panel4, "Direction Arrows (all bins)", (8, 20))
    canvas[0:cell_h, cell_w*3:cell_w*4] = panel4

    # ═══ Row 2: 4 个主方向影响力场叠加到 BEV ═══
    main_bins = [0, 3, 6, 9]  # N, E, S, W (12-bin)
    main_names = ['N (0°)', 'E (90°)', 'S (180°)', 'W (270°)']
    main_colors = [(255,180,100), (100,255,180), (100,100,255), (255,255,100)]

    for col, (k, name, mcolor) in enumerate(zip(main_bins, main_names, main_colors)):
        x_start = col * cell_w
        y_off = cell_h

        if k < len(R_fields) and k < len(layers):
            # 影响力场叠加到 BEV
            R = R_fields[k]
            panel = _overlay_grid_on_bev(bev_small, R, alpha=0.6, colormap=cv2.COLORMAP_COOL)

            # 叠加该方向的车辆箭头（只画属于该 bin 的车）
            for i, (tid, meta) in enumerate(current_meta.items()):
                bx = int(meta.get('bev_x', 0) * sx)
                by = int(meta.get('bev_y', 0) * sy)
                if not (0 <= bx < cell_w and 0 <= by < cell_h):
                    continue
                heading = meta.get('heading_deg', 0)
                bin_size = 360.0 / DIRECTION_BINS
                bin_idx = int(heading % 360.0 / bin_size) % DIRECTION_BINS
                if bin_idx == k:
                    rad = np.radians(heading)
                    alen = max(int(20 * sx), 8)
                    hx = int(bx + alen * np.cos(rad))
                    hy = int(by - alen * np.sin(rad))
                    cv2.arrowedLine(panel, (bx, by), (hx, hy), mcolor, 2, cv2.LINE_AA, tipLength=0.3)

            n_in_bin = int(layers[k].sum())
            _draw_label(panel, f"{name} (bin{k}, {n_in_bin} veh)", (8, 20))
            canvas[y_off:y_off+cell_h, x_start:x_start+cell_w] = panel

    # 底部信息栏
    info_y = ph - 25
    info = (f"Vehicles: {n_veh} | Conflicting: {n_conflict} | "
            f"Pairs: {len(conflict_result.conflict_pairs)} | Grid: {grid_size}x{grid_size}")
    cv2.putText(canvas, info, (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, THEME["text_secondary"], 1, cv2.LINE_AA)

    return canvas
