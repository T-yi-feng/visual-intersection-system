"""
数据面板渲染模块 — 紧凑卡片布局
"""

import cv2
import numpy as np
from utils.theme import THEME, phi_color, phi_label_en


def render_data_panel(size_hw, phi, motion_stats, conflict_count,
                      conflict_total, influences, conflict_result,
                      tid_to_inf, max_inf, site_name,
                      root_cause_pct=None):
    h, w = size_hw
    panel = np.full((h, w, 3), THEME["bg_canvas"], dtype=np.uint8)
    cx, cy = 8, 10

    # ── Card 1: Phi ──
    cw, ch = w - 16, 54
    _card(panel, cx, cy, cw, ch, accent=phi_color(phi))
    _text(panel, f"Phi  {phi:.3f}", cx + 12, cy + 22, phi_color(phi), 0.65)
    _text(panel, phi_label_en(phi), cx + 12, cy + 43, THEME["text_secondary"], 0.38)
    cy += ch + 6

    # ── Card 2: Stats ──
    cw2, ch2 = w - 16, 76
    _card(panel, cx, cy, cw2, ch2, label="Traffic")
    rows = [
        f"Vehicles  {motion_stats['total_count']}       Moving  {motion_stats['moving_count']}",
        f"Stationary  {motion_stats['stationary_count']}       Parked  {motion_stats['parked_count']}",
        f"Avg Speed  {motion_stats['avg_speed_mps']:.1f} m/s",
    ]
    for i, r in enumerate(rows):
        _text(panel, r, cx + 10, cy + 22 + i * 18, THEME["text_primary"], 0.38)
    cy += ch2 + 6

    # ── Card 3: Influence vs Root Cause Top-3 ──
    if conflict_count > 0 and influences and conflict_result:
        ranked = conflict_result.get_vehicles_ranked_by_influence()
        n = min(3, len(ranked))
        ch3 = 24 + n * 28 + 10
        _card(panel, cx, cy, cw2, ch3, label="Influence  /  Root Cause")
        _text(panel, "Vehicle   Inf%   Cause%", cx + 10, cy + 20, THEME["text_dim"], 0.28)
        for ri in range(n):
            idx, inf_val, v = ranked[ri]
            if inf_val <= 0: continue
            tid = v.get('track_id', idx)
            lbl = v.get('label', 'car')
            inf_pct = (inf_val / max(max_inf, 1e-8)) * 100.0
            rc_pct = root_cause_pct[idx] if (root_cause_pct is not None and idx < len(root_cause_pct)) else 0.0

            # Influence bar
            iy = cy + 24 + ri * 28
            ibx, ibw = cx + 14, int((cw2 - 110) * min(inf_pct / 20.0, 1.0))
            rbx = cx + 14 + (cw2 - 110) // 2 + 10
            rbw = int((cw2 - 110) * min(rc_pct / 20.0, 1.0))

            cv2.rectangle(panel, (ibx, iy), (ibx + cw2 - 110, iy + 10), THEME["border_panel"], 1)
            if ibw > 0:
                cv2.rectangle(panel, (ibx, iy), (ibx + ibw, iy + 10), (50, 140, 230), -1)  # 橙色 = Influence

            cv2.rectangle(panel, (rbx, iy), (rbx + cw2 - 110, iy + 10), THEME["border_panel"], 1)
            if rbw > 0:
                cv2.rectangle(panel, (rbx, iy), (rbx + rbw, iy + 10), (50, 50, 255), -1)  # 亮红 = Root Cause

            _text(panel, f"#{tid}", ibx, iy - 2, THEME["text_primary"], 0.28)
            _text(panel, f"Inf", ibx + cw2 - 102, iy - 2, THEME["text_dim"], 0.25)
            if rc_pct > 0:
                _text(panel, f"Cause", rbx + cw2 - 102, iy - 2, THEME["text_dim"], 0.25)

            # ID vehicle type
            _text(panel, f"#{tid} {lbl}", ibx, iy + 18, THEME["text_dim"], 0.26)
    else:
        ch3 = 36
        _card(panel, cx, cy, cw2, ch3, label="Conflict")
        _text(panel, "No conflict detected", cx + 12, cy + 22, THEME["text_dim"], 0.35)
    cy += ch3 + 6

    # ── Card 4: System ──
    ch4 = 36
    _card(panel, cx, cy, cw2, ch4, label="System")
    _text(panel, f"{site_name}    Interwoven {conflict_count}/{conflict_total}",
          cx + 10, cy + 22, THEME["text_secondary"], 0.35)

    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), THEME["border_panel"], 1)
    return panel


def _card(panel, x, y, w, h, label="", accent=None):
    roi = panel[y:y + h, x:x + w]
    ov = roi.copy(); ov[:] = THEME["bg_panel"]
    cv2.addWeighted(ov, 0.70, roi, 0.30, 0, dst=roi)
    cv2.rectangle(panel, (x, y), (x + w, y + h), THEME["border_panel"], 1)
    if accent:
        cv2.line(panel, (x + 2, y + 8), (x + 2, y + h - 8), accent, 2, cv2.LINE_AA)
    if label:
        _text(panel, label, x + 10, y + 15, THEME["text_dim"], 0.30)


def _text(panel, txt, x, y, color, scale):
    cv2.putText(panel, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
