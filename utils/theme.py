"""
统一色彩体系 —— 莫兰迪学术暗色主题

所有可视化模块从这里取色，不再硬编码 BGR 值。
设计目标：投影仪可读、长时间观看不刺眼、色盲友好。
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# 色板定义 (BGR 格式，供 OpenCV 直接使用)
# ═══════════════════════════════════════════════════════════════

THEME = {
    # ── 画布/面板 ──
    "bg_canvas":     (28, 24, 24),    # 全局最深背景
    "bg_panel":      (38, 32, 32),    # 卡片/面板底色
    "border_panel":  (62, 55, 55),    # 面板边框/分隔线

    # ── 文字层级 ──
    "text_primary":   (220, 220, 220),  # 标题、重要数值
    "text_secondary": (150, 150, 150),  # 注释、单位、标签
    "text_dim":       (100, 100, 100),  # 极弱文字

    # ── Phi 拥堵等级 (蓝→黄→橙→红渐变) ──
    "phi_low":        (160, 180, 120),  # Φ < 0.30  畅通 (莫兰迪绿)
    "phi_moderate":   (120, 185, 200),  # Φ 0.30-0.55 轻度 (莫兰迪黄)
    "phi_high":       (100, 150, 210),  # Φ 0.55-0.75 中度 (莫兰迪橙)
    "phi_critical":   (70, 80, 200),    # Φ > 0.75  严重 (莫兰迪红)

    # ── 归因等级 ──
    "attr_low":       (100, 160, 100),  # influence < 5%
    "attr_mid":       (80, 200, 180),   # 5-15%
    "attr_high":      (220, 140, 80),   # > 15%

    # ── 功能色 ──
    "accent":         (80, 160, 200),   # 强调/当前值/告警 (莫兰迪金)
    "info":           (200, 180, 160),  # 速度/统计信息 (莫兰迪蓝灰)
    "success":        (120, 180, 130),  # 正常状态
    "warning":        (90, 160, 200),   # 注意
    "danger":         (70, 80, 200),    # 危险/高冲突 (同 phi_critical)

    # ── 热力图 ──
    "heatmap_low":    (40, 30, 30),     # 零冲突区
    "heatmap_high":   (80, 50, 200),    # 高冲突区 (用于叠加混合)
}


# ═══════════════════════════════════════════════════════════════
# 颜色工具函数
# ═══════════════════════════════════════════════════════════════

def lerp_bgr(a: tuple, b: tuple, t: float) -> tuple:
    """BGR 颜色线性插值。t ∈ [0,1]，纯算术，零内存分配。"""
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def phi_color(phi: float) -> tuple:
    """
    Phi 拥堵指数 → BGR 连续渐变。

    使用四段莫兰迪色调，而非 HSV 彩虹色：
    0.00-0.30: 绿  → 黄   (phi_low → phi_moderate)
    0.30-0.55: 黄  → 橙   (phi_moderate → phi_high)
    0.55-0.75: 橙  → 红   (phi_high → phi_critical)
    0.75-1.00: 红          (phi_critical, 饱和)
    """
    if phi < 0.30:
        return lerp_bgr(THEME["phi_low"], THEME["phi_moderate"], phi / 0.30)
    elif phi < 0.55:
        return lerp_bgr(THEME["phi_moderate"], THEME["phi_high"], (phi - 0.30) / 0.25)
    elif phi < 0.75:
        return lerp_bgr(THEME["phi_high"], THEME["phi_critical"], (phi - 0.55) / 0.20)
    else:
        return THEME["phi_critical"]


def attr_color(influence_pct: float) -> tuple:
    """
    归因百分比 → BGR 颜色。
    20% 封顶（超过 20% 的极端归因保持最高色）。
    """
    t = min(abs(influence_pct) / 20.0, 1.0)
    if t < 0.5:
        return lerp_bgr(THEME["attr_low"], THEME["attr_mid"], t * 2.0)
    else:
        return lerp_bgr(THEME["attr_mid"], THEME["attr_high"], (t - 0.5) * 2.0)


def heatmap_color(intensity: float) -> tuple:
    """
    冲突场强度 → 热力图 BGR 颜色。
    莫兰迪版 VIRIDIS-like：深灰→蓝→绿→黄→红。
    intensity ∈ [0, 1]
    """
    t = max(0.0, min(1.0, intensity))
    # 4-stop gradient: dark → blue → green → yellow → red
    if t < 0.25:
        return lerp_bgr(THEME["heatmap_low"], (100, 60, 30), t / 0.25)
    elif t < 0.50:
        return lerp_bgr((100, 60, 30), (80, 140, 70), (t - 0.25) / 0.25)
    elif t < 0.75:
        return lerp_bgr((80, 140, 70), (60, 200, 180), (t - 0.50) / 0.25)
    else:
        return lerp_bgr((60, 200, 180), THEME["heatmap_high"], (t - 0.75) / 0.25)


def build_heatmap_lut(colormap: int = None) -> np.ndarray:
    """
    构建 256 级热力图颜色查找表 (LUT)。
    用于快速 field → BGR 转换：cv2.LUT(field_u8, lut)。

    如果指定 colormap (如 cv2.COLORMAP_VIRIDIS)，使用 OpenCV 内置；
    否则使用自定义莫兰迪热力色。
    """
    if colormap is not None:
        grad = np.arange(256, dtype=np.uint8).reshape(256, 1)
        return cv2.applyColorMap(grad, colormap)
    else:
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            lut[i, 0] = heatmap_color(i / 255.0)
        return lut


# 预构建 VIRIDIS LUT（色盲友好、感知均匀）
# 延迟初始化，避免在 import 时依赖 cv2
_HEATMAP_LUT = None
_HEATMAP_LUT_VIRIDIS = None


def get_heatmap_lut(use_viridis: bool = True):
    """获取热力图 LUT（惰性初始化 + 缓存）"""
    global _HEATMAP_LUT, _HEATMAP_LUT_VIRIDIS
    import cv2
    if use_viridis:
        if _HEATMAP_LUT_VIRIDIS is None:
            grad = np.arange(256, dtype=np.uint8).reshape(256, 1)
            _HEATMAP_LUT_VIRIDIS = cv2.applyColorMap(grad, cv2.COLORMAP_VIRIDIS)
        return _HEATMAP_LUT_VIRIDIS
    else:
        if _HEATMAP_LUT is None:
            _HEATMAP_LUT = build_heatmap_lut()
        return _HEATMAP_LUT


# ═══════════════════════════════════════════════════════════════
# Phi 等级标签
# ═══════════════════════════════════════════════════════════════

def phi_label(phi: float) -> str:
    """Phi → 中文拥堵等级"""
    if phi < 0.30:
        return "畅通"
    elif phi < 0.55:
        return "轻度拥堵"
    elif phi < 0.75:
        return "中度拥堵"
    else:
        return "严重拥堵"


def phi_label_en(phi: float) -> str:
    """Phi → English congestion level"""
    if phi < 0.30:
        return "Free Flow"
    elif phi < 0.55:
        return "Moderate"
    elif phi < 0.75:
        return "Heavy"
    else:
        return "Severe"
