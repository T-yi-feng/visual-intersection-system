"""
通用绘图工具模块

提供 CJK 文字渲染、颜色生成、箭头绘制等公共绘图函数。
"""

import cv2
import numpy as np
from pathlib import Path

# PIL 用于 CJK 文字渲染
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# CJK 字体候选路径
_CJK_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]

_CJK_FONT_CACHE = {}


def _contains_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _get_cjk_font(pixel_size: int):
    """加载/缓存 CJK 字体"""
    if not _PIL_AVAILABLE:
        return None

    if pixel_size in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[pixel_size]

    for font_path in _CJK_FONT_CANDIDATES:
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), pixel_size)
                _CJK_FONT_CACHE[pixel_size] = font
                return font
            except Exception:
                continue

    return None


def draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.5,
    thickness: int = 1,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
):
    """
    绘制带背景的文字（支持 CJK 通过 PIL 渲染）。
    """
    x, y = org

    if _contains_non_ascii(text) and _PIL_AVAILABLE:
        pixel_size = max(12, int(scale * 32))
        pil_font = _get_cjk_font(pixel_size)
        if pil_font:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            bbox = draw.textbbox((0, 0), text, font=pil_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x - 2, y - th - 2, x + tw + 2, y + 2], fill=bg_color)
            draw.text((x, y - th), text, fill=color, font=pil_font)
            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return

    # OpenCV 渲染（ASCII）
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


_ID_COLOR_CACHE: dict[int, tuple[int, int, int]] = {}
_ID_COLOR_MAX = 500


def id_color(track_id: int) -> tuple[int, int, int]:
    """为 track ID 生成确定性颜色（带缓存，上限 500 条）"""
    if track_id in _ID_COLOR_CACHE:
        return _ID_COLOR_CACHE[track_id]
    # 缓存满时清空（简单策略，track ID 不会回绕）
    if len(_ID_COLOR_CACHE) >= _ID_COLOR_MAX:
        _ID_COLOR_CACHE.clear()
    rng = np.random.RandomState(track_id * 7919 + 104729)
    color = tuple(int(c) for c in rng.randint(64, 255, 3))
    _ID_COLOR_CACHE[track_id] = color
    return color


def draw_radar_arrow(
    img: np.ndarray,
    p_tail: tuple[float, float],
    p_head: tuple[float, float],
    color: tuple[int, int, int],
    thickness: int = 1,
    head_len: float = 6.0,
    head_angle: float = 30.0,
    alpha: float = 0.6,
):
    """
    绘制雷达风格半透明箭头（线段 + V 形头部）。
    """
    overlay = img.copy()

    tail = (int(p_tail[0]), int(p_tail[1]))
    head = (int(p_head[0]), int(p_head[1]))

    cv2.line(overlay, tail, head, color, thickness, cv2.LINE_AA)

    # V 形头部
    dx = head[0] - tail[0]
    dy = head[1] - tail[1]
    length = max(np.hypot(dx, dy), 1e-6)
    ux, uy = dx / length, dy / length

    angle_rad = np.radians(head_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    for sign in [-1, 1]:
        rx = ux * cos_a - sign * uy * sin_a
        ry = ux * sin_a + sign * uy * cos_a
        tip = (int(head[0] - head_len * rx), int(head[1] - head_len * ry))
        cv2.line(overlay, head, tip, color, thickness, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def fit_for_display(
    img: np.ndarray,
    max_width: int = 1920,
    max_height: int = 1080,
    display_scale: float = 1.0,
) -> np.ndarray:
    """缩放图像以适应显示区域"""
    h, w = img.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1)) * display_scale
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def letterbox_to(
    img: np.ndarray,
    target_wh: tuple[int, int],
    bg_color: tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    """Letterbox 填充到目标尺寸"""
    tw, th = target_wh
    h, w = img.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)
    x_off = (tw - nw) // 2
    y_off = (th - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def enhance_realtime_clarity(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """轻量锐化"""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
