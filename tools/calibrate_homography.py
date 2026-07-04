
"""
快速标定工具 - 拖拽模式

加载现有标定点，在图上显示，用鼠标拖动调整位置，Enter 保存。
支持 u=撤销, r=重置原始, Esc=退出。

Usage
-----
python tools/calibrate_homography.py --site default
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_intersections_config(root: Path) -> dict:
    cfg_path = root / "configs" / "intersections.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    sites = data.get("sites")
    if not isinstance(sites, dict) or len(sites) == 0:
        raise RuntimeError("configs/intersections.json must contain non-empty 'sites'")
    default_site = str(data.get("default_site", ""))
    if default_site not in sites:
        default_site = next(iter(sites.keys()))
    return {"default_site": default_site, "sites": sites}


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path)


def ask_index(prompt: str, count: int, default_index: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [1-{count}] (default {default_index}): ").strip()
        if raw == "":
            return default_index
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw)
        print("Invalid selection.")


def ask_float(prompt: str, default_value: float) -> float:
    while True:
        raw = input(f"{prompt} (default {default_value:.3f}): ").strip()
        if raw == "":
            return float(default_value)
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("Invalid number, must be > 0.")


def choose_site(cfg: dict, explicit_site: str | None) -> str:
    sites = cfg["sites"]
    if explicit_site:
        if explicit_site not in sites:
            raise RuntimeError(f"Unknown site key: {explicit_site}")
        return explicit_site

    items = list(sites.items())
    print("Available intersections:")
    default_idx = 1
    for i, (k, site) in enumerate(items, start=1):
        name = site.get("display_name", k)
        print(f"  {i}. {k} - {name}")
        if k == cfg["default_site"]:
            default_idx = i
    idx = ask_index("Choose intersection", len(items), default_idx)
    return items[idx - 1][0]


def order_points_tl_tr_br_bl(points: list[tuple[int, int]]) -> np.ndarray:
    """将 4 个点排序为 TL, TR, BR, BL"""
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.asarray([tl, tr, br, bl], dtype=np.float32)


def build_world_points_from_edges(top_m: float, right_m: float,
                                   bottom_m: float, left_m: float) -> list[list[float]]:
    """从四边边长计算世界坐标四点"""
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([float(top_m), 0.0], dtype=np.float64)
    p4 = np.array([0.0, float(left_m)], dtype=np.float64)

    c0, c1 = p2, p4
    r0, r1 = float(right_m), float(bottom_m)
    dvec = c1 - c0
    d = float(np.hypot(dvec[0], dvec[1]))
    if d <= 1e-9:
        raise ValueError("Invalid edge settings: degenerate geometry")
    if d > (r0 + r1) + 1e-9 or d < abs(r0 - r1) - 1e-9:
        raise ValueError("Edge lengths are inconsistent (no quadrilateral solution).")

    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h = float(np.sqrt(max(0.0, r0 * r0 - a * a)))
    ex = dvec / d
    p = c0 + a * ex
    ey = np.array([-ex[1], ex[0]], dtype=np.float64)
    cand1 = p + h * ey
    cand2 = p - h * ey
    p3 = cand1 if cand1[1] >= cand2[1] else cand2

    return [
        [float(p1[0]), float(p1[1])],
        [float(p2[0]), float(p2[1])],
        [float(p3[0]), float(p3[1])],
        [float(p4[0]), float(p4[1])],
    ]


def read_default_edges(site: dict) -> tuple[float, float, float, float]:
    cal = site.get("calibration", {}) if isinstance(site.get("calibration", {}), dict) else {}
    edges = cal.get("world_edges_m", {}) if isinstance(cal.get("world_edges_m", {}), dict) else {}
    top = float(edges.get("top", 30.0))
    right = float(edges.get("right", 30.0))
    bottom = float(edges.get("bottom", top))
    left = float(edges.get("left", right))
    return top, right, bottom, left


# ============================================================
# 拖拽交互
# ============================================================

DRAG_RADIUS = 6  # 点的拖拽命中半径 (px)
POINT_COLOR = (0, 0, 255)       # 红色
LINE_COLOR = (0, 255, 255)      # 黄色
HINT_COLOR = (0, 200, 255)      # 橙黄


class DragCalibrator:
    """可拖拽的 4 点标定器"""

    def __init__(self, img: np.ndarray, points: list[tuple[int, int]]):
        self.base = img.copy()
        self.img = img.copy()
        self.points = list(points)       # 当前可拖拽点
        self.orig_points = list(points)  # 原始点（用于重置）
        self.dragging = -1               # 正在拖拽的点索引，-1=无
        self.hover_idx = -1              # 鼠标悬停的点索引
        self.img_h, self.img_w = img.shape[:2]

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        """钳制坐标到图像范围内"""
        return max(0, min(x, self.img_w - 1)), max(0, min(y, self.img_h - 1))

    def _redraw(self):
        """重绘全部"""
        self.img = self.base.copy()

        # 提示文字
        hint = ("Drag points to adjust. Keys: u=undo, r=reset, Enter=save, Esc=exit")
        cv2.putText(self.img, hint, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, HINT_COLOR, 2, cv2.LINE_AA)

        # 连线（TL→TR→BR→BL→TL）
        n = len(self.points)
        for i in range(n):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % n]
            cv2.line(self.img, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)

        # 检测点之间的距离，标记重合警告
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.points[i][0] - self.points[j][0]
                dy = self.points[i][1] - self.points[j][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < 30:
                    # 两点太近：画红色警告圈
                    mid = ((self.points[i][0] + self.points[j][0]) // 2,
                           (self.points[i][1] + self.points[j][1]) // 2)
                    cv2.circle(self.img, mid, 25, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(self.img, "TOO CLOSE!", (mid[0] - 40, mid[1] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 绘制 4 个点
        for i, (x, y) in enumerate(self.points):
            # 悬停/拖拽时高亮
            if i == self.hover_idx or i == self.dragging:
                cv2.circle(self.img, (x, y), DRAG_RADIUS + 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(self.img, (x, y), DRAG_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(self.img, (x, y), DRAG_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
            label = f"{i+1}: ({x},{y})"
            cv2.putText(self.img, label, (x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _find_point(self, mx: int, my: int) -> int:
        """查找鼠标附近的点索引，返回 -1 表示未命中"""
        best_idx = -1
        best_dist = DRAG_RADIUS + 1
        for i, (x, y) in enumerate(self.points):
            dist = ((mx - x) ** 2 + (my - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def on_mouse(self, event, x, y, _flags, _userdata):
        # OpenCV鼠标回调坐标已经是图像像素坐标（不受窗口缩放影响）
        # 直接使用，不需要额外缩放
        ix, iy = self._clamp(int(x), int(y))

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self._find_point(ix, iy)
            if idx >= 0:
                self.dragging = idx

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging >= 0:
                # 拖拽中：实时更新点位置（用图像坐标）
                self.points[self.dragging] = (ix, iy)
                self._redraw()
            else:
                # 悬停高亮
                idx = self._find_point(ix, iy)
                if idx != self.hover_idx:
                    self.hover_idx = idx
                    self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging >= 0:
                self.points[self.dragging] = (ix, iy)
                self.dragging = -1
                self._redraw()

    def run(self, window_name: str) -> list[tuple[int, int]] | None:
        """
        运行拖拽交互。

        Returns
        -------
        points : 4 个点的坐标列表，或 None（用户取消）
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

        # 初始窗口大小设为图像尺寸（避免自动缩放导致坐标偏移）
        cv2.resizeWindow(window_name, self.img_w, self.img_h)

        self._redraw()

        history = [list(self.points)]  # 撤销历史

        while True:
            cv2.imshow(window_name, self.img)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC - 取消
                print("Cancelled.")
                cv2.destroyAllWindows()
                return None

            if key in (13, 10):  # Enter - 保存
                if len(self.points) != 4:
                    print("Need exactly 4 points.")
                    continue
                # 打印拖拽后的原始坐标（排序前）
                print(f"\n--- 拖拽后原始坐标 (排序前) ---")
                for i, (x, y) in enumerate(self.points):
                    print(f"  点 {i+1}: ({x}, {y})")
                print(f"  图像尺寸: {self.img_w}x{self.img_h}")
                # 校验唯一性
                unique = set(self.points)
                if len(unique) < 4:
                    print("[ERROR] 4 个点中有重复，请调整后重试。")
                    continue
                # 校验近重复（30px 内）
                too_close = False
                for i in range(4):
                    for j in range(i + 1, 4):
                        dx = self.points[i][0] - self.points[j][0]
                        dy = self.points[i][1] - self.points[j][1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < 30:
                            print(f"[ERROR] 点 {i+1} 和点 {j+1} 距离仅 {dist:.0f}px，太近了，请拉开。")
                            too_close = True
                if too_close:
                    continue
                cv2.destroyAllWindows()
                return list(self.points)

            if key == ord("u"):  # 撤销
                if len(history) > 1:
                    history.pop()
                    self.points = list(history[-1])
                    self._redraw()
                    print(f"Undo → {self.points}")

            if key == ord("r"):  # 重置
                self.points = list(self.orig_points)
                history = [list(self.points)]
                self._redraw()
                print(f"Reset → {self.points}")

            # 记录历史（仅在非拖拽时）
            if self.dragging < 0:
                current = list(self.points)
                if not history or history[-1] != current:
                    history.append(current)

        cv2.destroyAllWindows()
        return None


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="快速标定工具 - 拖拽模式：加载现有标定点，拖动调整，Enter 保存"
    )
    parser.add_argument("--site", type=str, default="", help="Site key in configs/intersections.json")
    parser.add_argument("--top-m", type=float, default=0.0, help="Top edge length (m)")
    parser.add_argument("--right-m", type=float, default=0.0, help="Right edge length (m)")
    parser.add_argument("--bottom-m", type=float, default=0.0, help="Bottom edge length (m)")
    parser.add_argument("--left-m", type=float, default=0.0, help="Left edge length (m)")
    parser.add_argument("--no-edge-prompt", action="store_true", help="Do not prompt edge lengths interactively")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_intersections_config(root)
    site_key = choose_site(cfg, args.site.strip() or None)
    site = cfg["sites"][site_key]

    image_path = resolve_path(root, site.get("calibration_image", ""))
    if not image_path.exists():
        raise RuntimeError(f"Calibration image not found: {image_path}")

    out_homo = resolve_path(root, site.get("homography", ""))
    if str(out_homo).strip() == "":
        raise RuntimeError(f"Site '{site_key}' has no 'homography' path configured")

    # 读取现有标定点
    existing_points = None
    if out_homo.exists():
        try:
            data = json.loads(out_homo.read_text(encoding="utf-8"))
            pts = data.get("image_points", [])
            if len(pts) == 4:
                existing_points = [(int(p[0]), int(p[1])) for p in pts]
                print(f"Loaded existing points from {out_homo.name}: {existing_points}")
        except Exception:
            pass

    # 如果没有现有标定点，用默认四角
    if existing_points is None:
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        h, w = img.shape[:2]
        margin = 50
        existing_points = [
            (margin, margin),
            (w - margin, margin),
            (w - margin, h - margin),
            (margin, h - margin),
        ]
        print(f"No existing calibration found. Using default corners: {existing_points}")

    # 边长
    def_top, def_right, def_bottom, def_left = read_default_edges(site)
    top_m = float(args.top_m) if args.top_m > 0 else def_top
    right_m = float(args.right_m) if args.right_m > 0 else def_right
    bottom_m = float(args.bottom_m) if args.bottom_m > 0 else def_bottom
    left_m = float(args.left_m) if args.left_m > 0 else def_left

    if not args.no_edge_prompt:
        print("\nSet real-world edge lengths (meters):")
        top_m = ask_float("  Top edge", top_m)
        right_m = ask_float("  Right edge", right_m)
        bottom_m = ask_float("  Bottom edge", bottom_m)
        left_m = ask_float("  Left edge", left_m)

    world_points = build_world_points_from_edges(top_m, right_m, bottom_m, left_m)

    # 读取标定图
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # 启动拖拽标定
    calibrator = DragCalibrator(img, existing_points)
    win = f"Calibrate - {site_key} (drag to adjust)"
    result_points = calibrator.run(win)

    if result_points is None:
        print("No changes saved.")
        return

    # 拖拽模式下直接使用用户顺序（用户已按 TL→TR→BR→BL 拖好）
    # 不做排序——排序算法基于矩形假设，对梯形路口会出错
    print(f"--- 保存坐标 (按拖拽顺序 TL→TR→BR→BL) ---")
    names = ['TL', 'TR', 'BR', 'BL']
    for i, (x, y) in enumerate(result_points):
        print(f"  {names[i]}: ({x}, {y})")

    # 保存
    data = {
        "image_points": [[x, y] for x, y in result_points],
        "world_points_m": world_points,
        "comment": "Generated by tools/calibrate_homography.py (drag mode). Point order: TL, TR, BR, BL.",
        "calibration_image": str(image_path.relative_to(root)).replace('\\', '/'),
        "world_edges_m": {
            "top": float(top_m),
            "right": float(right_m),
            "bottom": float(bottom_m),
            "left": float(left_m),
        },
    }
    out_homo.parent.mkdir(parents=True, exist_ok=True)
    out_homo.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {out_homo}")
    print(f"image_points = {data['image_points']}")
    print(f"world_edges  = {data['world_edges_m']}")


if __name__ == "__main__":
    main()
