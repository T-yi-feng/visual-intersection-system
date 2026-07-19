"""
画布预分配池 —— 避免每帧 numpy 数组 malloc/free。

每帧创建多个 np.zeros((H,W,3)) 会产生多次内存分配 + GC 压力。
预分配画布在池中复用，用背景色填充替代重新分配。
"""

import numpy as np
from utils.theme import THEME


class CanvasPool:
    """预分配画布池"""

    def __init__(self):
        self._pool = {}

    def get(self, h: int, w: int, bg_color: tuple = None) -> np.ndarray:
        """
        获取 (h, w, 3) uint8 画布，用 bg_color 填充。
        如果 bg_color 为 None，使用 THEME['bg_canvas']。
        """
        if bg_color is None:
            bg_color = THEME["bg_canvas"]

        key = (h, w)
        if key not in self._pool:
            self._pool[key] = np.zeros((h, w, 3), dtype=np.uint8)

        canvas = self._pool[key]
        canvas[:] = bg_color
        return canvas

    def clear(self):
        """清空池（内存回收）"""
        self._pool.clear()

    def __len__(self):
        return len(self._pool)


# 全局单例
_canvas_pool = None


def get_canvas(h: int, w: int, bg_color: tuple = None) -> np.ndarray:
    """快捷获取预分配画布"""
    global _canvas_pool
    if _canvas_pool is None:
        _canvas_pool = CanvasPool()
    return _canvas_pool.get(h, w, bg_color)
