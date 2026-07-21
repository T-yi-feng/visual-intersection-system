"""
轻量级空间-外观记忆模块 (Lightweight Spatial-Appearance Memory)

解决遮挡后 track_id 丢失问题：
- 车辆被遮挡时存入记忆缓冲区
- 重新出现时用 IoU + 颜色相似度 + 运动连续性 匹配
- 恢复原 track_id，避免 ID 跳变

无额外模型推理开销，纯 NumPy/OpenCV 运算。
"""

import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class VehicleMemory:
    """单辆车的记忆快照"""
    track_id: int
    last_seen_frame: int
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)
    size: tuple  # (w, h)
    color_hist: Optional[np.ndarray] = None  # HSV 颜色直方图
    heading: float = 0.0
    label: str = 'car'


# ============================================================
# 外观特征提取
# ============================================================

def extract_color_histogram(
    frame: np.ndarray,
    bbox: tuple,
    hist_size: int = 32,
) -> np.ndarray:
    """
    提取车辆 ROI 的 HSV 颜色直方图。

    Parameters
    ----------
    frame : BGR 图像
    bbox : (x1, y1, x2, y2) 边界框
    hist_size : 每个通道的 bin 数

    Returns
    -------
    hist : flatten 后的直方图向量
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(hist_size * hist_size, dtype=np.float32)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros(hist_size * hist_size, dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        [hist_size, hist_size],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def color_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    计算两个颜色直方图的相似度（相关系数）。

    Returns
    -------
    similarity : float, 范围 [-1, 1]，越高越相似
    """
    if hist1 is None or hist2 is None:
        return 0.0
    if hist1.size == 0 or hist2.size == 0:
        return 0.0
    return float(cv2.compareHist(
        hist1.reshape(1, -1).astype(np.float32),
        hist2.reshape(1, -1).astype(np.float32),
        cv2.HISTCMP_CORREL,
    ))


# ============================================================
# 运动连续性
# ============================================================

def motion_continuity_score(
    prev_center: tuple,
    curr_center: tuple,
    max_dist: float = 100.0,
) -> float:
    """
    评估两点间的运动连续性分数。

    距离越近分数越高，范围 [0, 1]。
    """
    dx = curr_center[0] - prev_center[0]
    dy = curr_center[1] - prev_center[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return max(0.0, 1.0 - dist / max_dist)


# ============================================================
# 匹配打分
# ============================================================

def compute_match_score(
    candidate_bbox: tuple,
    candidate_hist: Optional[np.ndarray],
    candidate_center: tuple,
    memory: VehicleMemory,
    w_iou: float = 0.4,
    w_color: float = 0.3,
    w_motion: float = 0.3,
    max_motion_dist: float = 100.0,
) -> float:
    """
    计算检测候选与记忆条目的匹配分数。

    分数 = w_iou × IoU + w_color × 颜色相似度 + w_motion × 运动连续性
    """
    # IoU
    iou_score = _compute_iou(candidate_bbox, memory.bbox)

    # 颜色相似度
    color_score = color_similarity(candidate_hist, memory.color_hist)
    color_score = max(0.0, color_score)  # 负相关视为不匹配

    # 运动连续性
    motion_score = motion_continuity_score(
        memory.center, candidate_center, max_motion_dist
    )

    return w_iou * iou_score + w_color * color_score + w_motion * motion_score


def _compute_iou(box1: tuple, box2: tuple) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ============================================================
# 记忆缓冲区
# ============================================================

class MemoryBank:
    """
    记忆缓冲区：管理消失车辆的记忆，用于遮挡后 ID 恢复。

    Usage
    -----
    >>> bank = MemoryBank(max_age=5)
    >>> # 每帧调用
    >>> vehicles = bank.match_and_restore(vehicles, frame, frame_index)
    """

    def __init__(
        self,
        max_age: int = 5,
        match_threshold: float = 0.5,
        w_iou: float = 0.4,
        w_color: float = 0.3,
        w_motion: float = 0.3,
        max_motion_dist: float = 100.0,
    ):
        self.max_age = max_age
        self.match_threshold = match_threshold
        self.w_iou = w_iou
        self.w_color = w_color
        self.w_motion = w_motion
        self.max_motion_dist = max_motion_dist

        self.memories: dict[int, VehicleMemory] = {}
        self._active_ids: set[int] = set()

    def match_and_restore(
        self,
        vehicles: list[dict],
        frame: np.ndarray,
        frame_index: int,
    ) -> list[dict]:
        """
        对当前帧的检测结果执行记忆匹配。

        1. 记录当前活跃的 track_id
        2. 找出新分配的 track_id（不在上一帧活跃列表中）
        3. 对新 ID 尝试匹配记忆库
        4. 匹配成功 → 恢复原 ID
        5. 更新记忆库（活跃的存入，消失的保留）

        Parameters
        ----------
        vehicles : 当前帧的检测结果列表
        frame : 当前帧图像（用于提取外观特征）
        frame_index : 当前帧号

        Returns
        -------
        vehicles : 更新后的检测结果（ID 可能被恢复）
        """
        if not vehicles:
            self._active_ids = set()
            self._expire_memories(frame_index)
            return vehicles

        current_ids = {v['track_id'] for v in vehicles}
        new_ids = current_ids - self._active_ids

        # 对新 ID 尝试匹配记忆
        if new_ids and self.memories:
            for v in vehicles:
                if v['track_id'] not in new_ids:
                    continue

                bbox = v['bbox']
                center = v['center']
                hist = extract_color_histogram(frame, bbox)

                # 在记忆库中搜索最佳匹配
                best_score = 0.0
                best_tid = None
                for mem_tid, mem in self.memories.items():
                    if mem_tid in current_ids:
                        continue  # 已被其他检测占用
                    if frame_index - mem.last_seen_frame > self.max_age:
                        continue  # 记忆过期

                    score = compute_match_score(
                        bbox, hist, center, mem,
                        w_iou=self.w_iou,
                        w_color=self.w_color,
                        w_motion=self.w_motion,
                        max_motion_dist=self.max_motion_dist,
                    )
                    if score > best_score:
                        best_score = score
                        best_tid = mem_tid

                # 匹配成功 → 恢复 ID
                if best_tid is not None and best_score >= self.match_threshold:
                    old_tid = v['track_id']
                    # 恢复 heading（遮挡前的方向）
                    mem = self.memories.get(best_tid)
                    if mem is not None:
                        v['heading_deg'] = mem.heading
                    v['track_id'] = best_tid
                    # 删除旧的记忆条目
                    self.memories.pop(best_tid, None)
                    # 从当前检测中移除旧 ID 的条目（如果有的话）
                    # （这里 old_tid 是新分配的，不会在 vehicles 中有重复）

        # 更新记忆库
        self._active_ids = current_ids

        # 将当前活跃车辆存入记忆（供未来帧使用）
        for v in vehicles:
            bbox = v['bbox']
            center = v['center']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            hist = extract_color_histogram(frame, bbox)
            heading = v.get('heading_deg', 0.0)

            self.memories[v['track_id']] = VehicleMemory(
                track_id=v['track_id'],
                last_seen_frame=frame_index,
                bbox=bbox,
                center=center,
                size=(w, h),
                color_hist=hist,
                heading=heading,
                label=v.get('label', 'car'),
            )

        # 清理过期记忆
        self._expire_memories(frame_index)

        return vehicles

    def _expire_memories(self, frame_index: int):
        """清除超过 max_age 帧未出现的记忆"""
        expired = [
            tid for tid, mem in self.memories.items()
            if frame_index - mem.last_seen_frame > self.max_age
        ]
        for tid in expired:
            del self.memories[tid]

    def clear(self):
        """清空记忆库"""
        self.memories.clear()
        self._active_ids.clear()

    @property
    def size(self) -> int:
        """当前记忆库中的条目数"""
        return len(self.memories)
