"""
检测与跟踪模块 (Detection & Tracking)

负责：
- YOLO 模型加载与推理
- ByteTrack 多目标跟踪
- 车辆分类与标签稳定化
- 置信度过滤
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from math import atan2, hypot, pi
from collections import Counter

from utils.drawing import id_color

from ultralytics import YOLO


# ============================================================
# 常量
# ============================================================

COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

VEHICLE_KEYWORDS = {
    'car', 'truck', 'bus', 'van',
    'motorcycle', 'motorbike', 'bicycle',
    'ambulance', 'fire truck', 'police car',
}


# ============================================================
# 标签处理
# ============================================================

def is_vehicle_class(name: str) -> bool:
    """检查是否为车辆类别"""
    name_lower = name.lower()
    return any(kw in name_lower for kw in VEHICLE_KEYWORDS)


def canonical_vehicle_label(name: str) -> str:
    """将原始检测标签映射为标准车辆类型"""
    name_lower = name.lower()
    if 'truck' in name_lower or 'fire truck' in name_lower:
        return 'truck'
    if 'bus' in name_lower or 'ambulance' in name_lower:
        return 'bus'
    if 'van' in name_lower:
        return 'van'
    if 'motor' in name_lower:
        return 'motorcycle'
    if 'bicycle' in name_lower or 'bike' in name_lower:
        return 'bicycle'
    return 'car'


def stabilize_label(
    track_state: dict,
    track_id: int,
    label: str,
    window: int = 12,
) -> str:
    """
    标签稳定化：滑动窗口投票，减少帧间标签闪烁。
    """
    key = f'label_hist_{track_id}'
    if key not in track_state:
        track_state[key] = []

    hist = track_state[key]
    hist.append(label)
    if len(hist) > window:
        hist.pop(0)

    # 投票
    counts = Counter(hist)
    return max(counts, key=counts.get)


def cleanup_track_state(track_state: dict, active_ids: set[int]):
    """
    清理不活跃车辆的标签历史，防止内存泄漏。

    Parameters
    ----------
    track_state : dict - stabilize_label 使用的状态字典
    active_ids : set - 当前帧活跃的 track_id 集合
    """
    stale_keys = [k for k in track_state
                  if k.startswith('label_hist_')]
    for k in stale_keys:
        try:
            tid = int(k.split('_', 2)[-1])
        except (ValueError, IndexError):
            continue
        if tid not in active_ids:
            del track_state[k]


def size_refine_label(
    label: str,
    bbox_area: float,
    truck_min_area: float = 18000,
    bus_min_area: float = 22000,
    car_to_truck_area: float = 140000,
) -> str:
    """
    基于面积的标签修正。
    小面积的 truck/bus → car；大面积的 car → truck。
    """
    if label == 'truck' and bbox_area < truck_min_area:
        return 'car'
    if label == 'bus' and bbox_area < bus_min_area:
        return 'car'
    if label == 'car' and bbox_area > car_to_truck_area:
        return 'truck'
    return label


def pass_class_conf(
    label: str,
    conf: float,
    conf_car: float = 0.20,
    conf_truck: float = 0.28,
    conf_bus: float = 0.28,
    conf_van: float = 0.24,
    conf_other: float = 0.22,
) -> bool:
    """按类别置信度过滤"""
    thresholds = {
        'car': conf_car,
        'truck': conf_truck,
        'bus': conf_bus,
        'van': conf_van,
    }
    return conf >= thresholds.get(label, conf_other)


# ============================================================
# 检测与跟踪
# ============================================================

class VehicleDetector:
    """
    车辆检测器：封装 YOLO + ByteTrack。

    Usage
    -----
    >>> detector = VehicleDetector('yolo11m.pt', tracker_config='configs/bytetrack_stable.yaml')
    >>> result = detector.detect_frame(frame, timestamp=1.5)
    >>> for v in result.vehicles:
    ...     print(v['track_id'], v['label'], v['center'], v['conf'])
    """

    def __init__(
        self,
        model_path: str = 'yolo11m.pt',
        tracker_config: str = 'configs/bytetrack_stable.yaml',
        conf_car: float = 0.12,
        conf_truck: float = 0.15,
        conf_bus: float = 0.15,
        conf_van: float = 0.12,
        conf_other: float = 0.12,
        label_stabilize_window: int = 6,
    ):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_thresholds = {
            'conf_car': conf_car, 'conf_truck': conf_truck,
            'conf_bus': conf_bus, 'conf_van': conf_van,
        }
        self.conf_other = conf_other
        self.label_window = label_stabilize_window

        # 自动检测 CUDA，优先使用 GPU 加速
        import torch
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model.to('cuda')
            print(f"[INFO] YOLO 使用 GPU 加速: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("[INFO] YOLO 使用 CPU 推理（未检测到 CUDA）")

        # 跟踪状态
        self.track_state = {}
        self.next_track_id = 0

    def detect_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        imgsz: int = 1280,
        conf: float = 0.22,
        iou: float = 0.40,
        annotate: bool = True,
    ) -> 'DetectionResult':
        """
        对单帧运行检测+跟踪。

        Parameters
        ----------
        annotate : bool
            是否创建标注帧。关闭可省去每帧的 frame.copy() 开销。

        Returns
        -------
        DetectionResult
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            classes=COCO_VEHICLE_CLASS_IDS,
            device=self.device,
            max_det=600,
        )

        vehicles = []
        annotated = frame.copy() if annotate else None

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, c, cls_id in zip(boxes, ids, confs, cls_ids):
                    x1, y1, x2, y2 = box
                    label = self.model.names.get(cls_id, 'car')
                    label = canonical_vehicle_label(label)

                    # 面积修正
                    area = (x2 - x1) * (y2 - y1)
                    label = size_refine_label(label, area)

                    # 标签稳定化
                    label = stabilize_label(self.track_state, track_id, label, self.label_window)

                    # 置信度过滤
                    if not pass_class_conf(label, c, **self.conf_thresholds, conf_other=self.conf_other):
                        continue

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    vehicles.append({
                        'track_id': int(track_id),
                        'label': label,
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'center': (float(cx), float(cy)),
                        'conf': float(c),
                        'area': float(area),
                        'timestamp': timestamp,
                    })

                    # 绘制检测框和标签（仅在需要标注时）
                    if annotate:
                        color = id_color(int(track_id))
                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))
                        cv2.rectangle(annotated, pt1, pt2, color, 2)
                        text = f"{label} #{track_id} {c:.2f}"
                        cv2.putText(annotated, text, (int(x1), int(y1) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return DetectionResult(
            vehicles=vehicles,
            annotated_frame=annotated,
            raw_results=results,
            timestamp=timestamp,
        )

    def reset(self):
        """重置跟踪状态"""
        self.track_state.clear()
        try:
            if hasattr(self.model, 'predictor') and hasattr(self.model.predictor, 'trackers'):
                for tracker in self.model.predictor.trackers:
                    if hasattr(tracker, 'reset'):
                        tracker.reset()
        except (AttributeError, IndexError):
            pass  # ultralytics 内部 API 变化时静默降级


class DetectionResult:
    """检测结果"""

    def __init__(
        self,
        vehicles: list[dict],
        annotated_frame: np.ndarray,
        raw_results=None,
        timestamp: float = 0.0,
    ):
        self.vehicles = vehicles
        self.annotated_frame = annotated_frame
        self.raw_results = raw_results
        self.timestamp = timestamp

    @property
    def count(self) -> int:
        return len(self.vehicles)

    @property
    def track_ids(self) -> list[int]:
        return [v['track_id'] for v in self.vehicles]

    def get_by_track_id(self, track_id: int) -> Optional[dict]:
        for v in self.vehicles:
            if v['track_id'] == track_id:
                return v
        return None


# ============================================================
# 航向估计
# ============================================================

