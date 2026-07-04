"""
配置加载模块

负责加载车辆尺寸、箭头样式等配置文件。
"""

import json
import numpy as np
from pathlib import Path


def load_vehicle_size_map(path: str | Path) -> dict:
    """
    加载车辆尺寸配置。

    Returns
    -------
    dict[label -> {'length_m': float, 'width_m': float}]
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    result = {}
    if isinstance(data, dict):
        # 格式: {"car": {"length_m": 4.0, "width_m": 1.6}, ...}
        for label, info in data.items():
            result[label] = {
                'length_m': info.get('length_m', 4.0),
                'width_m': info.get('width_m', 1.6),
            }
    elif isinstance(data, list):
        # 格式: [{"label": "car", "length_m": 4.0, "width_m": 1.6}, ...]
        for entry in data:
            label = entry.get('label', entry.get('type', ''))
            result[label] = {
                'length_m': entry.get('length_m', 4.0),
                'width_m': entry.get('width_m', 1.6),
            }
    return result


def load_vehicle_arrow_style_map(path: str | Path) -> dict:
    """
    加载车辆箭头样式配置。

    Returns
    -------
    dict[label -> {'length_scale': float, 'alpha': float, 'thickness': float}]
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    result = {}
    if isinstance(data, dict):
        for label, info in data.items():
            result[label] = {
                'length_scale': info.get('length_scale', 3.0),
                'alpha': info.get('alpha', 0.55),
                'thickness': info.get('thickness', 1.5),
            }
    elif isinstance(data, list):
        for entry in data:
            label = entry.get('label', entry.get('type', ''))
            result[label] = {
                'length_scale': entry.get('length_scale', 3.0),
                'alpha': entry.get('alpha', 0.55),
                'thickness': entry.get('thickness', 1.5),
            }
    return result


def load_intersections(config_path: str | Path) -> dict:
    """
    加载路口配置。

    Returns
    -------
    dict with keys 'sites' and 'default_site'
    """
    data = json.loads(Path(config_path).read_text(encoding='utf-8'))
    sites = data.get('sites', {})
    if not sites:
        raise ValueError("intersections.json 中没有定义任何路口")
    default_site = data.get('default_site', list(sites.keys())[0])
    return {'sites': sites, 'default_site': default_site}


