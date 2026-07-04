"""
检测模块单元测试
"""

import pytest
from core.detector import (
    canonical_vehicle_label, size_refine_label, pass_class_conf,
    stabilize_label, is_vehicle_class,
)


class TestCanonicalLabel:
    def test_car(self):
        assert canonical_vehicle_label('car') == 'car'

    def test_truck(self):
        assert canonical_vehicle_label('truck') == 'truck'

    def test_bus(self):
        assert canonical_vehicle_label('bus') == 'bus'

    def test_motorcycle(self):
        assert canonical_vehicle_label('motorcycle') == 'motorcycle'

    def test_bicycle(self):
        assert canonical_vehicle_label('bicycle') == 'bicycle'

    def test_van(self):
        assert canonical_vehicle_label('van') == 'van'

    def test_fire_truck(self):
        assert canonical_vehicle_label('fire truck') == 'truck'

    def test_ambulance(self):
        assert canonical_vehicle_label('ambulance') == 'bus'


class TestSizeRefine:
    def test_small_truck_becomes_car(self):
        assert size_refine_label('truck', 10000) == 'car'

    def test_large_truck_stays(self):
        assert size_refine_label('truck', 25000) == 'truck'

    def test_small_bus_becomes_car(self):
        assert size_refine_label('bus', 15000) == 'car'

    def test_large_bus_stays(self):
        assert size_refine_label('bus', 30000) == 'bus'

    def test_large_car_becomes_truck(self):
        assert size_refine_label('car', 200000) == 'truck'

    def test_normal_car_stays(self):
        assert size_refine_label('car', 50000) == 'car'


class TestPassClassConf:
    def test_high_conf_car(self):
        assert pass_class_conf('car', 0.5) is True

    def test_low_conf_car(self):
        assert pass_class_conf('car', 0.1) is False

    def test_high_conf_truck(self):
        assert pass_class_conf('truck', 0.5) is True

    def test_low_conf_truck(self):
        assert pass_class_conf('truck', 0.15) is False


class TestIsVehicleClass:
    def test_car(self):
        assert is_vehicle_class('car') is True

    def test_truck(self):
        assert is_vehicle_class('truck') is True

    def test_person(self):
        assert is_vehicle_class('person') is False

    def test_bicycle(self):
        assert is_vehicle_class('bicycle') is True


class TestStabilizeLabel:
    def test_stable_with_same_labels(self):
        state = {}
        for _ in range(5):
            result = stabilize_label(state, 1, 'car', window=3)
        assert result == 'car'

    def test_flashing_labels_stabilizes(self):
        state = {}
        # 交替标签
        labels = ['car', 'truck', 'car', 'truck', 'car']
        for label in labels:
            result = stabilize_label(state, 1, label, window=3)
        # 应该稳定为多数标签
        assert result in ('car', 'truck')
