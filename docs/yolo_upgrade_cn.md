# YOLO识别稳定性升级方案（交通路口场景）

## 1. 已在代码中启用的推理稳定性增强

- 启用 `--yolo-augment 1`（测试时增强，提升召回稳定性）
- 启用 `--yolo-half 1`（CUDA时使用FP16，提升吞吐并降低抖动）
- 调整 `--conf/--iou/--max-det` 到更稳组合
- 新增 `configs/bytetrack_stable.yaml`，增强多目标连续跟踪稳定性

## 2. 推荐开源数据集（车辆检测）

以下数据集适合提升复杂路口中车辆检测的泛化能力：

1. BDD100K（白天/夜晚/雨天/城市道路）
2. UA-DETRAC（交通监控摄像头视角，车辆密集）
3. VisDrone（高角度与小目标场景）
4. Cityscapes / Cityscapes 3D（城市道路高质量标注）

建议优先顺序：
- 第一阶段：BDD100K + UA-DETRAC
- 第二阶段：加入VisDrone补小目标
- 第三阶段：按你当前路口视频做少量人工标注微调（500~2000帧）

## 3. 训练配置建议（Ultralytics）

建议从 `yolo11m.pt` 继续微调：

```bash
yolo detect train \
  data=configs/vehicle_mix_dataset.yaml \
  model=yolo11m.pt \
  epochs=120 \
  imgsz=1280 \
  batch=8 \
  device=0 \
  optimizer=AdamW \
  lr0=0.003 \
  lrf=0.01 \
  cos_lr=True \
  mosaic=0.8 \
  mixup=0.1 \
  copy_paste=0.0 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=3.0 translate=0.10 scale=0.30 shear=1.0 perspective=0.0005 \
  fliplr=0.5 \
  close_mosaic=10 \
  patience=30
```

说明：
- 路口监控更重视小目标和拥挤场景，`imgsz=1280` 与较高 `max_det` 更稳。
- 若显存允许可换 `yolo11l.pt` 获得更高精度；若实时性优先保持 `yolo11m.pt`。

## 4. 数据集YAML模板

在项目中创建 `configs/vehicle_mix_dataset.yaml`，按你的本地路径填写：

```yaml
path: datasets/vehicle_mix
train: images/train
val: images/val

autodownload: false

names:
  0: car
  1: bus
  2: truck
  3: van
  4: motorcycle
  5: bicycle
```

## 5. 与当前风险分析管线对齐的训练建议

- 类别尽量与当前代码一致：`car/bus/truck/van/motorcycle/bicycle`
- 标注时保证框底部贴地（有利于BEV投影稳定）
- 对遮挡重叠车辆，标注要完整并保持跨帧一致
- 单独抽取你当前路口视频进行微调，收益通常最大

## 6. 快速A/B验证

建议做两组对比：

1. 现有权重 + 新推理参数（已落地）
2. 微调后权重 + 同样推理参数

重点观察：
- ID切换次数是否下降
- 漏检和误检是否下降
- Phi曲线是否更平稳且事件定位更一致
