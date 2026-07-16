# BEV 变形修复计划

## 根因

**问题 A：两套四角坐标**
- `h_mat`（世界坐标计算）← 标定文件 image_points ✅ 正确
- `H_bev`（BEV warp）← `_DEFAULT_QUAD` 硬编码 ❌ 和标定不一致

两个变换用了不同的四边形，用户标的方格和 warp 用的方格不是同一个。

**问题 B：BEV 输出比例不匹配**
- `bev_w:bev_h = 1600:1100 = 1.45:1`
- `world_w:world_h = 50:50 = 1:1`（以 ziyou 为例）
- ppm_x = 32, ppm_y = 22 → 不一致 → 车辆框/箭头变形

## 改动

### 文件：`pipeline/engine.py`

| 行号 | 改动 | 影响 |
|------|------|------|
| 39-43 | 删除 `_DEFAULT_QUAD`（不再需要） | 无 |
| 182-202 | H_bev 的 src_pts 改用 `img_pts`（从标定文件加载） | BEV warp 现在和世界坐标对齐 |
| 200 | dst_pts 保持 `(bev_w, bev_h)` 不变 | 无 |
| 89-90 | `bev_h` 改为自动计算：`int(bev_w * world_h / world_w)` | BEV 输出比例与真实世界一致 |
| 207-209 | ppm 使用 x 方向一致值：`ppm = bev_w / world_w` | 车辆框大小正确 |
| 186-201 | 保留 `bev_quad` CLI 参数作为 override（直接模式用）| 向后兼容 |

### 不受影响的模块

| 模块 | 原因 |
|------|------|
| `core/conflict.py` | 用 world_x/y，不依赖 BEV 像素 |
| `core/detector.py` | 纯像素空间检测，不依赖 BEV |
| `core/motion.py` | 速度计算用 `pixel_to_world`（`h_mat`），不依赖 `H_bev` |
| `core/phi.py` | 无依赖 |
| `analysis/*.py` | 用 world_x/y + heading_deg，不直接依赖 BEV |
| `visualization/*.py` | 只接收 BEV 图像作为底图，自适应尺寸 |
| `tests/*.py` | 无依赖 |
| `tools/*.py` | 无依赖 |
| `configs/*.json` | 无变化 |

### 受影响的模块（需要验证，但不应有问题）

| 模块 | 影响 | 验证方式 |
|------|------|----------|
| `pipeline/engine.py` 航向计算 | `H_bev` 改了，heading_deg 可能改变 | 跑测试+肉眼验证 |
| `pipeline/engine.py` 可视化 | BEV 图尺寸改变，`scale_c` 自适应 | 窗口显示检查 |
| `visualization/conflict_debug.py` | 底图改，叠加层自适应 | 跑 --backtest 验证 |

## 风险

- `bev_w` 默认为 1600，对于 50m×50m 世界和 1920×1080 画面，建议先设为 1100（正方形）
- `heading_deg` 变化可能影响冲突分析的 direction bin 分配 — 但这是**改进**（航向现在从正确 warp 计算）
