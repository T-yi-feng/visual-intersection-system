# 视觉识别交叉口信息采集系统

> **Visual Recognition Intersection Information Collection System**  
> 基于方向场卷积的交通冲突检测与拥堵归因分析平台

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心算法：方向场卷积冲突检测](#3-核心算法方向场卷积冲突检测)
4. [拥堵指数 Phi](#4-拥堵指数-phi)
5. [因果溯源：水滴传播算法](#5-因果溯源水滴传播算法)
6. [可视化系统](#6-可视化系统)
7. [配置与参数](#7-配置与参数)
8. [快速启动](#8-快速启动)
9. [标定工具](#9-标定工具)
10. [教学演示模块](#10-教学演示模块)
11. [项目文件结构](#11-项目文件结构)
12. [依赖与环境](#12-依赖与环境)
13. [常见问题](#13-常见问题)

---

## 1. 项目概述

### 1.1 项目目标

从固定交通监控摄像头获取视频流，自动检测并跟踪车辆，通过鸟瞰图（BEV）透视变换将画面映射到世界坐标系，利用方向场卷积算法实时分析车辆间的交织冲突与拥堵程度，并量化每辆车对拥堵的贡献与因果根因。

### 1.2 核心功能

| 功能 | 说明 |
|------|------|
| **车辆检测与跟踪** | YOLO11m + ByteTrack，支持 6 类车型（car/truck/bus/van/motorcycle/bicycle） |
| **BEV 透视变换** | 基于四角标定的单应性矩阵，将倾斜画面映射为俯视鸟瞰图 |
| **方向场卷积冲突检测** | O(G²) 场卷积替代传统 O(N²) 成对几何方法（专利算法） |
| **拥堵指数 Phi** | 融合密度与速度衰减的加权指数，实时量化拥堵风险 |
| **单车归因** | 量化每辆车对当前拥堵的贡献度（Influence%） |
| **因果溯源（水滴算法）** | 稀疏矩阵迭代传播，定位拥堵"罪魁祸首"（Root Cause%） |
| **消融验证** | 逐级移除高归因车辆，验证归因准确性 |
| **事件自动导出** | 拥堵事件自动截图、CSV/JSON 摘要、消融结果 |
| **遮挡 ID 恢复** | 空间-外观记忆匹配，遮挡后恢复 track_id 和行驶方向 |

### 1.3 应用场景

- 城市交通管理交叉口拥堵监控
- 交通信号配时优化的数据支撑
- 交通规划与仿真验证
- 交通事故/异常事件事后回溯分析
- 自动驾驶路侧感知验证

---

## 2. 系统架构

### 2.1 模块分层

```
┌──────────────────────────────────────────────────────┐
│                    应用层                             │
│  run.py (CLI) │ launch.py (桌面) │ smart_console.py   │
├──────────────────────────────────────────────────────┤
│                  流水线编排层                          │
│               pipeline/engine.py                      │
├──────────┬──────────┬──────────┬─────────────────────┤
│  core/   │ analysis │ visual/  │  utils/              │
│  核心算法  │ 归因消融  │ 可视化    │  工具配置            │
├──────────┴──────────┴──────────┴─────────────────────┤
│               OpenCV / NumPy / PyTorch                │
└──────────────────────────────────────────────────────┘
```

### 2.2 流水线处理流程

```
输入: 视频文件
  ↓
1. 检测与跟踪: YOLO11 → ByteTrack → Vehicle[]
2. BEV 映射: H矩阵 → 世界坐标 (x_m, y_m, heading)
3. 轨迹管理: deque + EMA平滑
4. 运动分析: 速度计算 → moving/stationary/parked
5. Phi 计算: Φ = w_ρ·N/N_sat + w_v·(1-v/v_ref)
6. 冲突分析: 方向场卷积 + 归因（Phi>阈值时触发）
7. 因果溯源: 水滴传播 → Root Cause%
8. 可视化合成: 1920×1080 三行布局
9. 事件导出: 截图 + JSON/CSV 摘要
  ↓
输出: 可视化帧 + 事件摘要 + 实时指标
```

---

## 3. 核心算法：方向场卷积冲突检测

### 3.1 六步流程

```
Step 1: Scatter to Grid
  每辆车 → G×G 网格 (64×64)
  输出: 占用场 O, 速度场 V, 方向场 Θ

Step 2: Direction Binning (软分配)
  连续方向 0-360° → 12个方向箱 (30°每箱，高斯权重重分配)
  输出: O_k (k=0..11)

Step 3: Anisotropic Kernel
  各向异性高斯核，3σ≈7m 前向，3σ≈1.4m 侧向，扇形展开
  前向 sigma_along=3.0，侧向 sigma_perp=0.6

Step 4: Influence Convolution
  R_k = O_k ⊛ K_k (cv2.filter2D, GPU加速)

Step 5: Conflict Field
  C(x,y) = Σ R_a × R_b  (24 个冲突对)
  对向6 + 正交6 + 同向跟驰12

Step 6: Per-Vehicle Attribution
  Influence_i = R_{k_i}(P_i) × Σ R_{k'}(P_i)
```

### 3.2 核参数

| 参数 | 默认值 | 物理含义 |
|------|--------|---------|
| `arrow_half_len`=10 | 7.8m | 前向截断 |
| `kernel_half_width`=6 | 4.7m | 侧向半宽 |
| `sigma_along`=3.0 | 3σ≈7m | 前向有效范围 |
| `sigma_perp`=0.6 | 3σ≈1.4m | 侧向有效范围（并排低影响） |
| 扇形因子=0.6 | — | 前方越远侧向越宽 |

### 3.3 算法优势

| 维度 | 传统成对几何 | 方向场卷积 |
|------|------------|-----------|
| 复杂度 | O(N²) | O(G²) |
| GPU | 难以并行 | 天然 GPU 加速 |
| 冲突对 | N(N-1)/2 | 固定 24 对 |
| 消融 | 全量重算 | O(1) 重算单个方向箱 |

---

## 4. 拥堵指数 Phi

**公式**: Φ = w_ρ × min(1, N/N_sat) + w_v × max(0, 1 - v_avg/v_ref)

| Φ 范围 | 状态 | 颜色 |
|---------|------|------|
| 0.00–0.30 | 畅通 | 绿 |
| 0.30–0.55 | 轻度 | 黄 |
| 0.55–0.75 | 中度 | 橙 |
| 0.75–1.00 | 严重 | 红 |

默认参数: N_sat=40, v_ref=5.0m/s, w_ρ=0.4, w_v=0.6, 阈值=0.70

---

## 5. 因果溯源：水滴传播算法

### 5.1 原理

每辆车携带"水滴"，通过冲突场构建邻接矩阵，水沿拥堵链逆流传播汇聚到源头。

### 5.2 步骤

```
1. 构建 A[i][j] = conflict_mid × distance
   (j在i前方 + 中点有冲突)
2. 行归一化: A_norm = A / row_sum
3. 迭代: x += α · (A_norm.T @ x), α=0.4, 10-15轮
4. 冲突加权: x_final[i] = x[i] × min(conflict_pos[i]×10, 1)
5. 归一化 → RootCause%
```

### 5.3 高亮规则

| 颜色 | 条件 | 含义 |
|------|------|------|
| 🔴 红 | 水滴Top-2 + Inf>5% + 位置有冲突 | 根因 |
| 🟠 橙 | Influence > 15% | 高参与 |
| ⚪ 灰 | 其他 | 正常 |

---

## 6. 可视化系统

### 6.1 三行布局 (1920×1080)

```
Row 1 (400px): 原始视频帧 | 小BEV | 数据面板(4卡片)
Row 2 (500px): 全宽BEV + 冲突热力图 + 车辆框着色
Row 3 (180px): Phi时间线 + 事件摘要条
```

### 6.2 车辆着色

| 程度 | 填充 | 边框 | 标签 |
|------|------|------|------|
| Root Cause | 亮红 | 粗红+发光 | `#ID CAUSE xx%` |
| Influence>15% | 橙 | 橙+发光 | `#ID xx%` |
| 5-15% | 橙淡 | 灰细 | `#ID` |
| <5% | 灰蓝 | 暗线 | 无 |

---

## 7. 配置与参数

### 7.1 站点配置 (`configs/intersections.json`)

每个站点定义：视频目录、标定文件路径、模型路径、运行时参数（imgsz/conf/grid_size等）。

### 7.2 单应性标定 (`configs/homography_*.json`)

4对点 `[左上,右上,右下,左下]`，自动缩放到实际视频分辨率。

### 7.3 命令行参数

```bash
python run.py --site ziyou                    # 指定站点
python run.py --frame-stride 2                # 跳帧性能模式
python run.py --site ziyou --backtest         # 逐帧冲突分析
python run.py --model yolo11s.pt              # 轻量模型
```

---

## 8. 快速启动

### 8.1 桌面启动器（推荐）

```bash
python launcher.py
```

图形界面操作：选路口 → 选视频 → 选模型 → 选预设 → Launch。

预设：

| 预设 | imgsz | conf | 场景 |
|------|-------|------|------|
| Quick Start | 1280 | 0.22 | 日常使用 |
| High Quality | 1600 | 0.15 | 论文截图 |
| Fast Preview | 960 | 0.30 | 快速验证 |

### 8.2 命令行模式

```bash
python run.py                                # 交互选择
echo "1" | python run.py --site ziyou        # 自动模式
```

### 8.3 Web 控制台

```bash
streamlit run app/smart_console.py
```

---

## 9. 标定工具

```bash
python tools/calibrate_homography.py --site ziyou
```

鼠标拖拽4角点，u=撤销，r=重置，Enter=保存。

---

## 10. 教学演示模块

```bash
python demo_test/conflict_field_demo.py           # 核方向/参数交互
python demo_test/convolution_pipeline_demo.py     # 全流程7步骤
```

后者按 1-7 切换步骤：Scatter → Binning → R_k → Conflict → Pairs → Ablation → Root Cause。

---

## 11. 项目文件结构

```
核心源码:
├── run.py / launcher.py                # 入口
├── core/detector.py                    # YOLO+ByteTrack+记忆
├── core/conflict.py                    # 方向场卷积（核心）
├── core/phi.py                         # 拥堵指数
├── core/memory.py                      # 遮挡恢复
├── pipeline/engine.py                  # 主循环编排
├── analysis/root_cause.py              # 水滴溯源
├── analysis/attribution.py / ablation.py
├── visualization/data_panel.py / phi_chart.py / effects.py
├── utils/theme.py / canvas_pool.py / config_loader.py / drawing.py

配置文件:
├── configs/intersections.json
├── configs/homography_*.json
├── configs/traffic_risk_params.json

教学演示:
├── demo_test/conflict_field_demo.py
├── demo_test/convolution_pipeline_demo.py

文档:
├── docs/technical_architecture.md
├── DESIGN.md
├── docs/architecture_figure_design.md
├── docs/figure1_drawing_prompt.md
```

---

## 12. 依赖与环境

**Python**: 3.11+  
**关键依赖**: opencv-python, ultralytics, numpy, lap>=0.5.12, streamlit, pillow  
**GPU**: NVIDIA + CUDA（可选，CPU可运行但速度较低）  
**启动**: 推荐 Anaconda 环境运行，确保 `lap` 包正常安装

---

## 13. 常见问题

**Q: lap 找不到？**  
确保使用正确的 Python 环境安装，Anaconda 环境下 `pip install lap`。

**Q: YOLO 模型下载失败？**  
检查 `data/models/` 下是否有 `.pt` 文件，启动器会自动选择本地模型。

**Q: 性能太慢？**  
加 `--frame-stride 2` 跳帧或换 `yolo11s.pt` 轻量模型。

**Q: 标定无效？**  
四点顺序须为左上、右上、右下、左下；视频分辨率与标定点不匹配时系统自动缩放。

**Q: 中文显示乱码？**  
终端设 UTF-8：`$env:PYTHONIOENCODING='utf-8'`
