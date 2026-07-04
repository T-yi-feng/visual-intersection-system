# 视觉识别交叉口信息采集系统

**Visual Recognition Intersection Information Collection System**

基于计算机视觉的交叉口交通监控与安全分析系统。从斜视监控摄像头画面中检测车辆、追踪轨迹、分析冲突、计算拥堵指数，通过鸟瞰图（BEV）可视化展示。

> 课程设计作品 | 适用于创新创业比赛、科研实验、交通场景演示

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    展示层（Visualization）                        │
│  BEV车辆框着色 │ 拥堵热力图 │ Phi时间线图表 │ Web控制台界面      │
├─────────────────────────────────────────────────────────────────┤
│                    分析层（Analysis）                            │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ 运动状态分析  │  │ ★ 卷积冲突检测   │  │ 拥堵归因与消融    │  │
│  └──────────────┘  └──────────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    感知层（Perception）                          │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ 视频输入      │  │ YOLO11 检测       │  │ ByteTrack 追踪   │  │
│  └──────────────┘  └──────────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    数据层（Data）                                │
│  路口配置 │ 单应性标定 │ 风险参数 │ 模型权重 │ 输出结果存储      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心功能

### 1. 车辆检测与追踪
- **模型**：YOLO11m/l（支持全部YOLO11系列）
- **追踪**：ByteTrack 多目标追踪 + 滑动窗口标签稳定化
- **输出**：每个车辆的 ID、边界框、置信度、类别

### 2. 鸟瞰图变换（BEV）
- 基于单应性矩阵（Homography）的透视变换
- 斜视监控 → 俯视鸟瞰图，消除透视畸变
- 像素坐标 → 世界坐标（米制单位）
- 交互式标定工具（拖拽4点）

### 3. 卷积冲突检测（核心创新）
**传统方法 O(N²)：** 对每辆车构造方向箭头线段，两两判断是否相交。

**本系统方法（卷积）：**
1. BEV平面离散化为 64×64 网格
2. 车辆散布到网格，构建占用场、速度场、方向场
3. 方向分解为 12 个 bin（每30°一个）
4. 各向异性高斯核卷积 → 路径影响力场
5. 冲突对逐元素乘 → 冲突场
6. 每辆车归因分数 = O(1) 查表

**优势：** GPU可加速 | 包含空间信息 | 单车直接归因 | 增量消融O(1)/车

### 4. 拥堵风险指数（Phi）
```
Phi = w_ρ × ρ + w_v × η
```
- `ρ = min(1.0, active_count / N_sat)` — 密度项
- `η = max(0.0, 1.0 - avg_speed / v_ref)` — 速度衰减项
- Phi ∈ [0, 1]，0=畅通，1=严重拥堵

### 5. 车辆拥堵归因与消融实验
- 基于卷积影响力场量化每辆车对拥堵的贡献度
- 增量消融：逐级移除高归因车辆，观察拥堵变化
- 方向维度消融：按方向bin分组，分析哪个方向贡献最大

### 6. 可视化输出
- BEV 车辆框按归因分数着色（红=高贡献）
- 拥堵热力图叠加
- Phi 时间线图表（冷→热渐变）
- 三行布局：视频+BEV小图+数据 | 冲突分析全宽 | Phi时间线

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型

```bash
# 首次运行会自动下载，或手动下载放到 data/models/
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

### 3. 运行

```bash
# 交互模式（选择路口和视频）
python run.py

# 站点模式
python run.py --site huangshanlu

# 直接模式
python run.py data/videos/test.mp4 \
    --homography configs/homography_points_example.json \
    --risk-params configs/traffic_risk_params.json \
    --model data/models/yolo11m.pt \
    --imgsz 1280 \
    --frame-stride 3
```

### 4. Web 控制台

```bash
python -m streamlit run app/smart_console.py
# 打开 http://127.0.0.1:8501
```

---

## 项目结构

```
Visual System/
├── run.py                          # 主入口
├── core/                           # 核心算法
│   ├── conflict.py                 # ★ 卷积冲突检测
│   ├── detector.py                 # YOLO + ByteTrack
│   ├── motion.py                   # 运动状态分析
│   ├── phi.py                      # 拥堵指数计算
│   └── bev_transform.py            # BEV 鸟瞰变换
├── pipeline/
│   └── engine.py                   # 主循环引擎
├── analysis/
│   ├── attribution.py              # 车辆拥堵归因
│   └── ablation.py                 # 消融实验
├── visualization/
│   ├── congestion_overlay.py       # 拥堵热力图
│   ├── phi_chart.py                # Phi 时间线图表
│   └── conflict_debug.py           # 调试可视化
├── utils/
│   ├── async_writer.py             # 异步图像写入
│   ├── config_loader.py            # 配置加载
│   └── drawing.py                  # 绘图工具
├── tools/
│   ├── calibrate_homography.py     # 交互式标定
│   └── benchmark.py                # 性能基准测试
├── app/
│   └── smart_console.py            # Streamlit Web 控制台
├── tests/                          # 单元测试
├── configs/                        # 配置文件
├── data/                           # 模型、视频、标定
└── docs/                           # 文档
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `source` | — | 视频文件路径 |
| `--site` | — | 路口站点 key |
| `--model` | `data/models/yolo11s.pt` | YOLO 模型路径 |
| `--imgsz` | `640` | 推理图像尺寸（推荐1280提升小目标） |
| `--conf` | `0.15` | 检测置信度阈值 |
| `--frame-stride` | `1` | 帧跳步 |
| `--backtest` | `False` | 强制每帧执行冲突分析 |
| `--homography` | — | 单应性配置 JSON |
| `--risk-params` | — | 风险参数 JSON |
| `--grid-size` | `64` | BEV 网格边长 |
| `--max-frames` | `0` | 最大处理帧数（0=无限） |
| `--no-show-windows` | — | 无界面批处理模式 |

---

## 配置路口

### 新路口接入步骤

1. 运行标定工具：
   ```bash
   python tools/calibrate_homography.py --site <new_site>
   ```
2. 在视频帧上拖拽4个红点到路口角点，按 Enter 保存

3. 在 `configs/intersections.json` 添加站点配置

4. 将视频放到 `data/videos/<new_site>/`

### intersections.json 示例

```json
{
  "sites": {
    "my_site": {
      "display_name": "我的路口",
      "video_dir": "data/videos/my_site",
      "calibration_image": "data/calibration/my_site/image_000.png",
      "homography": "configs/homography_my_site.json",
      "risk_params": "configs/traffic_risk_params.json",
      "model": "data/models/yolo11m.pt"
    }
  }
}
```

---

## 运行时快捷键

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `p` | 暂停 / 继续 |
| `s` | 截图保存 |
| `+` / `-` | 调节显示缩放 |

---

## 技术栈

- **YOLO11** — 目标检测（Ultralytics）
- **ByteTrack** — 多目标追踪
- **OpenCV** — 图像处理、透视变换
- **NumPy** — 数值计算（卷积运算）
- **Streamlit** — Web 可视化控制台
- **PyTorch** — 深度学习推理后端

---

## 测试

```bash
python -m pytest tests/ -v
```

---

## 许可证

本项目为课程设计作品。
