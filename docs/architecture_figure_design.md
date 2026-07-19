# 系统架构图设计文档

> **论文级 Figure 1: System Architecture of Direction-Field Convolution Framework for Intersection Conflict Detection**
> 目标刊物：IEEE T-ITS / TRB / CVPR 级别视觉标准

---

## 一、图面总体布局

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Figure 1: Proposed Direction-Field Convolutional Framework for Traffic Conflict      │
│  Detection and Congestion Attribution at Urban Intersections                          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─ Panel A ─────────────┐   ┌─ Panel B (Core) ───────────────────────────────┐    │
│   │ Input & BEV Mapping   │   │    Direction-Field Convolutional Pipeline       │    │
│   │                       │   │                                                 │    │
│   │  ┌─────────┐          │   │  ┌──────────────────────────────┐              │    │
│   │  │ 原始监控  │          │   │  │    Step 1: Scatter to Grid    │              │    │
│   │  │ 倾斜画面  │          │   │  │  Vehicles → G×G Occupancy     │              │    │
│   │  │ + YOLO   │          │   │  │  O, Speed V, Direction Θ      │              │    │
│   │  │ + ByteT  │          │   │  └───────────┬──────────────────┘              │    │
│   │  └────┬─────┘          │   │              │                                  │    │
│   │       │                │   │              ▼                                  │    │
│   │       ▼                │   │  ┌──────────────────────────────┐              │    │
│   │  ┌─────────┐           │   │  │ Step 2: Direction Binning    │              │    │
│   │  │ 4-pt     │           │   │  │ 0-360° → 12 bins (30° each) │              │    │
│   │  │ Homog.   │           │   │  │ O_k binary layers (k=0..11) │              │    │
│   │  │ pixel→   │           │   │  └───────────┬──────────────────┘              │    │
│   │  │ world    │           │   │              │                                  │    │
│   │  └────┬─────┘           │   │              ▼                                  │    │
│   │       │                │   │  ┌──────────────────────────────┐              │    │
│   │       ▼                │   │  │ Step 3: Anisotropic Kernels  │              │    │
│   │  ┌─────────┐           │   │  │ K_k: Gaussian stretched       │              │    │
│   │  │ BEV      │           │   │  │ along driving direction       │              │    │
│   │  │ 鸟瞰图   │           │   │  │ σ_along ≫ σ_perp             │              │    │
│   │  │ 50m×50m │           │   │  └───────────┬──────────────────┘              │    │
│   │  └─────────┘           │   │              │                                  │    │
│   │                       │   │              ▼                                  │    │
│   └───────────────────────┘   │  ┌──────────────────────────────┐              │    │
│                               │  │ Step 4: Path Influence Conv. │              │    │
│                               │  │ R_k = O_k ⊛ K_k              │              │    │
│                               │  │ (cv2.filter2D, GPU-friendly) │              │    │
│                               │  └───────────┬──────────────────┘              │    │
│   ┌─ Panel C ─────────────┐   │              │                                  │    │
│   │ Output & Analysis     │   │              ▼                                  │    │
│   │                       │   │  ┌──────────────────────────────┐              │    │
│   │  ┌──────────────┐     │   │  │ Step 5: Conflict Field        │              │    │
│   │  │ Conflict      │◄────┼───┤  │ C = Σ R_a × R_b              │              │    │
│   │  │ Heatmap       │     │   │  │ 12 conflict pairs            │              │    │
│   │  │ (BEV overlay) │     │   │  │ (6 opposite + 6 orthogonal)  │              │    │
│   │  └──────┬───────┘     │   │  └───────────┬──────────────────┘              │    │
│   │         │             │   │              │                                  │    │
│   │         ▼             │   │              ▼                                  │    │
│   │  ┌──────────────┐     │   │  ┌──────────────────────────────┐              │    │
│   │  │ Per-Vehicle   │     │   │  │ Step 6: Attribution            │              │    │
│   │  │ Attribution   │     │   │  │ Influence_i = R_k(P_i)        │              │    │
│   │  │ Ranking       │     │   │  │ × Σ R_k'(P_i)                 │              │    │
│   │  └──────┬───────┘     │   │  └───────────┬──────────────────┘              │    │
│   │         │             │   │              │                                  │    │
│   │         ▼             │   │              ▼                                  │    │
│   │  ┌──────────────┐     │   │  ┌──────────────────────────────┐              │    │
│   │  │ Ablation      │     │   │  │ Step 7: Ablation Study        │              │    │
│   │  │ Verification  │     │   │  │ Remove top-K vehicles         │              │    │
│   │  │ (k-level)     │     │   │  │ → ΔConflict, ΔPhi            │              │    │
│   │  └──────┬───────┘     │   │  └───────────────────────────────┘              │    │
│   │         │             │   │                                                 │    │
│   │         ▼             │   └─────────────────────────────────────────────────┘    │
│   │  ┌──────────────┐     │
│   │  │ Phi Timeline  │     │   ┌─ Panel D (Sidebar) ──────────────────────────┐    │
│   │  │ + Event       │     │   │  Live Metrics & Smart Console                │    │
│   │  │ Export        │     │   │                                              │    │
│   │  │ (JSON/CSV)    │     │   │  ┌────────────────┐  ┌────────────────────┐ │    │
│   │  └──────────────┘     │   │  │  Congestion     │  │  Vehicle Ranking    │ │    │
│   │                       │   │  │  Index          │  │  Top-3 Contributors │ │    │
│   └───────────────────────┘   │  │  Φ = w_ρ·N/N_sat│  │  ID | Score | Type  │ │    │
│                               │  │      + w_v·v/v_ref│  │  12 | 18.2%  | truck│ │    │
│                               │  │  (Density+Speed) │  │   7 | 15.1%  | car  │ │    │
│                               │  │                  │  │  23 | 11.3%  | van  │ │    │
│                               │  │  ┌────────────┐  │  └────────────────────┘ │    │
│                               │  │  │  Phi Chart  │  │                        │    │
│   ┌─ Panel E ═══════════┐     │  │  │  Timeline   │  │  ┌────────────────────┐ │    │
│   │  1920×1080 Display  │     │  │  │  (120s)     │  │  │  Event Detection   │ │    │
│   │                     │     │  │  └────────────┘  │  │  Start / Peak / End │ │    │
│   │ Row 1 (400px):      │     │  └────────────────┘  │  + Auto Screenshot   │ │    │
│   │ Video | BEV | Data  │     │                      └────────────────────┘ │    │
│   │ Row 2 (500px):      │     │                      ┌────────────────────┐ │    │
│   │ Full-BEV + Conflict │     │                      │  Ablation Results  │ │    │
│   │ + Vehicle BBoxes    │     │                      │  Level | ΔΦ | ΔC   │ │    │
│   │ Row 3 (180px):      │     │                      │  K=1  |12%| 0.34   │ │    │
│   │ Phi Timeline Chart  │     │                      │  K=2  |23%| 0.51   │ │    │
│   └─────────────────────┘     │                      │  K=3  |35%| 0.68   │ │    │
│                               │                      └────────────────────┘ │    │
│                               └──────────────────────────────────────────────┘    │
│                                                                                      │
│   ┌─ Cross-Panel Connections (dashed arcs) ─────────────────────────────────────┐    │
│   │                                                                              │    │
│   │   Panel A (BEV) ──────────────► Panel B (Scatter to Grid)                    │    │
│   │   Panel B (Step 6: Influence) ──► Panel C (Attribution Ranking)              │    │
│   │   Panel C (Phi) ───────────────► Panel D (Timeline)                          │    │
│   │   Panel D (Event Trigger) ─────► Panel B (Step 5: Conflict, on-demand)       │    │
│   │   Panel C (Ablation) ─────────► Panel B (Step 7: Recompute)                  │    │
│   │                                                                              │    │
│   └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、配色方案（莫兰迪学术色系）

| 色名 | 色值 | 用途 |
|------|------|------|
| 莫兰迪蓝 | `#7B9EBF` / `#DCE4F0`(bg) | Panel B 主体、卷积步骤、公式框 |
| 莫兰迪绿 | `#8FAF8F` / `#DCE8DC`(bg) | 车辆检测、PPO、成功状态 |
| 莫兰迪橙 | `#D4A574` / `#F6EDE0`(bg) | 冲突热力图、Phi 指数、警告信号 |
| 莫兰迪紫 | `#9B8EBF` / `#E8E2F0`(bg) | 归因分析、抽象数学模块 |
| 暖灰 | `#B5B0A8` / `#F5F4F1`(bg) | Panel 底色、道路/基础设施 |
| 深红 | `#C0503C` | 高冲突区域、关键连线标注 |
| 文字色 | `#3A3A3A` | 正文 |
| 弱文字 | `#888888` | 注释、公式脚标 |

---

## 三、细节规范

### 3.1 Panel A: 输入与 BEV 映射（左侧，宽度占比 ~18%）

**构成**（自上而下）：
1. **原始监控画面图标**：画一个倾斜视角的路口简化透视图，几条车道交汇，远景建筑物轮廓。
2. **YOLO11 检测框示意**：在路口图上叠加 2-3 个彩色 bounding box（蓝=car，橙=truck），旁边标注 "YOLO11m + ByteTrack"。
3. **箭头** 指向下方。
4. **四角标定示意**：画一个带 4 个角点 (●) 的简化四边形平面图，标注 "4-Point Homography"。4 个角点从原始图的透视位置映射到 BEV 的正方形角点，用小虚线连接表示映射关系。
5. **箭头** 指向下方。
6. **BEV 鸟瞰图**：一个规整的正方形俯视平面图，内部有车道线 + 车辆矩形框（按真实尺寸比例），标注 "Bird's Eye View (50m × 50m)"。

**数据流箭头**：
- 一条粗箭头（蓝色）从 BEV 框的右侧横向穿出，跨过 Panel 间隙，进入 Panel B 的 Step 1，箭头标注 "World Coordinates (x_m, y_m, heading, speed)"

### 3.2 Panel B: 方向场卷积流水线（中部，核心，宽度占比 ~45%）

**这是全图的视觉重心，必须最大、最丰富。** 用浅紫色虚线大框包裹整个 Panel B，框标题 "Direction-Field Convolutional Conflict Detection (Patent Pending)"。

内部按 **6 个垂直排列的步骤** 布局（从上到下，Step 1 → Step 6），每步用浅灰圆角矩形包裹：

**Step 1: Scatter to Grid**
- 左侧小图标：几辆车（圆点）散布进一个 G×G 的网格（64×64）
- 网格用淡蓝色格子线表示
- 右侧输出标注：
  - "Occupancy Field O(x,y)"
  - "Speed Field V(x,y)"
  - "Direction Field Θ(x,y)"
- 这一步用**浅蓝色**框

**Step 2: Direction Binning**
- 左侧小图标：一个圆形方向盘，切分为 12 个 30° 扇形（k=0,1,...,11）
- 每个扇形涂不同深浅的莫兰迪蓝色（表示不同方向箱）
- 右侧标注：
  - "12 Direction Bins (30° each)"
  - "O_k ∈ {0,1}^{G×G} for k = 0,1,...,11"
- 用**浅绿色**框

**Step 3: Anisotropic Kernel Construction**
- 左侧小图标：3 个拉长的椭圆核，朝向不同方向（示意各向异性高斯）
  - 核 0（0° 方向）：水平拉长的椭圆
  - 核 3（90° 方向）：垂直拉长的椭圆
  - 核 6（180° 方向）：水平反向的椭圆
- 标注公式：
  - "σ_along = 5.0 cells (forward, speed-aware)"
  - "σ_perp  = 1.5 cells (cross-lane)"
- 右侧标注：
  - "K_k: Anisotropic Gaussian"
  - "Stretched along driving direction"
  - "Speed-aware: σ_along(v) = σ_0·(1 + α·v/v_ref)"
- 用**浅紫色**框

**Step 4: Path Influence Convolution**
- 左侧小图标：O_k 与 K_k 并列，中间用 ⊛ 符号连接，箭头指向右侧的 R_k 热力图
- R_k 热力图画成沿特定方向的模糊条纹状（示意卷积扩散效果）
- 标注：
  - "R_k = O_k ⊛ K_k"
  - "→ Path influence field in direction k"
  - "Implemented via cv2.filter2D (GPU-accelerated)"
- 用**浅蓝色**框

**Step 5: Conflict Field Computation**
- 这是最重要的步骤，用**醒目的橙色框** + 加粗边框
- 左侧图标：两个方向的 R_a 和 R_b 热力图（一个水平条纹、一个垂直条纹），交叉相乘产生 C(x,y)
- 中间放公式（大字、加粗）：
  - **"C(x,y) = Σ_{(a,b)∈ConflictPairs} R_a(x,y) × R_b(x,y)"**
- 下方标注：
  - "12 Conflict Pairs: 6 Opposing (180°) + 6 Orthogonal (~90°)"
  - "C(x,y) → Conflict Heatmap (red=high conflict)"
- 用一个小的 inset 图展示：BEV 鸟瞰图叠加红-蓝热力图

**Step 6: Per-Vehicle Attribution**
- 用**浅紫色**框
- 左侧图标：一辆车的示意图标注其在 G×G 网格中的位置 P_i，以及其方向箱 k_i
- 公式：
  - **"Influence_i = R_{k_i}(P_i) × Σ_{k'≠k_i} R_{k'}(P_i)"**
- 下方标注：
  - "Normalized to % (all vehicles sum to 100%)"
  - "→ Generates per-vehicle ranking"

**Step 7: Ablation Verification**
- 用**浅灰色**框，位于 Step 6 下方
- 图标：3 张缩略图并排，分别标 K=1, K=2, K=3，每张图上移除的车辆数递增，冲突热力图递减
- 标注：
  - "Iteratively remove top-K attributed vehicles"
  - "→ Measure ΔConflict and ΔPhi"
  - "O(1) recomputation: only affected direction bin"

### 3.3 Panel C: 输出与分析（右侧，宽度占比 ~22%）

**构成**（自上而下，与 Panel B 的 Step 5-7 右侧对齐）：

1. **Conflict Heatmap Overlay**：BEV 图叠加红蓝热力图（来自 Step 5），车辆框颜色按影响着色（红=高归因，绿=低归因），标注 "Vehicle Bounding Boxes: Color-coded by Attribution Score"

2. **Per-Vehicle Ranking Table**：一个小表格图标，展示 Top-3 车辆：
   - "ID 12 | truck | 18.2% influence"
   - "ID 7  | car   | 15.1% influence"
   - "ID 23 | van   | 11.3% influence"

3. **Phi Timeline Chart**：一个小型折线图图标，x 轴=时间（120s 窗口），y 轴=Phi ∈ [0,1]，线条颜色从蓝渐变到红（Phi 增高）。标注阈值线 "Event Threshold (Φ > 0.70)"。

4. **Event Export Box**：
   - "Screenshot × 1"
   - "Event Summary JSON"
   - "Vehicle Ranking CSV"
   - "Ablation Results CSV"

### 3.4 Panel D: 实时仪表盘（右下角，宽度占比 ~15%）

1. **Congestion Index 仪表**：一个半圆形仪表盘图标，指针指向当前 Phi 值，背景色从绿渐变到红
   - 公式标注：**"Φ = w_ρ·min(1,N/N_sat) + w_v·max(0,1-v_avg/v_ref)"**
   - "N_sat=40, v_ref=5.0 m/s"
   - "w_ρ=0.4, w_v=0.6"

2. **Streamlit Web 控制台小图标**：笔记本电脑屏幕上展示 Streamlit 界面
   - "Smart Console: Site management, Real-time monitoring, Event browsing"

### 3.5 Panel E: 显示布局（底部，跨 Panel A+B+C，细长条）

画一个 1920×1080 的三行布局缩略图：
- Row 1: "Video Frame + Mini BEV + Data Panel"
- Row 2: "Full-width BEV + Conflict Overlay + Colored BBoxes"
- Row 3: "Phi Timeline (120s, blue→red gradient)"

### 3.6 跨 Panel 连接线

用**不同颜色的虚线圆弧**表示跨区域数据流：

| 连线 | 颜色 | 样式 | 标注 |
|------|------|------|------|
| Panel A → Panel B (BEV→Scatter) | 蓝色 `#7B9EBF` | 实线弧线 | "World Coordinates" |
| Panel B (Step 6) → Panel C (Ranking) | 紫色 `#9B8EBF` | 实线弧线 | "Attribution Scores" |
| Panel B (Step 5) → Panel C (Heatmap) | 橙色 `#D4A574` | 实线弧线 | "Conflict Field C(x,y)" |
| Panel C (Phi) → Panel D (仪表) | 橙色 `#D4A574` | 虚线短弧 | "Φ_t" |
| Panel D (Event Trigger) → Panel B | 红色 `#C0503C` | 虚线长弧（反馈闭环） | "Φ > Threshold → Trigger Conflict Analysis" |

---

## 四、公式排版规范

所有公式使用 LaTeX 数学斜体。关键公式用加粗或加大字号突出：

| 公式 | 位置 | 视觉权重 |
|------|------|---------|
| $O_k \in \{0,1\}^{G \times G}, \quad k=0,\ldots,11$ | Panel B, Step 2 | 常规 |
| $R_k = O_k \circledast K_k$ | Panel B, Step 4 | 常规 |
| $\mathbf{C}(x,y) = \sum_{(a,b)} R_a(x,y) \cdot R_b(x,y)$ | Panel B, Step 5 | **加粗放大** |
| $\text{Influence}_i = R_{k_i}(P_i) \cdot \sum_{k'} R_{k'}(P_i)$ | Panel B, Step 6 | **加粗** |
| $\Phi = w_\rho \cdot \min(1, \frac{N}{N_{\text{sat}}}) + w_v \cdot \max(0, 1 - \frac{v_{\text{avg}}}{v_{\text{ref}}})$ | Panel D | 常规 |
| $h = H \cdot [p_x, p_y, 1]^\top$ | Panel A | 常规（单应性映射） |

---

## 五、视觉层次参考

参考风格：**Siamese VGGNet / ResNet 架构图的扁平化学术风**

- 底色：纯白 `#FFFFFF`
- 模块底色：浅灰 `#F5F4F1`
- 无阴影，只用 1px 浅色描边区分模块
- 所有文字为无衬线字体（Segoe UI / Helvetica Neue）
- 公式为衬线字体（Times New Roman / STIX Two Math）
- 箭头统一用 `stroke-linejoin: round`，端点用实心三角箭头
- 图标用极简几何形状（矩形+圆点+折线），无渐变无阴影，纯扁平
- 所有模块圆角 `border-radius: 8-12px`
- 连接线避免 90° 直角转弯，用圆弧过渡
