# Figure 1 绘图 Prompt

> 发给绘图者 / AI 绘图工具。可直接复制使用。

---

绘图需求文档 (发给绘图者的 Prompt)

【总体风格与排版要求】

参考风格：请参考经典的深度学习架构图（如 Siamese VGGNet、ResNet、YOLO 架构图）的风格。要求：扁平化、学术风、莫兰迪色系（莫兰迪蓝 #7B9EBF、莫兰迪绿 #8FAF8F、莫兰迪橙 #D4A574、莫兰迪紫 #9B8EBF）、圆角矩形、带浅灰色背景的分区模块。

构图方向：水平横向流（Left to Right），包含从右到左的"事件触发反馈"闭环虚线。

主标题：在图片正上方居中大字：
Figure 1: Proposed Direction-Field Convolutional Framework for Traffic Conflict Detection and Congestion Attribution at Urban Intersections

整体结构：将全图在底层用浅灰色矩形分为清晰的五个逻辑大区（5 Panels）。


【模块 A：输入与 BEV 透视映射 (A. Input & BEV Mapping)】

位于图片最左侧，宽度约占 18%。

自上而下的纵向流：

1. 顶部画一个倾斜视角的路口简化透视图标：几条车道交汇，远处有简化建筑轮廓。在路口图上叠加 2-3 个彩色检测框（蓝色框和橙色框），旁边小字标注 "YOLO11m + ByteTrack"。用绿色系框（#DCE8DC 底色）包裹。

2. 从检测框图标引出一条向下的实线箭头，指向中间的"四角标定示意"：画一个略带透视的四边形平面图，四个角用小圆点标记，四角用虚线连接到下方的一个规整正方形（表示 BEV 鸟瞰图的映射关系）。框内文字 "4-Point Homography (pixel → world)"。用橙色系框包裹。

3. 从标定框引出一条向下的实线箭头，指向底部的 BEV 鸟瞰图：一个规整的正方形平面图（50m × 50m），内部有简化的车道线 + 车辆矩形框（按实际尺寸比例缩小）。框内标注 "Bird's Eye View (50m × 50m)"。用蓝色系框包裹。

4. 从 BEV 框右侧引出一条粗的蓝色实线箭头，横向穿出，进入模块 B。箭头沿途标注文字："World Coordinates: (x_m, y_m, heading, speed)"。


【模块 B：方向场卷积冲突检测流水线 (B. Direction-Field Convolution Pipeline)】

此部分为全图的视觉中心，占据最大版面（宽度约占 45%），包含 7 个从上到下排列的步骤。

整体用一个大的浅紫色（#E8E2F0）虚线圆角矩形框包裹，虚线框标题标注在顶部："Direction-Field Convolutional Conflict Detection (Patent Pending)"。

内部 7 个步骤，每个步骤用独立的浅灰圆角矩形小框包裹，步骤之间用向下的实线箭头连接：

**Step 1: Scatter to Grid（浅蓝色框 #DCE4F0）**

左侧画一个小图标：几辆彩色小圆点（代表车辆）散布进一个 64×64 的网格上（用淡蓝色细线画格子）。图标下方或用箭头引出三个输出标注：
- "Occupancy Field O(x, y)"
- "Speed Field V(x, y)"
- "Direction Field Θ(x, y)"
标注文字旁加注："Grid: G×G = 64×64, World: W×H = 50m×50m"

**Step 2: Direction Binning（浅绿色框 #DCE8DC）**

左侧画一个圆形"方向盘"饼图，切分为 12 个 30° 扇形（k=0,1,...,11），每个扇形涂不同深浅的莫兰迪颜色（表示不同的方向箱）。右侧标注：
- "12 Direction Bins (30° each)"
- 公式：O_k ∈ {0,1}^{G×G} for k = 0, 1, ..., 11
- 小字注释："Soft assignment with Gaussian weights (σ=10°)"

**Step 3: Anisotropic Kernel Construction（浅紫色框 #E8E2F0）**

左侧画 3 个并排的拉长椭圆核，朝向分别为 0°（水平拉伸）、90°（垂直拉伸）、180°（水平反向），展示各向异性高斯核的形状。
右侧标注：
- "K_k: Anisotropic Gaussian Kernels"
- "σ_along(v) = σ₀ · (1 + α · v / v_ref)  [Speed-aware]"
- "σ_perp = 1.5 cells  [Cross-lane fixed]"
- 小字："Forward stretch: 15 cells | Cross-lane width: 4 cells"

**Step 4: Path Influence Convolution（浅蓝色框 #DCE4F0）**

左侧画 O_k（小网格）与 K_k（椭圆核）并列，中间用 ⊛（卷积符号）连接，向右箭头指向 R_k（模糊条纹热力图）。标注：
- 核心公式：R_k = O_k ⊛ K_k
- "→ Path influence field in direction k"
- 小字："Implemented via cv2.filter2D (GPU-accelerated)"

**Step 5: Conflict Field Computation（醒目的橙色框 #F6EDE0，加粗边框）**

这是最重要的一步，必须用加粗边框 + 醒目配色。

左侧画两个方向的热力图 R_a（水平条纹）和 R_b（垂直条纹），交叉相乘产生 C(x,y)（红-蓝热力图）。

中间放大字号、加粗显示核心公式：
C(x,y) = Σ_{(a,b)∈ConflictPairs} R_a(x,y) × R_b(x,y)

下方标注：
- "12 Conflict Pairs: 6 Opposing (180°) + 6 Orthogonal (~90°)"
- "Adaptive pair detection: active bins ≥ 2 vehicles, skip parallel (≤30°)"
- "→ Continuous Conflict Heatmap (red = high conflict intensity)"

右侧放一个小的 inset 图：BEV 鸟瞰图叠加红色冲突热力图。

**Step 6: Per-Vehicle Attribution（浅紫色框 #E8E2F0）**

左侧画一辆车在 G×G 网格中的位置标记 P_i，标注其方向箱 k_i。

公式（加粗）：
Influence_i = R_{k_i}(P_i) × Σ_{k'≠k_i} R_{k'}(P_i)

下方标注：
- "Normalized to percentage (∑Influence_i = 100%)"
- "→ Per-vehicle ranking by congestion contribution"
- "车辆尺寸加权：truck=4.2×, car=1.0×, motorcycle=0.3×"

**Step 7: Ablation Verification（浅灰色框）**

画 3 张缩略图并排，分别标注 "K=1", "K=2", "K=3"，每张图上移除车辆数递增，冲突热力图递减。标注：
- "Iteratively remove top-K attributed vehicles"
- "Measure ΔConflict and ΔPhi after each removal"
- "O(1) recomputation: only affected direction bin updated"


【模块 C：输出与事件导出 (C. Output & Event Export)】

位于模块 B 右侧，宽度约占 22%。自上而下排列：

1. **Conflict Heatmap Overlay**：BEV 鸟瞰图叠加红-蓝热力图（来自 Step 5），车辆矩形框按归因分数着色（红色框=高归因，绿色框=低归因）。标注："Vehicle Boxes: Color-coded by Influence Score"。

2. **Per-Vehicle Ranking Table**：画一个小表格图标，列出 Top-3 归因车辆：
   - "Rank 1: ID=12 | truck | 18.2%"
   - "Rank 2: ID=7  | car   | 15.1%"
   - "Rank 3: ID=23 | van   | 11.3%"
   标注："Congestion Attribution Ranking"。

3. **Phi Timeline Chart**：画一个小型折线图图标。x 轴=时间（标注 "120s window"），y 轴=Phi ∈ [0,1]，线条颜色从蓝色渐变到红色（Phi 增高对应拥堵加重）。在 y=0.70 处画一条红色水平虚线，标注 "Event Threshold"。

4. **Event Export Files**：画 4 个小文件图标并排：
   - 📷 "Screenshot (.jpg)"
   - 📊 "Event Summary (.json)"
   - 📋 "Vehicle Ranking (.csv)"
   - 🧪 "Ablation Results (.csv)"


【模块 D：拥堵指数与智能控制台 (D. Congestion Index & Smart Console)】

位于图片右下角，宽度约占 15%。

1. **拥堵指数（Phi）仪表盘**：画一个半圆形仪表盘图标，指针指向中间偏上位置。背景从绿色渐变到黄色再到红色。上方放大标注公式：
   Φ = w_ρ · min(1, N/N_sat) + w_v · max(0, 1 - v_avg/v_ref)
   下方标注参数：
   - "N_sat = 40 (saturation count)"
   - "v_ref = 5.0 m/s ≈ 18 km/h (free-flow)"
   - "w_ρ = 0.4 (density),  w_v = 0.6 (speed)"

2. **Streamlit Web 控制台图标**：画一个简化的笔记本电脑屏幕，屏幕上展示 Streamlit 界面布局。标注：
   - "Smart Console (Streamlit)"
   - "Real-time monitoring · Site switching · Event browsing"
   - "Parameter hot-reload (runtime adjustable)"


【模块 E：三行可视化布局 (E. 1920×1080 Display Layout)】

位于图片最底部，跨 Panel A+B+C+D 的细长条区域。

画一个 1920×1080 的缩略图，内部切分为三行：
- Row 1 (400px)：标注 "Video Frame (detection + tracking) | Mini BEV | Data Panel (Phi, Speed, Count, Top-3 Vehicles)"
- Row 2 (500px)：标注 "Full-width BEV + Conflict Heatmap Overlay + Color-coded Vehicle BBoxes"
- Row 3 (180px)：标注 "Phi Timeline Chart (120s window, Blue→Green→Yellow→Orange→Red gradient by Phi value)"

三行之间用细虚线分隔。


【全局跨模块连线 (Cross-Panel Connections)】

用不同颜色的虚线圆弧表示跨区域数据流，这是图的灵魂：

1. **BEV → Scatter（模块 A → 模块 B Step 1）**：蓝色实线弧线，标注 "World Coordinates (x_m, y_m, heading, speed)"。

2. **归因分数 → 排序表（模块 B Step 6 → 模块 C Ranking）**：紫色虚线弧线，标注 "Per-Vehicle Influence Scores"。

3. **冲突场 → 热力图叠加（模块 B Step 5 → 模块 C Heatmap）**：橙色实线弧线，标注 "Conflict Field C(x,y)"。

4. **Phi 指数 → 仪表盘（模块 C Phi Chart → 模块 D Gauge）**：橙色短虚线，标注 "Φ_t"。

5. **拥堵事件反馈闭环（模块 D → 模块 B）**：极其重要的一条**红色粗虚线**，从模块 D 的仪表盘位置，向下弯折到底部，再向左弯折到模块 B 的 Step 5（冲突场计算）。虚线上方标注文字："Phi > Threshold (0.70) → Trigger Conflict Analysis"。这是图的关键反馈机制——拥堵发生时自动启动冲突分析。

6. **环境反馈闭环（模块 C → 模块 A）**：从模块 C 右侧抽出一条向上的灰色虚线箭头，跨越整个图片的上方，重新指回最左侧的模块 A（输入视频帧）。在这条顶部虚线上标注文字："Frame-by-Frame Iteration & Trail Memory Update"。


【给设计师的特别叮嘱】

1. 请务必保留所有数学公式（如 Φ、O_k、R_k、C(x,y)、Influence_i 等），这是计算机视觉 / 交通工程交叉学科论文高级感的来源。

2. 尽量用图标 + 文字结合的形式：
   - 张量用叠在一起的方块
   - 网格用淡色方格线
   - 卷积用符号 ⊛
   - 车辆用彩色小圆点或矩形框
   - 热力图用红蓝渐变色块
   避免全是纯文字框，以达到参考图（Siamese VGGNet / ResNet 架构图）中丰富的视觉层级效果。

3. Step 5（冲突场计算）是核心创新点，必须在视觉上最突出——用加粗边框、最大字号公式、醒目的橙色底。

4. 所有箭头避免 90° 直角——用贝塞尔曲线圆弧过渡。

5. Panel 之间留 8-12px 的间隙，不要紧贴。

6. 莫兰迪色系的关键是**低饱和度**——所有颜色不能鲜艳，要有一层"灰调"覆盖在上面的感觉。白色底配浅灰模块底是最安全的组合。

7. 建议最终导出为 PDF 矢量图（论文用）和 PNG @300dpi（预览/PPT用）两个版本。
