# DESIGN.md — 可视化系统设计规范

> 视觉识别交叉口信息采集系统 | 2026-07-17 | v1.0

---

## 一、当前问题诊断

基于对全部 5 个可视化模块的逐行审查，识别出以下问题：

### 1.1 色彩系统碎片化

| 位置 | 当前做法 | 问题 |
|------|---------|------|
| `phi_chart.py` | BGR 硬编码 `(0,200,255)` | 与 congestion_overlay 的颜色公式不一致 |
| `congestion_overlay.py` | `g = 50*(1-abs(ratio-0.5)*2)` | 绿色通道人为压低到 50 |
| `engine.py:data_panel` | 每种信息一个颜色，无规律 | 6 种颜色散落，没有设计系统 |
| `conflict_debug.py` | 12 个 pastel 色硬编码数组 | 与其他模块完全不同的色系 |
| `drawing.py:id_color` | `track_id * 7919 + 104729` 取色 | 饱和度过高，在深色背景上刺眼 |

**结论**：没有一个统一的色彩设计系统。不同模块对相同语义（拥堵、归因、速度）用了不同颜色。

### 1.2 文字渲染性能陷阱

`draw_text_with_bg` 每遇到 CJK 字符就会做一次 `Image.fromarray` → PIL 渲染 → `np.array` 的全图拷贝。1920×1080 × 3 通道 = **6MB 内存拷贝** 每次调用。数据面板一行 10 次调用 = **60MB/帧** 的额外内存带宽。如果 CJK 文本只出现在数据面板而非每辆车标签上，问题不大；但如果 track_id 标签也触发 CJK（中文字符），就成性能瓶颈。

### 1.3 Phi Chart 缺乏图表语法

当前 Phi 曲线只有线段和圆点，缺少：
- 网格线（x 轴时间刻度、y 轴 Phi 刻度）
- 填充区域（曲线下方面积用渐变填充）
- 图例 / 阈值标签
- 当前值的大字号 callout

### 1.4 数据面板纯文本堆叠

640×400 的数据面板有 10 行文本 + 硬编码 Y 间距。没有：
- 信息分组（车辆统计 vs 归因排名 vs 系统状态）
- 视觉"卡片"分区
- 数值变化的动态高亮

### 1.5 BEV 热力图渲染粗糙

当前用 `step = max(2, grid_size // 32)` 抽稀渲染（64→每 2 格才画一格），失去了精细度。用 `cv2.applyColorMap` + `cv2.resize` 可以一次性渲染全部格子的平滑热力图且更快（GPU 加速 resize）。

---

## 二、设计目标

1. **统一色彩体系**——一套莫兰迪学术色板，所有模块一致性
2. **图表级 Phi Chart**——网格线、填充面积、轴标签、callout
3. **卡片式数据面板**——信息分组、视觉层级、动态高亮
4. **平滑 BEV 热力图**——GPU resize 替代手动抽稀，质量 + 性能双赢
5. **不增加帧耗时**——所有改进依赖 OpenCV GPU 加速路径，不引入新的逐像素循环

---

## 三、统一色彩体系

### 3.1 色板定义

```
莫兰迪学术暗色主题（适合投影 + 长时间观看不刺眼）

┌──────────────────────────────────────────────────────────┐
│ 语义         │ 名称        │ BGR (OpenCV)    │ 用途       │
├──────────────┼─────────────┼─────────────────┼────────────┤
│ 主文字       │ text_primary│ (220,220,220)   │ 标题/正文  │
│ 次文字       │ text_secondary│(150,150,150)  │ 注释/单位  │
│ 画布底色     │ bg_canvas   │ (24,24,28)      │ 全局背景  │
│ 面板底色     │ bg_panel    │ (32,32,38)      │ 卡片/面板 │
│ 面板边框     │ border_panel│ (55,55,62)      │ 分隔线    │
├──────────────┼─────────────┼─────────────────┼────────────┤
│ 畅通/低速    │ phi_low     │ (120,180,160)   │ Phi < 0.3 │
│ 轻度拥堵     │ phi_moderate│ (200,185,120)   │ Phi 0.3-0.55│
│ 中度拥堵     │ phi_high    │ (210,150,100)   │ Phi 0.55-0.75│
│ 严重拥堵     │ phi_critical│ (200,80,70)     │ Phi > 0.75│
├──────────────┼─────────────┼─────────────────┼────────────┤
│ 归因-低      │ attr_low    │ (100,160,100)   │ influence<5%│
│ 归因-中      │ attr_mid    │ (180,200,80)    │ 5-15%      │
│ 归因-高      │ attr_high   │ (80,140,220)    │ >15%       │
├──────────────┼─────────────┼─────────────────┼────────────┤
│ 强调色       │ accent      │ (200,160,80)    │ 当前值/告警│
│ 信息色       │ info        │ (160,180,200)   │ 速度/统计  │
└──────────────┴─────────────┴─────────────────┴────────────┘
```

### 3.2 颜色工具函数（新增 `utils/theme.py`）

```python
# 所有颜色查找统一入口
THEME = { ... }  # 上述色板

def phi_color(phi: float) -> tuple[int,int,int]:
    """Phi → BGR 的连续渐变（莫兰迪调色）"""
    if phi < 0.3:
        t = phi / 0.3
        return lerp_bgr(THEME['phi_low'], THEME['phi_moderate'], t)
    elif phi < 0.55:
        t = (phi - 0.3) / 0.25
        return lerp_bgr(THEME['phi_moderate'], THEME['phi_high'], t)
    elif phi < 0.75:
        t = (phi - 0.55) / 0.20
        return lerp_bgr(THEME['phi_high'], THEME['phi_critical'], t)
    else:
        return THEME['phi_critical']

def attr_color(influence_pct: float) -> tuple[int,int,int]:
    """归因百分比 → BGR"""
    t = min(influence_pct / 20.0, 1.0)  # 20% 封顶
    if t < 0.5:
        return lerp_bgr(THEME['attr_low'], THEME['attr_mid'], t * 2)
    else:
        return lerp_bgr(THEME['attr_mid'], THEME['attr_high'], (t - 0.5) * 2)
```

**性能**：`lerp_bgr` 是纯算术运算，无内存分配。替换当前各处散落的 `(b,g,r) = ...` 公式。

---

## 四、Phi Chart 升级（`visualization/phi_chart.py` 重构）

### 4.1 目标效果

```
┌──────────────────────────────────────────────────────────────┐
│  Φ                                                          │
│ 1.0 ┤                                              ╭─ 0.82  │  ← 当前值 callout
│     │                                           ╭──╯        │
│ 0.7 ┤- - - - - - - - - - - - - - - - - - - - -╱-╱-╱-╱-╱-╱- │  ← 阈值虚线 + 标签
│     │                                   ╭─────╯             │
│ 0.5 ┤                              ╭────╯                   │
│     │                         ╭────╯    ░░░░░               │  ← 曲线下填充
│ 0.3 ┤                    ╭────╯     ░░░░░░░░░               │
│     │               ╭────╯    ░░░░░░░░░░░░░░░               │
│ 0.0 ┤──────────────╯────────────────────────────            │
│     └─────┼─────────┼─────────┼─────────┼─────────┼─────    │
│        -100s      -75s      -50s      -25s       now        │  ← x轴时间刻度
│                                                             │
│  █ 拥堵事件 #3  持续 18.2s  峰值 Φ=0.82  归因: ID12(truck)  │  ← 事件摘要条
└──────────────────────────────────────────────────────────────┘
```

### 4.2 改进清单

| 改进项 | 当前 | 目标 | 性能影响 |
|--------|------|------|---------|
| 曲线下填充 | 无 | 半透明多边形（`cv2.fillPoly`） | 1 次 fillPoly，O(K) |
| 网格线 | 无 | 3 条水平 + 5 条垂直虚线 | 8 次 `cv2.line`，可忽略 |
| 轴标签 | 无 | Y: 0.0/0.3/0.5/0.7/1.0, X: 时间 | 5 次 `putText` (ASCII)，可忽略 |
| 阈值线标签 | 无线条 | 红色虚线 + "Φ=0.70 Event Threshold" | 1 次 `putText` |
| 当前值 callout | 小字 | 大字 + 圆角背景框 + 右侧浮动 | 1 次 `draw_text_with_bg` |
| 事件摘要条 | 无 | 图底部 30px 事件信息条 | 1 次 `cv2.rectangle` + text |
| 线段渲染 | 逐段 `cv2.line` | `cv2.polylines` 批量绘制 | 减少函数调用开销 |

### 4.3 曲线下渐变填充（实现）

```python
def _draw_fill_under_curve(canvas, xs, ys, color, alpha=0.25):
    """在曲线下方绘制半透明填充区域"""
    # 构建多边形: 曲线点 + 底部水平线
    pts = np.column_stack([xs, ys]).astype(np.int32)
    bottom_left  = [xs[0],  canvas_h - margin_b]
    bottom_right = [xs[-1], canvas_h - margin_b]
    poly = np.vstack([pts, [bottom_right], [bottom_left]])
    # 在 overlay 上绘制，然后混合
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)
```

**性能**：1 次 `canvas.copy()` (~500KB for 1920×180 ROI) + 1 次 `fillPoly` + 1 次 `addWeighted`。约 0.3ms。

### 4.4 批量线段绘制

```python
# 替代 for i in range(1, len(window_data)): cv2.line(...)
# 用 cv2.polylines 一次调用绘制全部线段
pts = np.column_stack([xs, ys]).reshape(-1, 1, 2).astype(np.int32)
cv2.polylines(canvas, [pts], isClosed=False, color=..., thickness=2, lineType=cv2.LINE_AA)
```

---

## 五、数据面板卡片化（`pipeline/engine.py` Row1-Right 重构）

### 5.1 目标布局

```
┌────────────────────────────┐
│  ╔══════════════════════╗  │
│  ║  Φ  0.72  ● 中度拥堵  ║  │  ← 卡片1: 拥堵状态 (彩色左边框)
│  ╚══════════════════════╝  │
│                             │
│  ┌── 车辆统计 ──────────┐  │
│  │  🚗 35 辆  │  v=2.1  │  │  ← 卡片2: 统计指标 (图标+数值)
│  │  🛑 静止 5  │  🅿 停 2 │  │
│  └──────────────────────┘  │
│                             │
│  ┌── 冲突归因 Top-3 ────┐  │
│  │  1. ID 12  truck ████│  │  ← 卡片3: 排行 (进度条)
│  │  2. ID  7  car   ███ │  │
│  │  3. ID 23  van   ██  │  │
│  └──────────────────────┘  │
│                             │
│  ┌── 系统状态 ──────────┐  │
│  │  site: ziyou          │  │  ← 卡片4: 元信息
│  │  FPS: 28.5  │  backtest│  │
│  └──────────────────────┘  │
└────────────────────────────┘
```

### 5.2 实现细节

**卡片函数**：

```python
def draw_card(canvas, x, y, w, h, title, rows, accent_color=None):
    """绘制一个信息卡片
    
    Args:
        accent_color: 左边框强调色，None 则不画
        rows: list of (label, value, value_color) 或 (text, color)
    """
    # 半透明底色
    overlay = canvas[y:y+h, x:x+w].copy()
    overlay[:] = THEME['bg_panel']
    cv2.addWeighted(overlay, 0.7, canvas[y:y+h, x:x+w], 0.3, 0, 
                    dst=canvas[y:y+h, x:x+w])
    
    # 圆角矩形边框
    cv2.rectangle(canvas, (x,y), (x+w,y+h), THEME['border_panel'], 1)
    
    # 左边强调线（如果有 accent_color）
    if accent_color:
        cv2.line(canvas, (x+2, y+8), (x+2, y+h-8), accent_color, 3)
    
    # 标题
    cv2.putText(canvas, title, (x+12, y+24), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, THEME['text_secondary'], 1)
    
    # 数据行
    for i, row in enumerate(rows):
        ry = y + 48 + i * 22
        # label + value 排版
        ...
```

**归因排行进度条**：

```python
def draw_attribution_bar(canvas, x, y, w, label, pct, color):
    """归因排行条目: label | ████░░░░ pct%"""
    bar_w = int(w * min(pct / 20.0, 1.0))  # 20% = 满条
    cv2.rectangle(canvas, (x+80, y+2), (x+80+bar_w, y+14), color, -1)
    cv2.rectangle(canvas, (x+80, y+2), (x+80+w, y+14), THEME['border_panel'], 1)
    cv2.putText(canvas, label, (x, y+12), ..., THEME['text_primary'], ...)
    cv2.putText(canvas, f"{pct:.1f}%", (x+80+bar_w+4, y+12), ..., color, ...)
```

**性能**：卡片渲染只有矩形 + 线条 + `putText`（ASCII 文本），无 CJK 路径，无需 PIL。整个数据面板约 0.5ms。

---

## 六、BEV 热力图渲染优化

### 6.1 问题

当前做法（`engine.py` 第 556-586 行）：

```python
step = max(2, grid_size // 32)  # 抽稀因子
for gy in range(0, grid_size, step):
    for gx in range(0, grid_size, step):
        # 逐格画小矩形，双重循环 O(G²/step²)
```

问题：
- 抽稀后画质粗糙（每 2 格才画一格）
- 双重循环在 Python 层慢
- 每格一次 `cv2.rectangle` 调用开销

### 6.2 改进方案

**用 `cv2.resize` 一次性渲染**：

```python
def render_heatmap_fast(conflict_field, target_w, target_h, colormap=None):
    """将 G×G 冲突场渲染为 target_w×target_h 的平滑热力图
    
    利用 cv2.resize 的 INTER_LINEAR 做双线性插值，
    比手动逐格渲染更快 + 更平滑。
    """
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    
    # conflict_field: (G, G) float32, 范围 [0, 1]
    field_u8 = (conflict_field * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(field_u8, colormap)
    smooth = cv2.resize(colored, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return smooth
```

**性能对比**：

| 方法 | 64×64 网格 渲染到 500×500 | 耗时估算 |
|------|--------------------------|---------|
| 当前：双重循环 `step=2` | 1024 次 `cv2.rectangle` | ~4ms |
| 当前：双重循环 `step=1` | 4096 次 `cv2.rectangle` | ~12ms |
| **改进：resize** | 1 次 `applyColorMap` + 1 次 `resize` | **~0.2ms** |

**附加收益**：可以换 colormap。`COLORMAP_JET` 是当前默认，但学术场景建议用 `COLORMAP_VIRIDIS`（感知均匀，色盲友好，更专业）。

### 6.3 热力图颜色图例

在 BEV 面板右下角叠加一个 20×100px 的垂直渐变条 + min/max 标注：

```python
def draw_colorbar(canvas, x, y, w, h, vmin=0, vmax=1, colormap=None):
    """绘制垂直颜色图例条"""
    grad = np.linspace(vmax, vmin, h, dtype=np.uint8).reshape(h, 1)  # 倒序
    if colormap is None:
        colormap = cv2.COLORMAP_VIRIDIS
    bar = cv2.applyColorMap(grad, colormap)
    canvas[y:y+h, x:x+w] = cv2.resize(bar, (w, h))
    cv2.putText(canvas, f"{vmax:.1f}", (x+w+4, y+10), ..., THEME['text_secondary'], ...)
    cv2.putText(canvas, f"{vmin:.1f}", (x+w+4, y+h-4), ..., THEME['text_secondary'], ...)
```

---

## 七、车辆可视化改进

### 7.1 当前

车辆在 BEV 上是旋转矩形（`cv2.fillPoly`）+ 灰色边框（`cv2.polylines`）+ 灰色方向箭头。所有车看起来几乎一样，归因差异仅靠 fill_color 的细微变化来体现，不够直观。

### 7.2 改进：三级视觉强调

```
低归因 (<5%):   细边框(1px) + 浅填充 + 小标签
中归因 (5-15%): 标准边框(1.5px) + 标准填充 + 标准标签
高归因 (>15%):  粗边框(2px) + 发光填充 + 大字标签 + 脉冲高亮
```

**"发光"效果实现（不增加额外开销）**：

```python
def draw_glow_box(canvas, pts, color, glow_radius=2):
    """带发光效果的旋转矩形框"""
    if glow_radius > 0:
        # 先用粗线画半透明光晕
        glow_overlay = canvas.copy()
        cv2.polylines(glow_overlay, [pts], isClosed=True, 
                      color=color, thickness=glow_radius + 2, lineType=cv2.LINE_AA)
        cv2.addWeighted(glow_overlay, 0.3, canvas, 0.7, 0, dst=canvas)
    # 主体边框
    cv2.polylines(canvas, [pts], isClosed=True, 
                  color=color, thickness=2, lineType=cv2.LINE_AA)
```

只在 `influence_pct > 15%` 的车辆上触发，通常只有 1-3 辆，额外开销可忽略。

### 7.3 方向箭头改进

当前灰色箭头在深色背景上对比度低。改为半透明白色 + 车速编码的箭头长度：

```python
arrow_len = base_len * (1.0 + 0.5 * min(speed / v_ref, 2.0))  # 速度×1~2
arrow_color = (240, 240, 240)  # 白色箭头，更好的对比度
```

---

## 八、性能预算与优化

### 8.1 每帧性能预算（目标 30 FPS = 33ms/帧）

| 阶段 | 当前耗时 | 优化后目标 | 优化手段 |
|------|---------|-----------|---------|
| YOLO 推理 | ~8ms | 8ms | 不改 |
| BEV 映射 | ~2ms | 2ms | 不改 |
| 冲突分析 | ~3ms | 3ms | 不改 |
| 热力图渲染 | ~4ms | **0.2ms** | `resize` 替代逐格循环 |
| 车辆渲染 | ~2ms | 2ms | 发光仅对 top-3，开销可忽略 |
| Phi Chart | ~1ms | 1.5ms | 增加填充+网格，net +0.5ms |
| 数据面板 | ~1ms | **0.5ms** | 卡片化后文本更少，无 CJK |
| 组合+显示 | ~2ms | 2ms | 不改 |
| **合计** | **~23ms** | **~19ms** | 净节省 4ms/帧，留更多余量 |

### 8.2 CJK 渲染优化

核心策略：**让 CJK 文本不出现在每帧渲染路径上**。

- 数据面板全部用 ASCII（英文标签 + 数字），`cv2.putText` 直接渲染，不走 PIL
- 仅在导出/截图/Streamlit 控制台中使用中文
- 如果需要中文数据面板：预渲染文字到纹理缓存（render once → reuse every frame）

```python
# 预渲染中文纹理（只在初始化时做一次）
_TEXT_CACHE = {}

def get_cjk_texture(text, font_size, color):
    key = (text, font_size, color)
    if key not in _TEXT_CACHE:
        # PIL 渲染一次，存入缓存
        pil_img = render_with_pil(text, font_size, color)
        _TEXT_CACHE[key] = pil_to_numpy(pil_img)
    return _TEXT_CACHE[key]
```

### 8.3 画布预分配

当前每帧都会创建新的 numpy 数组作为中间画布。改为预分配 + 复用：

```python
class CanvasPool:
    """预分配画布池，避免每帧 malloc"""
    def __init__(self):
        self._pool = {}
    
    def get(self, h, w):
        key = (h, w)
        if key not in self._pool:
            self._pool[key] = np.zeros((h, w, 3), dtype=np.uint8)
        canvas = self._pool[key]
        canvas[:] = THEME['bg_canvas']  # 重置为背景色
        return canvas
```

收益：避免每帧 `np.zeros((500, 1920, 3))` 的 ~3MB 内存分配。对 Python 的 GC 压力也有缓解。

---

## 九、实施路线

### 阶段 1：基础设施（30 分钟）

1. 新建 `utils/theme.py`：色板定义 + `phi_color()` + `attr_color()` + `lerp_bgr()`
2. 替换 `phi_chart.py`、`congestion_overlay.py`、`engine.py` 中的硬编码颜色引用
3. 新建 `utils/canvas_pool.py`：画布预分配

### 阶段 2：Phi Chart 重做（45 分钟）

1. 增加网格线 + 轴标签
2. 增加曲线下填充
3. 增加当前值 callout
4. 增加事件摘要条
5. `cv2.polylines` 批量替换逐段 `cv2.line`

### 阶段 3：数据面板卡片化（30 分钟）

1. 实现 `draw_card()` 函数
2. 实现 `draw_attribution_bar()` 进度条
3. 重构 `engine.py` 数据面板渲染块

### 阶段 4：BEV 热力图升级（15 分钟）

1. `cv2.resize` 渲染替换逐格循环
2. 换 `COLORMAP_VIRIDIS`
3. 添加颜色图例 `draw_colorbar()`

### 阶段 5：车辆视觉增强（15 分钟）

1. 三级视觉强调（低/中/高归因）
2. 发光效果（仅 top-3）
3. 速度感知箭头长度

---

## 十、验收标准

- [ ] 所有模块从 `utils/theme.py` 取色，无硬编码 BGR 值
- [ ] Phi Chart 有网格线、填充、轴标签、callout
- [ ] 数据面板使用卡片布局，视觉层次清晰
- [ ] BEV 热力图用 `cv2.resize` 渲染，平滑无抽稀感
- [ ] 热力图面板有颜色图例
- [ ] 高归因车辆视觉上明显区别于低归因车辆
- [ ] 每帧渲染总耗时 < 25ms（40 FPS 头部空间）
- [ ] 深色背景 + 莫兰迪色，投影仪下可读
- [ ] 纯 ASCII 文本路径不触发 PIL 渲染
