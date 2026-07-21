# 拥堵贡献度矩阵化计算方法

> 对应项目核心模块：`core/conflict.py` — 方向场卷积冲突检测  
> 对应项目版本：v2.0（方向场卷积替代成对几何）

---

## 1. 符号定义

| 符号 | 维度 | 含义 | 项目对应变量 |
|------|------|------|-------------|
| $G$ | 标量 | 栅格边长（默认 64） | `grid_size` |
| $N$ | 标量 | 当前帧车辆总数 | `len(vehicles)` |
| $K$ | 标量 | 方向箱数量（12） | `DIRECTION_BINS` |
| $P$ | 标量 | 冲突对数量（24） | `DEFAULT_CONFLICT_PAIRS` |
| $\texttt{O} \in \mathbb{R}^{G \times G}$ | 矩阵 | 占用场 | `O` |
| $\texttt{V} \in \mathbb{R}^{G \times G}$ | 矩阵 | 速度场 | `V` |
| $\texttt{O}_k \in \mathbb{R}^{G \times G}$ | 矩阵 | 方向 k 占用矩阵 | `layers[k]` |
| $\texttt{K}_k \in \mathbb{R}^{H \times W}$ | 矩阵 | 方向 k 各向异性核 | `kernels[k]` |
| $\texttt{R}_k \in \mathbb{R}^{G \times G}$ | 矩阵 | 方向 k 影响力场 | `R[k]` |
| $\texttt{C} \in \mathbb{R}^{G \times G}$ | 矩阵 | 冲突场 | `conflict_field` |
| $\Phi$ | 标量 | 拥堵指数 | `phi` |
| $\lambda_i$ | 标量 | 车辆 i 拥堵贡献度 | `influences[i]` |
| $\texttt{A} \in \mathbb{R}^{N \times N}$ | 稀疏矩阵 | 因果邻接矩阵 | `analysis/root_cause.py` |
| $\mathbf{x} \in \mathbb{R}^{N}$ | 向量 | 根因分数向量 | `root_cause_scores` |

---

## 2. 计算流程

### 2.1 方向场离散化与核矩阵构建

将 BEV 平面离散化为 $G \times G$ 均匀栅格，每帧 $N$ 辆车按世界坐标散布，构建**占用矩阵** $\texttt{O} \in \mathbb{R}^{G \times G}$：

$$
\texttt{O}[g_y, g_x] = \begin{cases}
1, & \text{若 } (g_x, g_y) = \text{grid}(c_x^i, c_y^i) \text{ 存在车辆 } i \\[2pt]
0, & \text{否则}
\end{cases}
$$

同时构建**速度场** $\texttt{V}$ 和**方向场** $\texttt{Theta}$（同格多车取平均）。

车辆连续航向角 $\theta_i$ 经**高斯软分配**至 $K=12$ 个方向箱（30° 每箱，$\sigma = 10°$），得到方向占用矩阵集合：

$$
\texttt{O}_k[g] = \sum_{i=1}^N \mathbb{1}_{\text{grid}(i)=g} \cdot w_k(\theta_i), \quad
w_k(\theta) = \exp\left(-\frac{(\theta - \theta_k)^2}{2\sigma^2}\right) \cdot \mathbb{1}_{\text{mask}(g) > 0}
$$

其中 $\theta_k = k \times 30^\circ$ 为方向箱 k 的中心角。

### 2.2 各向异性核卷积与影响力场

为每方向 k 预构建各向异性高斯核矩阵 $\texttt{K}_k \in \mathbb{R}^{H \times W}$：

$$
\texttt{K}_k[\Delta y, \Delta x] = \exp\left(-\frac{d_{\parallel}^2}{2\sigma_{\parallel}^2} - \frac{d_{\perp}^2}{2\sigma_{\perp}^2}\right), \quad
\sigma_{\parallel} = 3.0,\ \sigma_{\perp} = 0.6
$$

其中 $d_{\parallel}$、$d_{\perp}$ 分别为栅格距中心在方向 k 上的平行与垂直投影距离。前向范数 $3\sigma \approx 7\text{m}$，侧向 $3\sigma \approx 1.4\text{m}$，后向衰减速度 $3\times$ 前向。扇形因子 0.6 使前方越远侧向展宽越大。

通过二维卷积生成**方向影响力场矩阵**（核心算子）：

$$
\texttt{R}_k = \texttt{O}_k \circledast \texttt{K}_k, \quad \texttt{R}_k \in \mathbb{R}^{G \times G}
$$

以 `cv2.filter2D` 实现，天然 GPU 加速。此运算将原始 O(N²) 两两交织对计算的 $\Gamma_{ij} = \gamma_{ij} \times \beta_{ij}$ 替换为 **固定 K 次卷积 + 固定 P 次矩阵逐元素乘加**，复杂度由 O(N²) 降至 O(G²)。

### 2.3 冲突场矩阵合成

选取精简冲突方向对集合 $\mathcal{P} = \{(a, b)\}$（对向 6 + 正交 6 + 同向跟驰 12 = 24 对），构造冲突场矩阵：

$$
\texttt{C} = \sum_{(a, b) \in \mathcal{P}} \texttt{R}_a \odot \texttt{R}_b, \quad
\texttt{C} \in \mathbb{R}^{G \times G}
$$

其中 $\odot$ 为逐元素 Hadamard 积。$\texttt{C}[g]$ 值越大则栅格 $g$ 处交织强度越高，等价于原始专利分组公式中组内加权交织强度项 $\frac{2}{n_k(n_k-1)} \sum_{i=1}^{n_k} \sum_{j=i+1}^{n_k} \Gamma_{ij}$ 在空间上的密集化表达。传统成对几何需要处理 $N(N-1)/2$ 个 $\Gamma_{ij}$，而场合成仅需固定 24 个逐元素乘积。

### 2.4 Φ‑加权单车拥堵贡献度

**拥堵指数** $\Phi$ 融合密度与速度衰减：

$$
\Phi = w_\rho \cdot \min\left(1, \frac{N}{N_{\text{sat}}}\right) + w_v \cdot \max\left(0, 1 - \frac{v_{\text{avg}}}{v_{\text{ref}}}\right), \quad
\Phi \in [0, 1]
$$

默认参数：$N_{\text{sat}} = 40,\ v_{\text{ref}} = 5.0\text{m/s},\ w_\rho = 0.4,\ w_v = 0.6$。

记车辆 $i$ 的航向方向箱索引为 $k_i$，所处栅格位置为 $g_i = (g_y, g_x)$，则单车拥堵贡献度：

**边界条件**（$n_{k_i} \leq 1$，孤立或无交织车辆）：

$$
\lambda_i = \texttt{R}_{k_i}[g_i], \quad
\texttt{C}[g_i] = 0 \Rightarrow \lambda_i = 0
$$

即仅当冲突场在该位置有非零值时车辆方被计入拥堵贡献。

**多车核心公式**（$n_{k_i} \geq 2$，且 $\texttt{C}[g_i] > 0$）：

$$
\lambda_i = \texttt{R}_{k_i}[g_i] \times \sum_{k' \in \text{conflicts}(k_i)} \texttt{R}_{k'}[g_i]
$$

其中：
- $\texttt{R}_{k_i}[g_i]$：车辆自身路径影响力（对应原专利中 $\gamma_{id}$ 经空间平滑的等效值）
- $\sum_{k' \in \text{conflicts}(k_i)} \texttt{R}_{k'}[g_i]$：该位置上所有冲突方向的路径影响力之和（对应原专利中 $\frac{2\Phi}{\text{n}_k(\text{n}_k-1)}\sum\Gamma_{ij}$ 与干扰因子 T_ij 的集成）
- $\Phi$ 作为动态风险权重，替代原专利中的 $\Phi_t$
- 产品形式替代加和形式，因场值均为非负且冲突对已预选，乘积天然捕获"双向同时高影响"的物理语义

全局 **N 辆车的归因矩阵化**（一次 GPU 索引即可完成）：

$$
\boldsymbol{\lambda} = \text{diag}\big(\texttt{R}_{\mathbf{k}}[\mathbf{g}]\big) \cdot \text{diag}\Big(\sum_{k' \in \mathcal{P}(\mathbf{k})} \texttt{R}_{k'}[\mathbf{g}]\Big), \quad
\boldsymbol{\lambda} \in \mathbb{R}^N
$$

其中 $\mathbf{k} \in \mathbb{R}^N$ 为每辆车所属方向箱索引，$\mathbf{g} \in \mathbb{R}^{N \times 2}$ 为每辆车栅格坐标。

### 2.5 水滴传播因果溯源矩阵

归因完成后，基于冲突场 C 构造**稀疏邻接矩阵** $\texttt{A} \in \mathbb{R}^{N \times N}$，通过幂迭代定位拥堵根因：

$$
\texttt{A}_{ij} = \texttt{C}[\text{mid}(i,j)] \times \exp\left(-\frac{\|\mathbf{p}_i - \mathbf{p}_j\|_2^2}{2(D_{\max}/2)^2}\right) \times \mathbb{1}_{j \text{ 在 } i \text{ 前方}} \times \mathbb{1}_{\|\mathbf{p}_i - \mathbf{p}_j\|_2 < D_{\max}}
$$

其中 $D_{\max} = 15\text{m}$ 为最大搜索距离，$\text{mid}(i,j)$ 为两车中点栅格坐标，$\texttt{C}[\text{mid}]$ 值越大则因果传递强度越高。

行归一化后迭代传播：

$$
\texttt{A}_{\text{norm}} = \texttt{A} \oslash \max(\texttt{A} \cdot \mathbf{1}, \varepsilon), \quad
\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \cdot (\texttt{A}_{\text{norm}}^\mathsf{T} \cdot \mathbf{x}_t), \quad
\alpha = 0.4
$$

收敛后以冲突场加权得最终根因分数：

$$
\mathbf{x}^*[i] = \mathbf{x}_\infty[i] \times \min\big(\texttt{C}[g_i] \times 10, 1\big), \quad
\text{RootCausePct}[i] = \frac{\mathbf{x}^*[i]}{\sum \mathbf{x}^*} \times 100\%
$$

Top‑2 且 Influence > 10% 的车辆标记为红色根因车辆。

---

## 3. 算法对比

| 维度 | 原始专利方法（成对几何） | 本项目方法（方向场卷积） |
|------|------------------------|------------------------|
| 复杂度 | $O(N^2)$ | $O(G^2)$，**与 N 解耦** |
| GPU 加速 | 难以并行 | 原生支持（cv2::filter2D） |
| 冲突对数量 | $N(N-1)/2$ | 固定 24 对 |
| 消融重算 | 全量 $O(N^2)$ | $O(1)$ — 单 bin 单次卷积 |
| 分组方式 | 人工定义第 k 组 | 12 方向箱自动分组 |
| 交织强度 $\Gamma_{ij}$ | 两两枚举 | 场合成 $\texttt{C} = \sum \texttt{R}_a \odot \texttt{R}_b$ |
| 单车归因 | 分组级 $\lambda_k$ 均值分摊 | 单车级 $\lambda_i$ 精确定位 |
