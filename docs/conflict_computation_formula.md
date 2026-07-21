# 冲突计算数学公式

> 方向场卷积冲突检测的完整矩阵推导

---

## 1. 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $G$ | 栅格边长 | 标量，默认 $G=64$ |
| $K$ | 方向箱数 | 标量，$K=12$ |
| $N$ | 车辆数 | 每帧动态 |
| $\mathbf{O}$ | 占用场 | $\mathbb{R}^{G \times G}$ |
| $\mathbf{V}$ | 速度场 | $\mathbb{R}^{G \times G}$ |
| $\boldsymbol{\Theta}$ | 方向场 | $\mathbb{R}^{G \times G}$ |
| $\mathbf{O}_k$ | 第 $k$ 个方向箱的占用场 | $\mathbb{R}^{G \times G}$ |
| $\mathbf{K}_k$ | 第 $k$ 个方向箱的各向异性核 | $\mathbb{R}^{S \times S}$，$S=2H+1$ |
| $\mathbf{R}_k$ | 第 $k$ 个方向的路径影响力场 | $\mathbb{R}^{G \times G}$ |
| $\mathbf{C}$ | 冲突场 | $\mathbb{R}^{G \times G}$ |
| $\mathbf{P}$ | 冲突对集合 | $\mathbb{Z}_K \times \mathbb{Z}_K$，$|\mathbf{P}|=24$ |

---

## 2. 散布：车辆 → 栅格场

单辆车 $i$ 在位置 $(x_i, y_i)$ 散布到网格：

$$
g_x^{(i)} = \left\lfloor \frac{x_i - x_{\text{origin}}}{\Delta x} \right\rfloor,\quad
g_y^{(i)} = \left\lfloor \frac{y_i - y_{\text{origin}}}{\Delta y} \right\rfloor
$$

$$
\mathbf{O}(g_y, g_x) = \sum_{i=1}^{N} \delta(g_y - g_y^{(i)})\;\delta(g_x - g_x^{(i)})
$$

$$
\mathbf{V}(g_y, g_x) = \frac{\sum_i v_i \cdot \delta(g_y - g_y^{(i)})\;\delta(g_x - g_x^{(i)})}{\mathbf{O}(g_y, g_x)}
$$

$$
\boldsymbol{\Theta}(g_y, g_x) = \arctan\left(
\frac{\sum_i \sin\theta_i \cdot \delta(\ldots)}{\sum_i \cos\theta_i \cdot \delta(\ldots)}
\right)
$$

---

## 3. 方向分解（软分配）

将方向场 $\boldsymbol{\Theta}$ 分解为 $K$ 个方向箱：

$$
\mathbf{O}_k(g_y, g_x) = \mathbf{W}_k(g_y, g_x) \cdot \mathbf{M}(g_y, g_x)
$$

其中 $\mathbf{M}$ 是有效掩膜（$\mathbf{O} > 0$ 的位置为 1），$\mathbf{W}_k$ 是高斯权重：

$$
\mathbf{W}_k(g_y, g_x) = \exp\left(-\frac{1}{2} \cdot \frac{\Delta\theta_k(g_y, g_x)^2}{\sigma^2}\right)
$$

$$
\Delta\theta_k(g_y, g_x) = \min\left(
\bigl|\boldsymbol{\Theta}(g_y, g_x) - \theta_k\bigr|,\;
360^\circ - \bigl|\boldsymbol{\Theta}(g_y, g_x) - \theta_k\bigr|
\right)
$$

$$
\theta_k = k \cdot \frac{360^\circ}{K} \quad (k = 0, 1, \ldots, K-1)
$$

$$
\sigma = \frac{360^\circ}{3K}
$$

---

## 4. 各向异性核构建

对方向箱 $k$，核 $\mathbf{K}_k$ 定义如下（经 $\text{flip}(·, -1)$ 预翻转以抵消 $\text{filter2D}$ 的翻转）：

$$
\mathbf{K}_k(u, v) = A \cdot f_{\text{along}}(u, v) \cdot f_{\text{perp}}(u, v)
$$

$A$ 为归一化因子使 $\sum_{u,v} \mathbf{K}_k(u, v) = 1$。

**沿方向投影**：

$$
\begin{pmatrix} u_{\|} \\ u_\perp \end{pmatrix} =
\begin{pmatrix} \cos\phi_k & -\sin\phi_k \\ \sin\phi_k & \cos\phi_k \end{pmatrix}
\begin{pmatrix} u \\ v \end{pmatrix}
$$

其中 $\phi_k$ 为方向箱 $k$ 的角度（经 $y$ 轴取反校正）。

**速度感知的沿方向衰减**：

$$
\sigma_{\text{along}}(v_i) = \sigma_0 \cdot \left(1 + \alpha \cdot \frac{v_i}{v_{\text{ref}}}\right)
$$

其中 $\sigma_0 = 3.0$ cells, $\alpha = 1.0$, $v_{\text{ref}} = 5.0$ m/s。

$$
f_{\text{along}}(u, v) =
\begin{cases}
\exp\left(-\dfrac{1}{2} \cdot \dfrac{u_\parallel^2}{\sigma_{\text{along}}^2}\right), & u_\parallel \ge 0 \text{ (前向)} \\[6pt]
\exp\left(-\dfrac{1}{2} \cdot \dfrac{u_\parallel^2}{(0.33 \cdot \sigma_{\text{along}})^2}\right), & u_\parallel < 0 \text{ (后向)}
\end{cases}
$$

**扇形展开的垂直方向衰减**：

$$
\sigma_{\text{perp}}(u_\parallel) = \sigma_{\perp0} \cdot
\begin{cases}
1 + 0.6 \cdot \dfrac{u_\parallel}{\sigma_{\text{along}}}, & u_\parallel \ge 0 \\[6pt]
1, & u_\parallel < 0
\end{cases}
$$

$$
f_{\text{perp}}(u, v) = \exp\left(-\frac{1}{2} \cdot \frac{u_\perp^2}{\sigma_{\text{perp}}(u_\parallel)^2}\right)
$$

**截断条件**：

核仅在 $\|u_\parallel\| \le L_{\text{eff}}$ 范围内非零，其中：

$$
L_{\text{eff}} =
\begin{cases}
H, & u_\parallel \ge 0 \\[3pt]
0.33 \cdot H, & u_\parallel < 0
\end{cases}
$$

$H = 10$ cells 为核半长。

---

## 5. 路径影响力场（卷积）

第 $k$ 个方向的影响力场由 $\mathbf{O}_k$ 与核 $\mathbf{K}_k$ 做卷积得到：

$$
\mathbf{R}_k = \mathbf{O}_k \circledast \mathbf{K}_k
$$

该卷积通过 OpenCV 的 `cv2.filter2D` 实现：

$$
\mathbf{R}_k(x, y) = \sum_{u=-H}^{H} \sum_{v=-H}^{H}
\mathbf{O}_k(x+u, y+v) \cdot \tilde{\mathbf{K}}_k(u, v)
$$

其中 $\tilde{\mathbf{K}}_k = \text{flip}(\mathbf{K}_k, -1)$，即 $\tilde{\mathbf{K}}_k(u, v) = \mathbf{K}_k(-u, -v)$。

**输出**： $K$ 个 $G \times G$ 矩阵 $\mathbf{R}_0, \mathbf{R}_1, \ldots, \mathbf{R}_{K-1}$。

---

## 6. 冲突场计算

冲突场定义为所有冲突对的方向影响力场逐元素乘积之和：

$$
\mathbf{C} = \sum_{(a,b) \in \mathbf{P}} \mathbf{R}_a \odot \mathbf{R}_b
$$

其中 $\odot$ 表示逐元素乘法（Hadamard 积）。用标量索引表示：

$$
\mathbf{C}(x, y) = \sum_{p=1}^{|\mathbf{P}|} \mathbf{R}_{a_p}(x, y) \cdot \mathbf{R}_{b_p}(x, y)
$$

**冲突对定义**（24 对）：

$$
\begin{aligned}
\mathbf{P}_{\text{opp}} &= \{(0,6), (1,7), (2,8), (3,9), (4,10), (5,11)\} \\
\mathbf{P}_{\text{orth}} &= \{(0,3), (1,4), (2,5), (3,6), (4,7), (5,8)\} \\
\mathbf{P}_{\text{same}} &= \{(0,0), (1,1), \ldots, (11,11)\}
\end{aligned}
$$

$$
\mathbf{P} = \mathbf{P}_{\text{opp}} \cup \mathbf{P}_{\text{orth}} \cup \mathbf{P}_{\text{same}}
$$

---

## 7. 单车归因

第 $i$ 辆车的归因分数由其所在网格位置 $(g_x^{(i)}, g_y^{(i)})$ 处的冲突场值决定：

**方向箱索引**：

$$
k_i = \left\lfloor \frac{\theta_i \bmod 360^\circ}{360^\circ / K} \right\rfloor \bmod K
$$

**冲突方向集合**：属于同一冲突对的对方方向

$$
\Psi(k_i) = \{b \mid (k_i, b) \in \mathbf{P} \;\text{或}\; (b, k_i) \in \mathbf{P}\}
$$

**归因分数**：

$$
\text{Influence}_i = \mathbf{R}_{k_i}(g_y^{(i)}, g_x^{(i)}) \cdot
\sum_{k' \in \Psi(k_i)} \mathbf{R}_{k'}(g_y^{(i)}, g_x^{(i)})
$$

第一项是车辆自身方向的影响场在自身位置的值；第二项是所有冲突方向在该位置的场值之和。两者相乘——自身影响越强、冲突方向的干扰越多 → 归因越高。

**归一化为百分比**：

$$
\text{Influence\%}_i = \frac{\text{Influence}_i}{\max_j \text{Influence}_j} \times 100\%
$$

---

## 8. 水滴传播因果溯源

### 8.1 邻接矩阵

构建 $N \times N$ 邻接矩阵 $\mathbf{A}$，$\mathbf{A}_{ij}$ 表示水从车辆 $i$ 流向车辆 $j$ 的强度：

$$
\mathbf{A}_{ij} = \mathbf{C}_{\text{mid}}(i, j) \cdot d_{ij}
$$

其中：

$$
\mathbf{C}_{\text{mid}}(i, j) = \mathbf{C}\left(\left\lfloor\frac{y_i + y_j}{2\Delta y}\right\rfloor,\;
\left\lfloor\frac{x_i + x_j}{2\Delta x}\right\rfloor\right)
$$

$$
d_{ij} = \exp\left(-\frac{1}{2} \cdot \frac{\|p_i - p_j\|_2^2}{(0.5 \cdot D_{\max})^2}\right)
$$

**约束条件**：仅当车辆 $j$ 在车辆 $i$ 的前方（正投影距离 $> 0$）且两车中点处冲突场值 $\mathbf{C}_{\text{mid}} > 0$ 时，$\mathbf{A}_{ij}$ 非零。

### 8.2 行归一化

$$
\tilde{\mathbf{A}}_{ij} = \frac{\mathbf{A}_{ij}}{\max(\sum_j \mathbf{A}_{ij},\; \varepsilon)},\quad \varepsilon = 10^{-10}
$$

### 8.3 迭代传播

$$
\mathbf{x}^{(0)} = [1, 1, \ldots, 1]^{\mathsf{T}} \in \mathbb{R}^N
$$

$$
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \alpha \cdot \tilde{\mathbf{A}}^{\mathsf{T}} \mathbf{x}^{(t)}
$$

其中 $\alpha = 0.4$，迭代次数 $T = \max(5, \lfloor 1.5N \rfloor)$。

### 8.4 冲突场加权

$$
\tilde{x}_i = x_i^{(T)} \cdot \min\left(10 \cdot \mathbf{C}(g_y^{(i)}, g_x^{(i)}),\; 1.0\right)
$$

### 8.5 归一化

$$
\text{RootCause\%}_i = \frac{\tilde{x}_i}{\sum_j \tilde{x}_j} \times 100\%
$$

---

## 9. 高亮判别准则

| 颜色 | 条件 |
|------|------|
| 🔴 红色 | $\text{RootCause\%}_i$ 为 Top-2 且 $\text{Influence\%}_i > 5\%$ 且 $\mathbf{C}(g_y^{(i)}, g_x^{(i)}) > 0$ |
| 🟠 橙色 | $\text{Influence\%}_i > 15\%$ |
| ⚪ 灰色 | 其他 |

---

## 10. 计算复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| 散布 | $O(N)$ | 线性于车辆数 |
| 方向分解 | $O(K \cdot G^2)$ | $K=12, G=64$ |
| 卷积 | $O(K \cdot G^2)$ | $\text{cv2.filter2D}$，GPU 加速 |
| 冲突场 | $O(|\mathbf{P}| \cdot G^2)$ | $|\mathbf{P}|=24$ |
| 归因 | $O(N \cdot K)$ | 线性于车辆数 |
| 水滴传播 | $O(T \cdot N^2)$ | $T \approx 15$，但 $\mathbf{A}$ 稀疏 |
| **总复杂度** | $\mathbf{O(G^2)}$ | **不随车辆数 $N$ 增长** |
