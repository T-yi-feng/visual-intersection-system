"""
公式渲染脚本 — 将 LaTeX 公式导出为高分辨率 PNG

用法: python utils/render_formulas.py
输出: docs/formulas/*.png (透明背景, 600 DPI)
"""

import sys, re, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "formulas"
OUT.mkdir(parents=True, exist_ok=True)

# ── 需要渲染的公式列表 ──
FORMULAS = [
    # ---- 散布 ----
    ("O_def", r"$\mathbf{O}(g_y, g_x) = \sum_{i=1}^{N} \delta(g_y - g_y^{(i)})\;\delta(g_x - g_x^{(i)})$"),
    ("V_def", r"$\mathbf{V}(g_y, g_x) = \frac{\sum_i v_i \cdot \delta(g_y - g_y^{(i)})\;\delta(g_x - g_x^{(i)})}{\mathbf{O}(g_y, g_x)}$"),
    ("Theta_def", r"$\boldsymbol{\Theta}(g_y, g_x) = \arctan\left(\frac{\sum_i \sin\theta_i \cdot \delta(\ldots)}{\sum_i \cos\theta_i \cdot \delta(\ldots)}\right)$"),

    # ---- 软分配 ----
    ("soft_assign", r"$\mathbf{O}_k(g_y, g_x) = \mathbf{W}_k(g_y, g_x) \cdot \mathbf{M}(g_y, g_x)$"),
    ("soft_weight", r"$\mathbf{W}_k(g_y, g_x) = \exp\left(-\frac{1}{2} \cdot \frac{\Delta\theta_k(g_y, g_x)^2}{\sigma^2}\right)$"),
    ("theta_k", r"$\theta_k = k \cdot \frac{360^\circ}{K} \quad (k = 0, 1, \ldots, K-1)$"),

    # ---- 核 ----
    ("kernel_def", r"$\mathbf{K}_k(u, v) = A \cdot f_{\text{along}}(u, v) \cdot f_{\text{perp}}(u, v)$"),
    ("kernel_rot", r"$\begin{pmatrix} u_\parallel \\ u_\perp \end{pmatrix} = \begin{pmatrix} \cos\phi_k & -\sin\phi_k \\ \sin\phi_k & \cos\phi_k \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}$"),
    ("sigma_speed", r"$\sigma_{\text{along}}(v_i) = \sigma_0 \cdot \left(1 + \alpha \cdot \frac{v_i}{v_{\text{ref}}}\right)$"),
    ("fan_shape", r"$\sigma_{\text{perp}}(u_\parallel) = \sigma_{\perp0} \cdot \begin{cases} 1 + 0.6 \cdot \frac{u_\parallel}{\sigma_{\text{along}}}, & u_\parallel \ge 0 \\ 1, & u_\parallel < 0 \end{cases}$"),
    ("f_along", r"$f_{\text{along}}(u, v) = \begin{cases} \exp\left(-\frac{1}{2} \cdot \frac{u_\parallel^2}{\sigma_{\text{along}}^2}\right), & u_\parallel \ge 0 \\[6pt] \exp\left(-\frac{1}{2} \cdot \frac{u_\parallel^2}{(0.33 \cdot \sigma_{\text{along}})^2}\right), & u_\parallel < 0 \end{cases}$"),
    ("f_perp", r"$f_{\text{perp}}(u, v) = \exp\left(-\frac{1}{2} \cdot \frac{u_\perp^2}{\sigma_{\text{perp}}(u_\parallel)^2}\right)$"),

    # ---- 卷积 ----
    ("conv_Rk", r"$\mathbf{R}_k = \mathbf{O}_k \circledast \mathbf{K}_k$"),
    ("conv_filter", r"$\mathbf{R}_k(x, y) = \sum_{u=-H}^{H} \sum_{v=-H}^{H} \mathbf{O}_k(x+u, y+v) \cdot \tilde{\mathbf{K}}_k(u, v)$"),

    # ---- 冲突场 ----
    ("conflict_field", r"$\mathbf{C} = \sum_{(a,b) \in \mathbf{P}} \mathbf{R}_a \odot \mathbf{R}_b$"),
    ("conflict_scalar", r"$\mathbf{C}(x, y) = \sum_{p=1}^{|\mathbf{P}|} \mathbf{R}_{a_p}(x, y) \cdot \mathbf{R}_{b_p}(x, y)$"),
    ("pairs_def", r"$\mathbf{P} = \mathbf{P}_{\text{opp}} \cup \mathbf{P}_{\text{orth}} \cup \mathbf{P}_{\text{same}}$"),

    # ---- 归因 ----
    ("influence", r"$\text{Influence}_i = \mathbf{R}_{k_i}(g_y^{(i)}, g_x^{(i)}) \cdot \sum_{k' \in \Psi(k_i)} \mathbf{R}_{k'}(g_y^{(i)}, g_x^{(i)})$"),
    ("influence_pct", r"$\text{Influence\%}_i = \frac{\text{Influence}_i}{\max_j \text{Influence}_j} \times 100\%$"),

    # ---- 水滴传播 ----
    ("A_def", r"$\mathbf{A}_{ij} = \mathbf{C}_{\text{mid}}(i, j) \cdot \exp\left(-\frac{1}{2} \cdot \frac{\|p_i - p_j\|_2^2}{(0.5 \cdot D_{\max})^2}\right)$"),
    ("A_norm", r"$\tilde{\mathbf{A}}_{ij} = \frac{\mathbf{A}_{ij}}{\max(\sum_j \mathbf{A}_{ij},\; \varepsilon)}$"),
    ("water_iter", r"$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + \alpha \cdot \tilde{\mathbf{A}}^{\mathsf{T}} \mathbf{x}^{(t)}$"),
    ("water_weight", r"$\tilde{x}_i = x_i^{(T)} \cdot \min\left(10 \cdot \mathbf{C}(g_y^{(i)}, g_x^{(i)}),\; 1.0\right)$"),
    ("rootcause_pct", r"$\text{RootCause\%}_i = \frac{\tilde{x}_i}{\sum_j \tilde{x}_j} \times 100\%$"),

    # ---- Phi ----
    ("phi_def", r"$\Phi = w_\rho \cdot \min\left(1, \frac{N}{N_{\text{sat}}}\right) + w_v \cdot \max\left(0, 1 - \frac{v_{\text{avg}}}{v_{\text{ref}}}\right)$"),

    # ---- 复杂度 ----
    ("complexity", r"$\text{总复杂度} = \mathbf{O(G^2)}$"),
]


def render_to_png(tex: str, path: Path, dpi: int = 300):
    """用 matplotlib 将 LaTeX 公式渲染为透明背景 PNG"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(len(tex) * 0.05, 0.7))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${tex.strip('$')}$", fontsize=20, ha='center', va='center',
            transform=ax.transAxes)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=True,
                pad_inches=0.15, facecolor='none', edgecolor='none')
    plt.close(fig)


def main():
    print(f"Rendering {len(FORMULAS)} formulas to {OUT}/ ...")
    print("  (using matplotlib built-in math renderer)")
    for name, tex in FORMULAS:
        try:
            path = OUT / f"{name}.png"
            render_to_png(tex, path)
            print(f"  OK: {name}.png")
        except Exception as e:
            print(f"  FAIL: {name}: {e}")

    print(f"\nDone! {len(list(OUT.glob('*.png')))} images in {OUT}")


if __name__ == '__main__':
    main()
