"""
核分析 — 各向异性扇形核的多维度可视化

运行: python analysis/experiments/kernel_analysis.py
输出: docs/experiment_results/kernel_*.png
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "experiment_results"

import numpy as np
from math import exp, cos, sin, pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib import rcParams
rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

C_BLUE = '#5b9bd5'
C_ORANGE = '#d4956b'
C_GREEN = '#6baf6b'
C_RED = '#d55b5b'
C_PURPLE = '#9b8ebf'
C_GOLD = '#e2b96f'
C_CYAN = '#49d9ff'
C_GRAY = '#999999'

# 当前系统的默认核参数
SIGMA_ALONG = 3.0
SIGMA_PERP = 0.6
HALF_LEN = 10
HALF_WIDTH = 6
CELL_M = 0.78  # m/cell


def build_kernel(heading_deg, sigma_a=SIGMA_ALONG, sigma_p=SIGMA_PERP,
                 half_len=HALF_LEN, half_w=HALF_WIDTH, fan=True):
    """构建各向异性核（支持等宽和扇形）"""
    k_size = 2 * half_len + 1
    K = np.zeros((k_size, k_size), dtype=np.float32)
    cx = cy = half_len
    theta = heading_deg * pi / 180.0
    ux, uy = cos(theta), -sin(theta)
    nx, ny = sin(theta), cos(theta)
    for i in range(k_size):
        for j in range(k_size):
            dx, dy = j - cx, i - cy
            along = dx * ux + dy * uy
            perp = dx * nx + dy * ny
            if along >= 0:
                eff_sigma = sigma_a
                eff_half = half_len
            else:
                eff_sigma = sigma_a * 0.33
                eff_half = half_len * 0.33
            f_along = exp(-0.5 * (along / eff_sigma) ** 2) if abs(along) <= eff_half else 0.0
            if fan and along >= 0:
                fan_s = sigma_p * (1.0 + 0.6 * along / max(sigma_a, 1e-6))
            else:
                fan_s = sigma_p
            f_perp = exp(-0.5 * (perp / max(fan_s, 0.1)) ** 2)
            K[i, j] = f_along * f_perp
    K /= K.sum()
    return K


# ═══════════════════════════════════════════════════════════════
# Fig 1: 核 3D 曲面 — 前后不对称 + 扇形展开
# ═══════════════════════════════════════════════════════════════

def fig1_3d_surface():
    """3D 曲面展示核的前后不对称和扇形展开"""
    K = build_kernel(0, fan=True)
    k_size = K.shape[0]
    X, Y = np.meshgrid(np.arange(k_size) - HALF_LEN, np.arange(k_size) - HALF_LEN)
    Z = K

    fig = plt.figure(figsize=(14, 5.5))

    # 3D 曲面
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none',
                            alpha=0.92, antialiased=True, linewidth=0)
    ax1.set_xlabel('Lateral (cells)', fontsize=10, labelpad=8)
    ax1.set_ylabel('Forward (cells)', fontsize=10, labelpad=8)
    ax1.set_zlabel('Amplitude', fontsize=10, labelpad=6)
    ax1.set_title('3D Surface of Anisotropic Fan Kernel\n(heading=0°, forward=right)',
                  fontsize=11, fontweight='bold')
    ax1.view_init(elev=28, azim=-65)
    fig.colorbar(surf, ax=ax1, shrink=0.6, label='Normalized Weight')

    # 俯视热力图 + 标注
    ax2 = fig.add_subplot(122)
    extent = [-HALF_LEN, HALF_LEN, -HALF_LEN, HALF_LEN]
    im = ax2.imshow(K, cmap='viridis', origin='lower', extent=extent, interpolation='bilinear')
    fig.colorbar(im, ax=ax2, shrink=0.8, label='Weight')

    # 标注关键尺寸
    ax2.plot([0, SIGMA_ALONG], [0, 0], 'w-', linewidth=2, alpha=0.8)
    ax2.annotate(f'   sigma_along={SIGMA_ALONG}cell\n   (3sigma~{3*SIGMA_ALONG*CELL_M:.1f}m)',
                 xy=(SIGMA_ALONG, 0), xytext=(SIGMA_ALONG+0.5, 3), fontsize=8, color='white', fontweight='bold')

    ax2.plot([0, 0], [-SIGMA_PERP, SIGMA_PERP], 'w--', linewidth=1.5, alpha=0.6)
    ax2.annotate(f'sigma_perp={SIGMA_PERP}cell',
                 xy=(0, SIGMA_PERP), xytext=(1.5, SIGMA_PERP+0.3), fontsize=8, color='white')

    # 前向/后向标注
    ax2.annotate('FORWARD\n(slow decay)', xy=(HALF_LEN*0.6, 0), fontsize=9, color='#88ff88', fontweight='bold', ha='center')
    ax2.annotate('BACKWARD\n(fast decay, 0.33x)', xy=(-HALF_LEN*0.5, 0), fontsize=8, color='#ff8888', fontweight='bold', ha='center')

    # 扇形展开标注
    fan_w = SIGMA_PERP * (1 + 0.6 * SIGMA_ALONG / SIGMA_ALONG)
    ax2.plot([SIGMA_ALONG, SIGMA_ALONG], [-fan_w, fan_w], 'c-', linewidth=1.5, alpha=0.7)
    ax2.plot([0, SIGMA_ALONG], [SIGMA_PERP, fan_w], 'c:', linewidth=1, alpha=0.5)
    ax2.plot([0, SIGMA_ALONG], [-SIGMA_PERP, -fan_w], 'c:', linewidth=1, alpha=0.5)
    ax2.annotate('Fan expansion\n1.6x wider at sigma_along',
                 xy=(SIGMA_ALONG, fan_w), xytext=(SIGMA_ALONG+2, fan_w+1.5),
                 fontsize=7, color=C_CYAN, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_CYAN, lw=0.8))

    ax2.set_xlabel('Lateral (cells)', fontsize=10)
    ax2.set_ylabel('Forward (cells)', fontsize=10)
    ax2.set_title('Kernel Top View with Key Dimensions', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')

    fig.tight_layout()
    path = OUT / "kernel_3d_surface.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 2: 12 方向核的极坐标响应 + 各向异性轮廓
# ═══════════════════════════════════════════════════════════════

def fig2_polar_response():
    """12 个方向核的极坐标方向响应曲线 + 各向异性轮廓对比"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    # 12 个方向 + 4 个选中的放大轮廓
    highlight = [0, 3, 6, 9]  # N, E, S, W
    colors_ring = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GOLD,
                   C_CYAN, '#ff6b6b', '#6bff6b', '#ffb86b', '#6bb5ff', '#d66bff']

    # 左上：12 个核缩略图排布
    for k in range(12):
        K = build_kernel(k * 30, fan=True)
        ax = axes[0]
        # 只画 4 个主方向的
        if k in highlight:
            idx = highlight.index(k)
            pos = [(0, 0), (0, 1), (1, 0), (1, 1)][idx]
            ax_sub = axes[pos[0]*3 + pos[1] + 1]
            ax_sub.imshow(K, cmap='viridis', origin='lower', interpolation='bilinear')
            ax_sub.set_title(f'{k*30}° (bin {k})', fontsize=9, fontweight='bold')
            ax_sub.axis('off')

    # 第5个小图：0°核的截面曲线
    K0 = build_kernel(0, fan=True)
    K0_uni = build_kernel(0, fan=False)
    mid = K0.shape[0] // 2
    ax_section = axes[5]
    ax_section.plot(K0[mid, :], '-', color=C_BLUE, linewidth=2.5, label='Fan kernel (ours)')
    ax_section.plot(K0_uni[mid, :], '--', color=C_ORANGE, linewidth=2, label='Uniform kernel')
    ax_section.axvline(x=HALF_LEN, color=C_GRAY, linestyle=':', alpha=0.5)
    ax_section.axvline(x=HALF_LEN - HALF_LEN // 3, color=C_GRAY, linestyle=':', alpha=0.5)
    ax_section.annotate('forward\ncutoff', xy=(HALF_LEN, max(K0[mid, :])*0.5), fontsize=7, color=C_GRAY)
    ax_section.set_xlabel('Position along direction (cells)', fontsize=9)
    ax_section.set_ylabel('Amplitude', fontsize=9)
    ax_section.set_title('Forward Profile: Uniform vs Fan', fontsize=10, fontweight='bold')
    ax_section.legend(fontsize=8)
    ax_section.grid(True, alpha=0.2)

    axes[0].axis('off')
    fig.suptitle('Kernel Shapes for 12 Direction Bins (30° intervals)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = OUT / "kernel_12_directions.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 3: 速度感知核 — 不同速度下的核形态变化
# ═══════════════════════════════════════════════════════════════

def fig3_speed_aware():
    """展示速度感知核：不同车速下 sigma_along（核长度）的变化"""
    speeds = [0, 2, 5, 10, 15]  # m/s
    labels = ['0 m/s (stopped)', '2 m/s (crawl)', '5 m/s (city)', '10 m/s (fast)', '15 m/s (highway)']
    alpha = 1.0
    v_ref = 5.0

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for si, (v, label) in enumerate(zip(speeds, labels)):
        sigma_v = SIGMA_ALONG * (1 + alpha * v / v_ref)
        K = build_kernel(0, sigma_a=sigma_v, fan=True)
        axes[si].imshow(K, cmap='viridis', origin='lower', interpolation='bilinear')
        axes[si].set_title(f'{label}\nsigma_fwd={sigma_v:.1f}cell ({sigma_v*CELL_M:.1f}m)',
                           fontsize=8, fontweight='bold')
        axes[si].axis('off')

    fig.suptitle('Speed-Adaptive Kernel: sigma_along(v) = sigma_0 * (1 + alpha * v / v_ref)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = OUT / "kernel_speed_aware.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 4: 核参数扫掠 — (sigma_along, sigma_perp) 组合对比
# ═══════════════════════════════════════════════════════════════

def fig4_parameter_sweep():
    """sigma_along vs sigma_perp 的组合扫掠"""
    sa_vals = [1, 2, 3, 5, 8]
    sp_vals = [0.3, 0.6, 1.0, 1.5]

    fig, axes = plt.subplots(len(sp_vals), len(sa_vals), figsize=(14, 10))
    for ri, sp in enumerate(sp_vals):
        for ci, sa in enumerate(sa_vals):
            K = build_kernel(0, sigma_a=sa, sigma_p=sp, fan=True)
            axes[ri][ci].imshow(K, cmap='viridis', origin='lower', interpolation='bilinear')
            axes[ri][ci].axis('off')
            if ri == 0:
                axes[ri][ci].set_title(f'sa={sa}', fontsize=9, fontweight='bold')
            if ci == 0:
                axes[ri][ci].set_ylabel(f'sp={sp}', fontsize=9, fontweight='bold', rotation=90, labelpad=6)

    fig.suptitle('Kernel Shape: (sigma_along, sigma_perp) Sweep  |  Current selection: sa=3, sp=0.6',
                 fontsize=13, fontweight='bold')
    # 标记当前选择
    ri_sel = sp_vals.index(0.6)
    ci_sel = sa_vals.index(3)
    for r in range(len(sp_vals)):
        for c in range(len(sa_vals)):
            if r == ri_sel and c == ci_sel:
                axes[r][c].patch.set_edgecolor(C_RED)
                axes[r][c].patch.set_linewidth(3)
                axes[r][c].patch.set_facecolor('none')
                axes[r][c].text(0.5, -0.08, 'SELECTED', transform=axes[r][c].transAxes,
                               fontsize=8, color=C_RED, fontweight='bold', ha='center')

    fig.tight_layout()
    path = OUT / "kernel_parameter_sweep.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 5: 核的有效影响区域 — 不同阈值下的等高线
# ═══════════════════════════════════════════════════════════════

def fig5_contour_footprint():
    """核在不同阈值下的等高线（有效影响范围）"""
    thresholds = [0.5, 0.2, 0.1, 0.05, 0.01]
    colors_list = ['#ff0000', '#ff8800', '#ffff00', '#00ff00', '#0088ff']
    styles = ['-', '--', '-.', ':', ':']
    assert len(thresholds) == len(colors_list) == len(styles), 'lengths must match'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ai, (heading, title) in enumerate(zip([0, 45, 90], ['0° (East)', '45° (NE)', '90° (North)'])):
        K = build_kernel(heading, fan=True)
        ax = axes[ai]

        # 基准图
        ax.imshow(K, cmap='gray_r', origin='lower', interpolation='bilinear', alpha=0.3,
                  extent=[-HALF_LEN, HALF_LEN, -HALF_LEN, HALF_LEN])

        k_max = K.max()
        for ti, th in enumerate(thresholds):
            level = k_max * th
            cs = ax.contour(K, levels=[level], colors=colors_list[ti], linewidths=1.5,
                           linestyles=styles[ti % len(styles)],
                           extent=[-HALF_LEN, HALF_LEN, -HALF_LEN, HALF_LEN])
            # 绘制第一条等高线的图例
            if ai == 0 and ti == 0:
                from matplotlib.lines import Line2D
                pass  # 图例在循环外用 proxy 处理

        # 箭头标注方向
        rad = heading * pi / 180
        ax.annotate('', xy=(HALF_LEN*0.8*cos(rad), HALF_LEN*0.8*(-sin(rad))),
                    xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='white', lw=2))
        ax.text(HALF_LEN*0.5, HALF_LEN*0.1, 'forward', fontsize=7, color='white', fontweight='bold')

        ax.set_xlim(-HALF_LEN, HALF_LEN)
        ax.set_ylim(-HALF_LEN, HALF_LEN)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('cells', fontsize=9)
        ax.set_ylabel('cells', fontsize=9)
        ax.grid(True, alpha=0.15)
        if ai == 0:
            from matplotlib.lines import Line2D
            proxy = [Line2D([0], [0], color=c, linewidth=1.5, linestyle=s)
                     for c, s in zip(colors_list[:len(thresholds)], styles[:len(thresholds)])]
            ax.legend(proxy, [f'{th*100:.0f}% peak' for th in thresholds],
                     fontsize=6, loc='upper right')

    fig.suptitle('Influence Region Footprint at Various Thresholds', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = OUT / "kernel_contour_footprint.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 6: 前后不对称率 — sigma_along 对 front/back 比率的影响
# ═══════════════════════════════════════════════════════════════

def fig6_asymmetry_ratio():
    """不同 sigma_along 下的前/后不对称比率"""
    sa_range = np.linspace(1, 10, 19)
    ratios = []
    for sa in sa_range:
        K = build_kernel(0, sigma_a=sa, fan=True)
        mid = K.shape[0] // 2
        fwd = K[mid, mid:].sum()
        bwd = K[mid, :mid].sum()
        ratios.append(fwd / max(bwd, 1e-10))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sa_range, ratios, '-', color=C_BLUE, linewidth=2.5)
    ax.axhline(y=3.0, color=C_GRAY, linestyle='--', alpha=0.4)
    ax.axvline(x=SIGMA_ALONG, color=C_RED, linestyle=':', alpha=0.6)
    ax.annotate(f'sigma_along={SIGMA_ALONG}\nratio={np.interp(SIGMA_ALONG, sa_range, ratios):.1f}x',
                xy=(SIGMA_ALONG, np.interp(SIGMA_ALONG, sa_range, ratios)),
                xytext=(SIGMA_ALONG+1, np.interp(SIGMA_ALONG, sa_range, ratios)+1),
                fontsize=10, color=C_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    ax.set_xlabel('sigma_along (cells)', fontsize=11)
    ax.set_ylabel('Forward / Backward Ratio', fontsize=11)
    ax.set_title('Kernel Asymmetry: Forward vs Backward Influence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, max(ratios)*1.2)

    fig.tight_layout()
    path = OUT / "kernel_asymmetry_ratio.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 7: 扇形展开率 — 不同距离处的侧向宽度
# ═══════════════════════════════════════════════════════════════

def fig7_fan_width():
    """扇形核在不同前向距离处的侧向有效宽度"""
    distances = np.linspace(0, HALF_LEN, 50)
    base_sigma = SIGMA_PERP
    widths = [base_sigma * (1 + 0.6 * d / max(SIGMA_ALONG, 1e-6)) for d in distances]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(distances, widths, '-', color=C_GREEN, linewidth=2.5)
    ax1.axhline(y=SIGMA_PERP, color=C_ORANGE, linestyle='--', alpha=0.5, label='Uniform kernel width')

    ax1.axvline(x=SIGMA_ALONG, color=C_RED, linestyle=':', alpha=0.5)
    ax1.annotate(f'sigma_along={SIGMA_ALONG}\nwidth={base_sigma*(1+0.6):.2f}cell ({(1+0.6):.1f}x)',
                 xy=(SIGMA_ALONG, base_sigma*(1+0.6)),
                 xytext=(SIGMA_ALONG+1, base_sigma*(1+0.6)+0.3),
                 fontsize=9, color=C_RED, fontweight='bold')

    ax1.set_xlabel('Forward Distance (cells)', fontsize=11)
    ax1.set_ylabel('Effective Lateral Width (sigma_perp, cells)', fontsize=11)
    ax1.set_title('Fan Expansion: Lateral Width vs Forward Distance', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    # 右侧小图：示意图
    ax2 = fig.add_axes([0.62, 0.55, 0.25, 0.3])
    K = build_kernel(0, fan=True)
    k_disp = (K / K.max() * 255).astype(np.uint8)
    k_color = plt.cm.viridis(k_disp)
    ax2.imshow(k_disp, cmap='viridis', origin='lower', interpolation='bilinear')
    ax2.plot([0, SIGMA_ALONG*2], [SIGMA_PERP*2, SIGMA_PERP*2*(1+0.6*2)], 'c-', linewidth=1.5)
    ax2.plot([0, SIGMA_ALONG*2], [-SIGMA_PERP*2, -SIGMA_PERP*2*(1+0.6*2)], 'c-', linewidth=1.5)
    ax2.axis('off')
    ax2.set_title('Fan shape\n(cyan lines)', fontsize=8, color=C_CYAN)

    fig.tight_layout()
    path = OUT / "kernel_fan_width.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════
# 运行所有核分析
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*60)
    print("  各向异性扇形核 — 多维度分析")
    print(f"  输出: {OUT}")
    print("="*60)
    fig1_3d_surface()
    fig2_polar_response()
    fig3_speed_aware()
    fig4_parameter_sweep()
    fig5_contour_footprint()
    fig6_asymmetry_ratio()
    fig7_fan_width()
    print("="*60)
    print(f"  完成! 共 {len(list(OUT.glob('kernel_*.png')))} 张图")
    print("="*60)
