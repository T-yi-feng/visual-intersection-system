"""
消融实验深度分析 — 多场景、多维度消融验证

运行: python analysis/experiments/ablation_study.py
输出: docs/experiment_results/ablation_*.png
"""

import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "experiment_results"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

from core.conflict import *
from analysis.root_cause import compute_root_cause, root_cause_to_pct

# 颜色
C_BLUE = '#5b9bd5'
C_ORANGE = '#d4956b'
C_GREEN = '#6baf6b'
C_RED = '#d55b5b'
C_PURPLE = '#9b8ebf'
C_GOLD = '#e2b96f'
C_GRAY = '#999999'


def build_scene(scene_type, n_vehicles=8):
    """构建指定场景的车辆列表"""
    np.random.seed(42)
    if scene_type == 'single_queue':
        return [{'cx': 15+i*3.5, 'cy': 30, 'speed_mps': max(0.0, 5-i*0.6),
                 'heading_deg': 90, 'label': 'car', 'track_id': i} for i in range(n_vehicles)]

    elif scene_type == 'two_queues':
        v = []
        for i in range(n_vehicles//2):
            v.append({'cx': 15+i*3.5, 'cy': 32, 'speed_mps': max(0.5, 4-i*0.5),
                      'heading_deg': 90, 'label': 'car', 'track_id': i})
        for i in range(n_vehicles//2):
            v.append({'cx': 15+i*3.5, 'cy': 28, 'speed_mps': max(0.5, 4-i*0.5),
                      'heading_deg': 90, 'label': 'car', 'track_id': 10+i})
        return v

    elif scene_type == 'cross':
        v = []
        for i in range(n_vehicles//2):
            v.append({'cx': 20+i*3, 'cy': 35-i*2, 'speed_mps': max(0.0, 4-i*0.5),
                      'heading_deg': 90, 'label': 'car', 'track_id': i})
        for i in range(n_vehicles//2):
            v.append({'cx': 25+i*2, 'cy': 20+i*2, 'speed_mps': max(0.0, 3-i*0.4),
                      'heading_deg': 0, 'label': 'car', 'track_id': 10+i})
        return v

    return []


def run_ablation(vehicles, grid_cfg, kernel_cfg):
    """对一组车辆执行消融实验，返回各级的冲突场剩余比"""
    O, V, T, m = scatter_vehicles_to_grid(vehicles, grid_cfg)
    layers = decompose_direction(T, m, DIRECTION_BINS)
    C_base, _, R = compute_conflict_field(layers, build_all_directional_kernels(DIRECTION_BINS, kernel_cfg), DEFAULT_CONFLICT_PAIRS)
    influences = compute_vehicle_influence(vehicles, R, grid_cfg, DEFAULT_CONFLICT_PAIRS)

    base_sum = C_base.sum() if C_base.sum() > 0 else 1.0
    ranked = sorted(enumerate(influences), key=lambda x: x[1], reverse=True)

    conflict_remaining = [100.0]
    for ki in range(len(ranked)):
        removed_indices = [ranked[j][0] for j in range(ki+1)]
        remaining = [v for j, v in enumerate(vehicles) if j not in removed_indices]
        if len(remaining) < 1:
            conflict_remaining.append(0.0)
            continue
        O2, V2, T2, m2 = scatter_vehicles_to_grid(remaining, grid_cfg)
        l2 = decompose_direction(T2, m2, DIRECTION_BINS)
        C2, _, _ = compute_conflict_field(l2, build_all_directional_kernels(DIRECTION_BINS, kernel_cfg), DEFAULT_CONFLICT_PAIRS)
        conflict_remaining.append(C2.sum() / base_sum * 100)

    return conflict_remaining, ranked, influences, C_base


# ═══════════════════════════════════════════════════════════════
# Fig 1: 三种场景的消融曲线对比
# ═══════════════════════════════════════════════════════════════

def fig1_scene_comparison():
    """单排、双排、交叉 三种场景的消融曲线"""
    print("Fig 1: 场景消融对比")
    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()

    scenes = {
        'Single Queue (car-following)': build_scene('single_queue', 8),
        'Two Parallel Queues': build_scene('two_queues', 8),
        'Crossing Traffic': build_scene('cross', 8),
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [C_BLUE, C_GREEN, C_ORANGE]

    for (name, vehicles), color in zip(scenes.items(), colors):
        conflict_remaining, ranked, _, _ = run_ablation(vehicles, grid_cfg, kernel_cfg)
        ks = list(range(len(conflict_remaining)))
        ax.plot(ks, conflict_remaining, 'o-', color=color, linewidth=2, markersize=5, label=name)
        # 标注前3个移除车辆
        for ri in range(min(3, len(ranked))):
            idx, val = ranked[ri]
            tid = vehicles[idx].get('track_id', idx)
            lbl = vehicles[idx].get('label', 'car')
            ax.annotate(f'K={ri+1}: #{tid}', xy=(ri+1, conflict_remaining[ri+1]),
                       fontsize=7, color=color, fontweight='bold')

    ax.axhline(y=50, color=C_GRAY, linestyle='--', alpha=0.3)
    ax.set_xlabel('Ablation Level K (number of vehicles removed)', fontsize=11)
    ax.set_ylabel('Conflict Field Remaining (%)', fontsize=11)
    ax.set_title('Ablation Comparison: Three Traffic Scenarios', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    path = OUT / "ablation_scene_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 2: Top-K vs Random Removal 对比
# ═══════════════════════════════════════════════════════════════

def fig2_topk_vs_random():
    """Top-K 归因移除 vs 随机移除的冲突场降幅对比"""
    print("Fig 2: Top-K vs Random")
    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()
    vehicles = build_scene('single_queue', 10)

    # Top-K 消融
    conflict_topk, ranked, inf, C_base = run_ablation(vehicles, grid_cfg, kernel_cfg)

    # Random 消融（10 次取平均）
    random_results = []
    for trial in range(20):
        np.random.seed(trial * 100)
        perm = np.random.permutation(len(vehicles))
        cr = [100.0]
        for ki in range(len(vehicles)):
            removed = set(perm[:ki+1])
            remaining = [v for j, v in enumerate(vehicles) if j in removed]
            if len(remaining) < 1:
                cr.append(0.0)
                continue
            O2, V2, T2, m2 = scatter_vehicles_to_grid(remaining, grid_cfg)
            l2 = decompose_direction(T2, m2, DIRECTION_BINS)
            C2, _, _ = compute_conflict_field(l2, build_all_directional_kernels(DIRECTION_BINS, kernel_cfg), DEFAULT_CONFLICT_PAIRS)
            cr.append(C2.sum() / C_base.sum() * 100 if C_base.sum() > 0 else 0)
        random_results.append(cr)

    random_mean = np.mean(random_results, axis=0)
    random_std = np.std(random_results, axis=0)
    ks = list(range(len(conflict_topk)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, conflict_topk, 'o-', color=C_RED, linewidth=2.5, markersize=6, label='Top-K Attribution Removal')
    ax.plot(ks, random_mean, 's--', color=C_GRAY, linewidth=2, markersize=5, label='Random Removal (avg of 20 trials)')
    ax.fill_between(ks, random_mean-random_std, random_mean+random_std, alpha=0.1, color=C_GRAY, label='Random ±1σ')

    # 阴影区域：Top-K 比 Random 多降低的冲突量
    ax.fill_between(ks, conflict_topk, random_mean, alpha=0.15, color=C_GREEN,
                    label=f'Extra reduction by attribution-guided removal')

    # 关键点标注
    for k in [1, 3, 5]:
        eff = random_mean[k] - conflict_topk[k]
        ax.annotate(f'K={k}: +{eff:.1f}% extra\ndrop vs random',
                    xy=(k, (conflict_topk[k]+random_mean[k])/2),
                    fontsize=8, color=C_GREEN, fontweight='bold',
                    ha='center')

    ax.set_xlabel('Removal Level K', fontsize=11)
    ax.set_ylabel('Conflict Field Remaining (%)', fontsize=11)
    ax.set_title('Attribution-Guided Ablation vs Random Removal\n(10-vehicle queue)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    path = OUT / "ablation_topk_vs_random.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path.name}")

    # 数值输出
    improvement = {k: random_mean[k] - conflict_topk[k] for k in range(1, 6)}
    with open(OUT / "ablation_topk_vs_random.json", 'w') as f:
        json.dump({'topk': [float(x) for x in conflict_topk],
                   'random_mean': [float(x) for x in random_mean],
                   'random_std': [float(x) for x in random_std],
                   'improvement': {k: float(v) for k, v in improvement.items()}}, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# Fig 3: 冲突对配置消融 — 不同冲突对组合的效果
# ═══════════════════════════════════════════════════════════════

def fig3_pair_ablation():
    """不同冲突对配置对冲突场的影响"""
    print("Fig 3: 冲突对配置消融")
    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)
    vehicles = build_scene('cross', 8)

    O, V, T, m = scatter_vehicles_to_grid(vehicles, grid_cfg)
    layers = decompose_direction(T, m, DIRECTION_BINS)

    pair_configs = {
        'Opposing only (6 pairs)': [(0,6),(1,7),(2,8),(3,9),(4,10),(5,11)],
        'Opposing + Orthogonal (12 pairs)': DEFAULT_CONFLICT_PAIRS[:12],
        'All 24 pairs (full)': DEFAULT_CONFLICT_PAIRS,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, pairs) in zip(axes, pair_configs.items()):
        C, pair_results, _ = compute_conflict_field(layers, kernels, pairs)
        C_disp = (C / max(C.max(), 1e-8) * 255).astype(np.uint8)
        ax.imshow(C_disp, cmap='inferno', origin='lower', interpolation='bilinear')
        ax.set_title(f'{name}\npeak={C.max():.4f}  pairs={len(pairs)}', fontsize=9, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Conflict Field Under Different Pair Configurations', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = OUT / "ablation_pair_configs.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 4: 队列长度 vs 消融效率
# ═══════════════════════════════════════════════════════════════

def fig4_queue_length_efficiency():
    """不同队列长度下，移除 Top-K 百分比车辆的冲突降幅"""
    print("Fig 4: 队列长度 vs 消融效率")
    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()

    lengths = [4, 6, 8, 10, 12, 15]
    k_percentages = [10, 20, 30]  # 移除百分之几的车辆

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for kp in k_percentages:
        drops = []
        for n in lengths:
            vehicles = build_scene('single_queue', n)
            cr, ranked, _, _ = run_ablation(vehicles, grid_cfg, kernel_cfg)
            k_val = max(1, int(n * kp / 100))
            if k_val < len(cr):
                drops.append(100 - cr[k_val])
            else:
                drops.append(100 - cr[-1])
        ax.plot(lengths, drops, 'o-', linewidth=2, markersize=6, label=f'Remove Top {kp}%')

    ax.set_xlabel('Queue Length (vehicles)', fontsize=11)
    ax.set_ylabel('Conflict Field Drop (%)', fontsize=11)
    ax.set_title('Ablation Efficiency: Queue Length vs Removal Impact', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = OUT / "ablation_queue_length.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 5: 冲突场热力图 — 逐级移除可视化
# ═══════════════════════════════════════════════════════════════

def fig5_heatmap_sequence():
    """逐级移除 Top-K 车辆的冲突场热力图序列"""
    print("Fig 5: 消融热力图序列")
    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)
    vehicles = build_scene('cross', 6)

    O, V, T, m = scatter_vehicles_to_grid(vehicles, grid_cfg)
    layers = decompose_direction(T, m, DIRECTION_BINS)
    C_base, _, R = compute_conflict_field(layers, kernels, DEFAULT_CONFLICT_PAIRS)
    influences = compute_vehicle_influence(vehicles, R, grid_cfg, DEFAULT_CONFLICT_PAIRS)

    ranked = sorted(enumerate(influences), key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    # Baseline
    C_disp = (C_base / max(C_base.max(), 1e-8) * 255).astype(np.uint8)
    axes[0].imshow(C_disp, cmap='inferno', origin='lower', interpolation='bilinear')
    axes[0].set_title(f'Baseline (all {len(vehicles)} vehicles)', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    remaining = list(vehicles)
    for ki in range(len(ranked)):
        idx, val = ranked[ki]
        tid = vehicles[idx].get('track_id', idx)
        remaining = [v for i, v in enumerate(remaining) if i != 0] if ki == 0 else [v for v in remaining if v not in [vehicles[ranked[j][0]] for j in range(ki+1)]]

        # 重新构建
        O2, V2, T2, m2 = scatter_vehicles_to_grid(remaining, grid_cfg)
        l2 = decompose_direction(T2, m2, DIRECTION_BINS)
        C2, _, _ = compute_conflict_field(l2, kernels, DEFAULT_CONFLICT_PAIRS)
        decay = C2.sum() / C_base.sum() * 100 if C_base.sum() > 0 else 0

        C_disp2 = (C2 / max(C2.max(), 1e-8) * 255).astype(np.uint8)
        axes[ki+1].imshow(C_disp2, cmap='inferno', origin='lower', interpolation='bilinear')
        axes[ki+1].set_title(f'K={ki+1}: remove #{tid}\n{decay:.0f}% remain', fontsize=8, fontweight='bold')
        axes[ki+1].axis('off')

        if ki >= 6:
            break

    for a in axes:
        a.axis('off')

    fig.suptitle('Conflict Field Decay Through Sequential Top-K Removal (Crossing Scenario)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = OUT / "ablation_heatmap_sequence.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path.name}")


# ═══════════════════════════════════════════════════════════════
# Fig 6: 冲突对贡献分解 — 各冲突类型的占比
# ═══════════════════════════════════════════════════════════════

def fig6_pair_contribution():
    """不同类型冲突对（对向/正交/同向）在总冲突场中的占比"""
    print("Fig 6: 冲突对贡献分解")
    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)

    # 混合场景
    vehicles = build_scene('cross', 6) + build_scene('single_queue', 4)
    O, V, T, m = scatter_vehicles_to_grid(vehicles, grid_cfg)
    layers = decompose_direction(T, m, DIRECTION_BINS)

    opposite_pairs = [(0,6),(1,7),(2,8),(3,9),(4,10),(5,11)]
    orthogonal_pairs = [(0,3),(1,4),(2,5),(3,6),(4,7),(5,8)]
    same_pairs = [(i,i) for i in range(12)]

    contributions = {}
    for name, pairs in [('Opposing (180°)', opposite_pairs),
                        ('Orthogonal (~90°)', orthogonal_pairs),
                        ('Same-direction (0°)', same_pairs)]:
        C, _, _ = compute_conflict_field(layers, kernels, pairs)
        contributions[name] = C.sum()

    total = sum(contributions.values()) or 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    labels = list(contributions.keys())
    sizes = [v/total*100 for v in contributions.values()]
    colors = [C_RED, C_ORANGE, C_BLUE]
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90, explode=(0.03, 0.03, 0.03))
    ax1.set_title('Conflict Contribution by Pair Type', fontsize=12, fontweight='bold')

    # 柱状图（绝对数值）
    bars = ax2.barh(list(contributions.keys()), [v/total*100 for v in contributions.values()],
                    color=colors, edgecolor='white', linewidth=0.5, height=0.5)
    for bar, val in zip(bars, sizes):
        ax2.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Contribution to Total Conflict (%)', fontsize=11)
    ax2.set_title('Absolute Contribution (normalized)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(sizes)*1.4+5)
    ax2.grid(True, alpha=0.2, axis='x')

    fig.tight_layout()
    path = OUT / "ablation_pair_contribution.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path.name}")


# ═══════════════════════════════════════════════════════════════
# 运行
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*60)
    print("  消融实验深度分析")
    print(f"  输出: {OUT}")
    print("="*60)
    fig1_scene_comparison()
    fig2_topk_vs_random()
    fig3_pair_ablation()
    fig4_queue_length_efficiency()
    fig5_heatmap_sequence()
    fig6_pair_contribution()
    print("="*60)
    print(f"  完成! 共 {len(list(OUT.glob('ablation_*.png')))} 张图")
    print("="*60)
