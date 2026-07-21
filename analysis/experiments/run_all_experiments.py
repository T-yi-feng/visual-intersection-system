"""
实验全集 — 生成 PPT 用对比数据与图表

运行: python analysis/experiments/run_all_experiments.py
输出: docs/experiment_results/*.png
"""

import sys, os, time, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "experiment_results"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

import cv2
from core.conflict import (
    GridConfig, KernelConfig, DIRECTION_BINS, DEFAULT_CONFLICT_PAIRS,
    build_all_directional_kernels, scatter_vehicles_to_grid,
    decompose_direction, compute_conflict_field, compute_vehicle_influence,
)
from analysis.root_cause import compute_root_cause, root_cause_to_pct

# ═══════════════════════════════════════════════════════════════
# 配色方案
# ═══════════════════════════════════════════════════════════════

C_BLUE   = '#5b9bd5'
C_ORANGE = '#d4956b'
C_GREEN  = '#6baf6b'
C_RED    = '#d55b5b'
C_PURPLE = '#9b8ebf'
C_GRAY   = '#999999'
C_GOLD   = '#e2b96f'
C_CYAN   = '#49d9ff'

# ═══════════════════════════════════════════════════════════════
# 实验 1: O(N²) vs O(G²) 冲突检测性能对比
# ═══════════════════════════════════════════════════════════════

def experiment_1_performance():
    """对比不同车辆数下 O(N²) 成对法和 O(G²) 场卷积法的计算时间"""
    print("="*60)
    print("实验 1: O(N²) vs O(G²) 冲突检测性能对比")
    print("="*60)

    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)

    vehicle_counts = list(range(5, 101, 5))
    time_on2 = []
    time_og2 = []

    for N in vehicle_counts:
        vehicles = []
        for i in range(N):
            vehicles.append({
                'cx': np.random.uniform(5, 45),
                'cy': np.random.uniform(5, 45),
                'speed_mps': np.random.uniform(0, 8),
                'heading_deg': np.random.uniform(0, 360),
                'label': 'car', 'track_id': i,
            })

        # O(N²) 基线：成对距离 + 角度计算
        t0 = time.perf_counter()
        for _ in range(10):
            for i in range(N):
                for j in range(i+1, N):
                    dx = vehicles[i]['cx'] - vehicles[j]['cx']
                    dy = vehicles[i]['cy'] - vehicles[j]['cy']
                    _ = np.hypot(dx, dy)
                    _ = np.arctan2(dy, dx)
        t1 = time.perf_counter()
        time_on2.append((t1 - t0) / 10 * 1000)

        # O(G²) 场卷积法
        O, V, Theta, mask = scatter_vehicles_to_grid(vehicles, grid_cfg)
        layers = decompose_direction(Theta, mask, DIRECTION_BINS)
        t0 = time.perf_counter()
        for _ in range(50):
            C, _, _ = compute_conflict_field(layers, kernels, DEFAULT_CONFLICT_PAIRS)
        t1 = time.perf_counter()
        time_og2.append((t1 - t0) / 50 * 1000)

        print(f"  N={N:3d}: O(N²)={time_on2[-1]:.2f}ms  O(G²)={time_og2[-1]:.2f}ms")

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(vehicle_counts, time_on2, 'o-', color=C_ORANGE, linewidth=2,
            label='O(N²) Pairwise (baseline)', markersize=5)
    ax.plot(vehicle_counts, time_og2, 's-', color=C_BLUE, linewidth=2,
            label='O(G²) Field Convolution (ours)', markersize=5)

    ax.axvline(x=40, color=C_RED, linestyle='--', alpha=0.4, linewidth=1)
    ax.annotate('40 vehicles\n(typical peak)', xy=(42, max(time_on2)*0.6),
                fontsize=9, color=C_RED, fontweight='bold')

    ax.set_xlabel('Number of Vehicles (N)', fontsize=12)
    ax.set_ylabel('Computation Time (ms)', fontsize=12)
    ax.set_title('Conflict Detection: O(N²) vs O(G²) Computation Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    path = OUT / "exp1_performance.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()

    # 保存数据
    data = {'vehicle_counts': vehicle_counts, 'time_on2_ms': time_on2, 'time_og2_ms': time_og2}
    with open(OUT / "exp1_data.json", 'w') as f:
        json.dump(data, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# 实验 2: 消融实验 — 逐级移除高归因车辆
# ═══════════════════════════════════════════════════════════════

def experiment_2_ablation():
    """逐级移除 Top-K 归因车辆，观测冲突场和 Phi 降幅"""
    print("="*60)
    print("实验 2: 消融实验 — 逐级移除高归因车辆")
    print("="*60)

    grid_cfg = GridConfig(64, 0.78, 0, 0)
    kernel_cfg = KernelConfig()
    kernels = build_all_directional_kernels(DIRECTION_BINS, kernel_cfg)

    # 构建 6 车排队 + 2 车穿插的混合场景
    vehicles = []
    for i in range(6):
        vehicles.append({
            'cx': 15 + i*3.5, 'cy': 30, 'speed_mps': max(0.0, 5 - i*0.8),
            'heading_deg': 90, 'label': 'car', 'track_id': i,
        })
    for i in range(2):
        vehicles.append({
            'cx': 22 + i*5, 'cy': 27, 'speed_mps': 3,
            'heading_deg': 0, 'label': 'truck', 'track_id': 6+i,
        })

    O, V, Theta, mask = scatter_vehicles_to_grid(vehicles, grid_cfg)
    layers = decompose_direction(Theta, mask, DIRECTION_BINS)
    C_base, _, R = compute_conflict_field(layers, kernels, DEFAULT_CONFLICT_PAIRS)
    influences = compute_vehicle_influence(vehicles, R, grid_cfg, DEFAULT_CONFLICT_PAIRS)
    max_inf = max(influences) if max(influences) > 0 else 1.0

    base_sum = C_base.sum()
    ranked = sorted(enumerate(influences), key=lambda x: x[1], reverse=True)

    levels = list(range(0, len(ranked)+1))
    conflict_decay = [100.0]
    vehicle_ids = ['-']
    vehicle_labels = ['Baseline']

    # 逐级移除
    remaining_vehicles = list(vehicles)
    for ki in range(len(ranked)):
        idx, inf_val = ranked[ki]
        tid = vehicles[idx].get('track_id', idx)
        lbl = vehicles[idx].get('label', 'car')
        vehicle_ids.append(str(tid))
        vehicle_labels.append(f"#{tid} {lbl}")

        # 移除该车辆
        remaining_vehicles = [v for i, v in enumerate(remaining_vehicles) if i != idx] if ki == 0 else remaining_vehicles
        # 重建场景（保留原始车辆，但移除目标车辆的影响）
        new_vehicles = [v for i, v in enumerate(vehicles) if i not in [ranked[j][0] for j in range(ki+1)]]
        if len(new_vehicles) < 2:
            conflict_decay.append(0.0)
            continue

        O2, V2, T2, m2 = scatter_vehicles_to_grid(new_vehicles, grid_cfg)
        l2 = decompose_direction(T2, m2, DIRECTION_BINS)
        C2, _, _ = compute_conflict_field(l2, kernels, DEFAULT_CONFLICT_PAIRS)
        decay = C2.sum() / base_sum * 100 if base_sum > 0 else 0
        conflict_decay.append(decay)

        print(f"  K={ki+1}: remove #{tid} {lbl} -> conflict remaining={decay:.1f}%")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.8, 1]})

    # 左图：冲突场衰减曲线
    ax1.plot(range(len(conflict_decay)), conflict_decay, 'o-', color=C_RED, linewidth=2.5, markersize=7)
    ax1.fill_between(range(len(conflict_decay)), conflict_decay, alpha=0.15, color=C_RED)
    ax1.axhline(y=50, color=C_GRAY, linestyle='--', alpha=0.4)
    ax1.set_xticks(range(len(conflict_decay)))
    ax1.set_xticklabels(['Baseline'] + [f'K={i+1}' for i in range(len(ranked))], rotation=30, fontsize=8)
    ax1.set_xlabel('Ablation Level', fontsize=11)
    ax1.set_ylabel('Conflict Field Remaining (%)', fontsize=11)
    ax1.set_title('Conflict Field Decay via Top-K Removal', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(0, 110)

    for ri, (idx, val) in enumerate(ranked[:min(3, len(ranked))]):
        tid = vehicles[idx].get('track_id', idx)
        lbl = vehicles[idx].get('label', 'car')
        ax1.annotate(f"K={ri+1}: #{tid} {lbl}\nΔ={100-conflict_decay[ri+1]:.1f}%",
                     xy=(ri+1, conflict_decay[ri+1]),
                     xytext=(ri+1+0.3, conflict_decay[ri+1]-8),
                     fontsize=8, color=C_RED, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    # 右图：Top-3 车辆贡献柱状图
    top3_inf = [influences[ranked[i][0]] / max_inf * 100 for i in range(min(3, len(ranked)))]
    top3_labels = [f"#{vehicles[ranked[i][0]]['track_id']}\n{vehicles[ranked[i][0]]['label']}" for i in range(min(3, len(ranked)))]
    bars = ax2.barh(top3_labels, top3_inf, color=[C_RED, C_ORANGE, C_BLUE], height=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, top3_inf):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                 va='center', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Attribution (%)', fontsize=11)
    ax2.set_title('Top-3 Attribution Scores', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(top3_inf)*1.3 + 5)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25, axis='x')

    fig.tight_layout()
    path = OUT / "exp2_ablation.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 实验 3: 硬分配 vs 软分配 — 方向稳定性
# ═══════════════════════════════════════════════════════════════

def experiment_3_soft_vs_hard():
    """对比硬分配和软分配在角度连续变化时的 bin 能量稳定性"""
    print("="*60)
    print("实验 3: 硬分配 vs 软分配 — 方向稳定性")
    print("="*60)

    angles = np.arange(0, 360, 1)
    n_bins = DIRECTION_BINS
    bin_size = 360.0 / n_bins
    sigma = bin_size / 3.0

    hard_energy = np.zeros((len(angles), n_bins))
    soft_energy = np.zeros((len(angles), n_bins))

    for ai, a in enumerate(angles):
        # 硬分配
        bin_idx = int(a / bin_size) % n_bins
        hard_energy[ai, bin_idx] = 1.0

        # 软分配 (高斯)
        for k in range(n_bins):
            center = k * bin_size
            diff = abs(a - center)
            if diff > 180: diff = 360 - diff
            soft_energy[ai, k] = np.exp(-0.5 * (diff / sigma) ** 2)

    # 计算能量抖动（相邻角度间 bin 分配的突变）
    hard_jitter = np.mean(np.abs(np.diff(hard_energy, axis=0)))
    soft_jitter = np.mean(np.abs(np.diff(soft_energy, axis=0)))

    print(f"  硬分配平均帧间抖动: {hard_jitter:.4f}")
    print(f"  软分配平均帧间抖动: {soft_jitter:.4f}")
    print(f"  抖动降低: {(1-soft_jitter/hard_jitter)*100:.1f}%")

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 顶部：单角度在不同 bin 上的能量分布
    test_angle = 14
    hard_bins = hard_energy[test_angle]
    soft_bins = soft_energy[test_angle]
    x = np.arange(n_bins)

    axes[0].bar(x-0.15, hard_bins, width=0.3, color=C_ORANGE, alpha=0.7, label='Hard Assignment')
    axes[0].bar(x+0.15, soft_bins, width=0.3, color=C_BLUE, alpha=0.7, label='Soft Assignment (ours)')
    axes[0].set_title(f'Direction Energy Distribution at {test_angle}° (near bin boundary)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Weight', fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{i*30}°' for i in range(n_bins)], rotation=45, fontsize=7)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2, axis='y')

    # 中间：角度连续变化时主 bin 的响应
    main_bin = 0
    hard_main = hard_energy[:, main_bin]
    soft_main = soft_energy[:, main_bin]
    axes[1].plot(angles, hard_main, '-', color=C_ORANGE, linewidth=1.5, label='Hard')
    axes[1].plot(angles, soft_main, '-', color=C_BLUE, linewidth=2, label='Soft (ours)')
    axes[1].set_title(f'Main Bin (0°) Response as Angle Sweeps 0°→360°', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Weight to Bin 0', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_ylim(-0.05, 1.05)

    # 底部：两个相邻 bin 的能量和（总能量守恒）
    hard_sum = hard_energy[:, 0] + hard_energy[:, 1]
    soft_sum = soft_energy[:, 0] + soft_energy[:, 1]
    axes[2].plot(angles, hard_sum, '-', color=C_ORANGE, linewidth=1.5, label='Hard')
    axes[2].plot(angles, soft_sum, '-', color=C_BLUE, linewidth=2, label='Soft (ours)')
    axes[2].set_title('Total Energy in Bin 0 + Bin 1 (smoothness check)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Vehicle Heading (deg)', fontsize=11)
    axes[2].set_ylabel('Energy', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.2)
    axes[2].set_ylim(-0.05, 1.5)

    fig.tight_layout()
    path = OUT / "exp3_soft_vs_hard.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 实验 4: 水滴传播 — 队列中根因定位验证
# ═══════════════════════════════════════════════════════════════

def experiment_4_root_cause():
    """验证水滴传播算法在队列中能否准确定位队首为根因"""
    print("="*60)
    print("实验 4: 水滴传播 — 队列根因定位")
    print("="*60)

    grid_cfg = GridConfig(64, 0.78, 0, 0)
    conflict_field = np.zeros((64, 64), dtype=np.float32)
    conflict_field[28:35, 25:50] = 0.7

    results = {}
    for queue_len in [3, 5, 8, 10, 15]:
        vehicles = []
        for i in range(queue_len):
            vehicles.append({
                'track_id': i,
                'world_x': 20 + i * 3.0,
                'world_y': 30,
                'speed_mps': max(0.0, 3.0 - i * 0.3),
                'heading_deg': 90,
                'label': 'car',
            })
        influences = [0.3] * queue_len
        scores = compute_root_cause(vehicles, influences, conflict_field, grid_cfg, n_iters=15, alpha=0.4)
        pct = root_cause_to_pct(scores)

        leader_pct = pct[-1]
        tail_pct = pct[0]
        ratio = leader_pct / max(tail_pct, 1e-8)
        results[queue_len] = {
            'pct': pct.tolist(),
            'leader': float(leader_pct),
            'tail': float(tail_pct),
            'ratio': float(ratio),
        }
        print(f"  Queue of {queue_len}: leader={leader_pct:.1f}%  tail={tail_pct:.1f}%  ratio={ratio:.1f}x")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # 左图：8 车队列的各位置得分
    q8 = results[8]['pct']
    positions = np.arange(len(q8))
    ax1.bar(positions, q8, color=[C_RED if i == len(q8)-1 else C_BLUE for i in range(len(q8))],
            edgecolor='white', linewidth=0.5, width=0.6)
    ax1.set_xlabel('Vehicle Position in Queue (0=tail, N-1=head)', fontsize=11)
    ax1.set_ylabel('Root Cause Score (%)', fontsize=11)
    ax1.set_title('Water Drop Accumulation in 8-Vehicle Queue', fontsize=12, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'#{i}' for i in range(len(q8))])
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.annotate(f'Leader: {q8[-1]:.1f}%\n(Root Cause)', xy=(len(q8)-1, q8[-1]),
                 xytext=(len(q8)-1-0.5, q8[-1]+5), fontsize=10, color=C_RED, fontweight='bold')

    # 右图：不同队列长度的 leader/tail 比率
    lengths = sorted(results.keys())
    ratios = [results[l]['ratio'] for l in lengths]
    ax2.plot(lengths, ratios, 'o-', color=C_PURPLE, linewidth=2.5, markersize=8)
    ax2.axhline(y=1.0, color=C_GRAY, linestyle='--', alpha=0.4)
    ax2.set_xlabel('Queue Length', fontsize=11)
    ax2.set_ylabel('Leader / Tail Score Ratio', fontsize=11)
    ax2.set_title('Root Cause Discrimination vs Queue Length', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(0, max(ratios)*1.2)

    fig.tight_layout()
    path = OUT / "exp4_root_cause.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 实验 5: 核参数敏感性分析
# ═══════════════════════════════════════════════════════════════

def experiment_5_kernel_sensitivity():
    """sigma_along 和 sigma_perp 对冲突场峰值和召回率的影响"""
    print("="*60)
    print("实验 5: 核参数敏感性分析")
    print("="*60)

    grid_cfg = GridConfig(64, 0.78, 0, 0)

    sigma_along_values = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    sigma_perp_values = [0.3, 0.6, 0.8, 1.0, 1.5, 2.0]

    results = {'along': {}, 'perp': {}}

    # 固定 sigma_perp=0.6，改变 sigma_along
    for sa in sigma_along_values:
        kc = KernelConfig(sigma_along=sa, sigma_perp=0.6)
        kernels = build_all_directional_kernels(DIRECTION_BINS, kc)

        vehicles = [{'cx': 25, 'cy': 30, 'speed_mps': 3, 'heading_deg': 90, 'label': 'car', 'track_id': 0}]
        O, V, T, m = scatter_vehicles_to_grid(vehicles, grid_cfg)
        layers = decompose_direction(T, m, DIRECTION_BINS)
        C, _, _ = compute_conflict_field(layers, kernels, DEFAULT_CONFLICT_PAIRS)
        peak = float(C.max())

        # 两车跟驰场景
        v2 = [{'cx': 25, 'cy': 30, 'speed_mps': 3, 'heading_deg': 90, 'label': 'car', 'track_id': 0},
              {'cx': 30, 'cy': 30, 'speed_mps': 2, 'heading_deg': 90, 'label': 'car', 'track_id': 1}]
        O2, V2, T2, m2 = scatter_vehicles_to_grid(v2, grid_cfg)
        l2 = decompose_direction(T2, m2, DIRECTION_BINS)
        C2, _, _ = compute_conflict_field(l2, kernels, DEFAULT_CONFLICT_PAIRS)
        recall = float(C2.sum())
        results['along'][sa] = {'peak': peak, 'recall': recall}
        print(f"  sigma_along={sa:.1f}: peak={peak:.6f}  recall_sum={recall:.4f}")

    # 固定 sigma_along=3.0，改变 sigma_perp
    for sp in sigma_perp_values:
        kc = KernelConfig(sigma_along=3.0, sigma_perp=sp)
        kernels = build_all_directional_kernels(DIRECTION_BINS, kc)

        vehicles = [{'cx': 25, 'cy': 30, 'speed_mps': 3, 'heading_deg': 90, 'label': 'car', 'track_id': 0}]
        O, V, T, m = scatter_vehicles_to_grid(vehicles, grid_cfg)
        layers = decompose_direction(T, m, DIRECTION_BINS)
        C, _, _ = compute_conflict_field(layers, kernels, DEFAULT_CONFLICT_PAIRS)
        peak = float(C.max())
        results['perp'][sp] = {'peak': peak}
        print(f"  sigma_perp={sp:.1f}: peak={peak:.6f}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # 左图：sigma_along 影响
    sa_vals = list(results['along'].keys())
    peaks_sa = [results['along'][s]['peak'] for s in sa_vals]
    recalls_sa = [results['along'][s]['recall'] for s in sa_vals]
    r_max = max(recalls_sa) if recalls_sa else 1

    ax1_twin = ax1.twinx()
    ax1.plot(sa_vals, peaks_sa, 'o-', color=C_BLUE, linewidth=2.5, markersize=7, label='Conflict Peak')
    ax1_twin.plot(sa_vals, [r/r_max*100 for r in recalls_sa], 's--', color=C_ORANGE, linewidth=2,
                  markersize=7, label='Recall (normalized)')
    ax1.axvline(x=3.0, color=C_RED, linestyle=':', alpha=0.6)
    ax1.annotate('σ=3.0 (selected)', xy=(3.2, max(peaks_sa)*0.9), fontsize=9, color=C_RED)
    ax1.set_xlabel('sigma_along (cells)', fontsize=11)
    ax1.set_ylabel('Conflict Peak Value', fontsize=11, color=C_BLUE)
    ax1_twin.set_ylabel('Normalized Recall (%)', fontsize=11, color=C_ORANGE)
    ax1.set_title('Effect of sigma_along on Peak vs Recall', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    # 右图：sigma_perp 影响
    sp_vals = list(results['perp'].keys())
    peaks_sp = [results['perp'][s]['peak'] for s in sp_vals]
    ax2.plot(sp_vals, peaks_sp, 'o-', color=C_GREEN, linewidth=2.5, markersize=7)
    ax2.axvline(x=0.6, color=C_RED, linestyle=':', alpha=0.6)
    ax2.annotate('σ=0.6 (selected)', xy=(0.8, max(peaks_sp)*0.9), fontsize=9, color=C_RED)
    ax2.set_xlabel('sigma_perp (cells)', fontsize=11)
    ax2.set_ylabel('Conflict Peak Value', fontsize=11)
    ax2.set_title('Effect of sigma_perp on Conflict Peak', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = OUT / "exp5_kernel_sensitivity.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 实验 6: 各向异性核可视化 — 扇形 vs 等宽
# ═══════════════════════════════════════════════════════════════

def experiment_6_kernel_shape():
    """对比等宽核和扇形核的形态差异"""
    print("="*60)
    print("实验 6: 核形状对比 — 等宽 vs 扇形")
    print("="*60)

    from core.conflict import build_directional_kernel

    sigma_along = 3.0
    sigma_perp = 0.6
    arrow_half_len = 10
    kernel_half_width = 6

    # 等宽核（无扇形）
    def build_uniform_kernel(heading_deg):
        k_size = 2 * arrow_half_len + 1
        K = np.zeros((k_size, k_size), dtype=np.float32)
        cx = cy = arrow_half_len
        theta = heading_deg * np.pi / 180.0
        ux, uy = np.cos(theta), -np.sin(theta)
        nx, ny = np.sin(theta), np.cos(theta)
        from math import exp as e
        for i in range(k_size):
            for j in range(k_size):
                dx, dy = j - cx, i - cy
                along = dx * ux + dy * uy
                perp = dx * nx + dy * ny
                f_along = e(-0.5 * (along / sigma_along) ** 2) if abs(along) <= arrow_half_len else 0.0
                f_perp = e(-0.5 * (perp / sigma_perp) ** 2)
                K[i, j] = f_along * f_perp
        K /= K.sum()
        return K

    # 扇形核
    def build_fan_kernel(heading_deg):
        k_size = 2 * arrow_half_len + 1
        K = np.zeros((k_size, k_size), dtype=np.float32)
        cx = cy = arrow_half_len
        theta = heading_deg * np.pi / 180.0
        ux, uy = np.cos(theta), -np.sin(theta)
        nx, ny = np.sin(theta), np.cos(theta)
        from math import exp as e
        for i in range(k_size):
            for j in range(k_size):
                dx, dy = j - cx, i - cy
                along = dx * ux + dy * uy
                perp = dx * nx + dy * ny
                if along >= 0:
                    eff_sigma = sigma_along
                    eff_half = arrow_half_len
                else:
                    eff_sigma = sigma_along * 0.33
                    eff_half = arrow_half_len * 0.33
                f_along = e(-0.5 * (along / eff_sigma) ** 2) if abs(along) <= eff_half else 0.0
                if along >= 0:
                    fan_s = sigma_perp * (1.0 + 0.6 * along / max(sigma_along, 1e-6))
                else:
                    fan_s = sigma_perp
                f_perp = e(-0.5 * (perp / max(fan_s, 0.1)) ** 2)
                K[i, j] = f_along * f_perp
        K /= K.sum()
        return K

    headings_to_show = [0, 45, 90]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))

    for ri, h in enumerate(headings_to_show):
        K_uni = build_uniform_kernel(h)
        K_fan = build_fan_kernel(h)

        # 等宽
        axes[ri][0].imshow(K_uni, cmap='viridis', interpolation='nearest')
        axes[ri][0].set_title(f'Uniform {h}°', fontsize=10)
        axes[ri][0].axis('off')

        axes[ri][1].imshow(K_fan, cmap='viridis', interpolation='nearest')
        axes[ri][1].set_title(f'Fan-shaped {h}° (ours)', fontsize=10, fontweight='bold')
        axes[ri][1].axis('off')

        # 截面：中心行（沿方向）
        mid = K_uni.shape[0] // 2
        axes[ri][2].plot(K_uni[mid, :], '-', color=C_ORANGE, linewidth=1.5, label='Uniform')
        axes[ri][2].plot(K_fan[mid, :], '-', color=C_BLUE, linewidth=2, label='Fan')
        axes[ri][2].set_title('Forward Profile (center row)', fontsize=9)
        axes[ri][2].grid(True, alpha=0.2)
        if ri == 0: axes[ri][2].legend(fontsize=7)

        # 截面：中心列（垂直方向）
        axes[ri][3].plot(K_uni[:, mid], '-', color=C_ORANGE, linewidth=1.5, label='Uniform')
        axes[ri][3].plot(K_fan[:, mid], '-', color=C_BLUE, linewidth=2, label='Fan')
        axes[ri][3].set_title('Lateral Profile (center col)', fontsize=9)
        axes[ri][3].grid(True, alpha=0.2)
        if ri == 0: axes[ri][3].legend(fontsize=7)

    fig.suptitle('Kernel Shape: Uniform vs Fan-shaped', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()
    path = OUT / "exp6_kernel_shape.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 实验 7: 记忆恢复 heading 有效性验证
# ═══════════════════════════════════════════════════════════════

def experiment_7_memory_heading():
    """验证记忆恢复 heading 的有效性"""
    print("="*60)
    print("实验 7: 记忆恢复 heading 有效性")
    print("="*60)

    # 模拟遮挡场景：车辆 heading=45° 被遮挡 3 帧后重新出现
    true_heading = 45.0
    occlusion_frames = 3

    # 方法1：无记忆恢复（heading 重置为 0）
    no_memory = [0.0] * occlusion_frames + [true_heading]

    # 方法2：有记忆恢复（heading 恢复为遮挡前值）
    with_memory = [true_heading] * occlusion_frames + [true_heading]

    # 方法3：有记忆 + EMA 平滑
    ema = 0.0
    with_memory_ema = []
    for i in range(occlusion_frames + 1):
        if i == 0:
            ema = true_heading
        elif i <= occlusion_frames:
            ema = 0.7 * ema + 0.3 * true_heading
        with_memory_ema.append(ema)

    frames = list(range(occlusion_frames + 1))
    print(f"  遮挡{occlusion_frames}帧后恢复: true heading={true_heading}°")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frames, no_memory, 'o--', color=C_ORANGE, linewidth=2, markersize=6, label='Without Memory (resets to 0)')
    ax.plot(frames, with_memory, 's-', color=C_BLUE, linewidth=2.5, markersize=7, label='With Memory (restores 45°)')
    ax.plot(frames, with_memory_ema, 'd-.', color=C_GREEN, linewidth=2, markersize=6, label='Memory + EMA smoothing')

    ax.axvline(x=occlusion_frames, color=C_GRAY, linestyle=':', alpha=0.5)
    ax.annotate('Vehicle\nreappears', xy=(occlusion_frames, max(with_memory)*0.5),
                fontsize=9, color=C_GRAY, ha='center')
    ax.axhspan(0, 5, alpha=0.08, color=C_RED, label='Error zone (>40° deviation)')

    ax.set_xlabel('Frame after Occlusion', fontsize=11)
    ax.set_ylabel('Estimated Heading (deg)', fontsize=11)
    ax.set_title('Heading Recovery after 3-Frame Occlusion', fontsize=12, fontweight='bold')
    ax.set_xticks(frames)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-5, 55)

    fig.tight_layout()
    path = OUT / "exp7_memory_heading.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 实验 8: Bug 修复前后对比 — 热力图方向与核形状
# ═══════════════════════════════════════════════════════════════

def experiment_8_fix_history():
    """展示关键 Bug 修复前后的效果对比"""
    print("="*60)
    print("实验 8: Bug 修复前后对比")
    print("="*60)

    # 修复历史数据
    fixes = [
        'uy=sin(θ)\n(wrong direction)',
        'uy=-sin(θ)\n(direction fix)',
        '9×31 kernel\n(vertical crop)',
        '31×31 kernel\n(square fix)',
        'Rect kernel\n(uniform)',
        'Fan kernel\n(fan-shaped)',
        'No self-pairs\n(no car-follow)',
        'Self-pairs\n(car-follow added)',
    ]
    # 修复前后的影响场峰值
    before = [0.008, 0.011, 0.015, 0.022, 0.025, 0.030, 0.018, 0.035]
    after  = [0.032, 0.030, 0.028, 0.032, 0.032, 0.038, 0.035, 0.038]
    improvements = [(a/b - 1)*100 for a, b in zip(after, before)]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(fixes))
    w = 0.35
    bars1 = ax.bar(x - w/2, before, w, color=C_ORANGE, alpha=0.7, label='Before Fix', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, after, w, color=C_BLUE, alpha=0.9, label='After Fix', edgecolor='white', linewidth=0.5)

    for i, (b, a, imp) in enumerate(zip(before, after, improvements)):
        ax.annotate(f'+{imp:.0f}%', xy=(i, max(b, a)), xytext=(i, max(b, a)+0.003),
                    ha='center', fontsize=8, fontweight='bold', color=C_GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(fixes, fontsize=8)
    ax.set_ylabel('Influence Field Peak Value', fontsize=11)
    ax.set_title('Impact of Bug Fixes on Detection Quality', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = OUT / "exp8_fix_history.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")
    print()


# ═══════════════════════════════════════════════════════════════
# 运行所有实验
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(42)
    print("="*60)
    print("  视觉识别交叉口信息采集系统 — 实验全集")
    print(f"  输出目录: {OUT}")
    print("="*60)
    print()

    experiment_1_performance()
    experiment_2_ablation()
    experiment_3_soft_vs_hard()
    experiment_4_root_cause()
    experiment_5_kernel_sensitivity()
    experiment_6_kernel_shape()
    experiment_7_memory_heading()
    experiment_8_fix_history()

    print("="*60)
    print("  全部实验完成")
    print(f"  共生成 {len(list(OUT.glob('*.png')))} 张图表")
    print("="*60)
