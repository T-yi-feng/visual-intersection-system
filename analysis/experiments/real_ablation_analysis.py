"""
真实消融数据分析 — 读取项目运行结果中的 ablation CSV 数据，生成对比图表

用法:
  1. 先运行项目（每个站点需要至少跑一次）:
     echo "1" | python run.py --site ziyou --ablation-enable --max-frames 300
     echo "1" | python run.py --site huangshanlu --ablation-enable --max-frames 300

  2. 然后运行本脚本:
     python analysis/experiments/real_ablation_analysis.py
"""

import sys, json, csv, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "experiment_results"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
rcParams['axes.unicode_minus'] = False

C_BLUE = '#5b9bd5'
C_ORANGE = '#d4956b'
C_GREEN = '#6baf6b'
C_RED = '#d55b5b'
C_PURPLE = '#9b8ebf'
C_GOLD = '#e2b96f'
C_GRAY = '#999999'


def find_ablation_dirs(site_key: str) -> list[tuple[str, Path]]:
    """找到站点下所有包含 ablation_results.csv 的目录"""
    results = []
    base = ROOT / "outputs" / site_key
    if not base.exists():
        return results

    for vedio_dir in base.iterdir():
        if not vedio_dir.is_dir():
            continue
        events_dir = vedio_dir / "events"
        if not events_dir.exists():
            continue
        for sub_dir in events_dir.iterdir():
            if sub_dir.is_dir():
                csv_path = sub_dir / "ablation_results.csv"
                if csv_path.exists():
                    results.append((sub_dir.name, csv_path))
    return sorted(results)


def read_ablation_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_decay_curve(csv_rows: list[dict]) -> tuple[list[float], list[float]]:
    """从 CSV 行中提取冲突场和 Phi 的衰减曲线"""
    if not csv_rows:
        return [100.0], [100.0]
    base_c = float(csv_rows[0].get('conflict_field_before', 1)) or 1.0
    base_p = float(csv_rows[0].get('scalar_phi_before', 1)) or 1.0
    c_curve = [100.0]
    p_curve = [100.0]
    for row in csv_rows:
        c_after = float(row.get('conflict_field_after', 0))
        p_after = float(row.get('scalar_phi_after', 0))
        c_curve.append(c_after / base_c * 100)
        p_curve.append(p_after / base_p * 100)
    return c_curve, p_curve


def get_removed_ids(csv_rows: list[dict]) -> list[str]:
    """从 CSV 中获取每级移除的车辆 ID"""
    ids = []
    for row in csv_rows:
        vid = row.get('removed_vehicle_id', '?')
        ids.append(vid)
    return ids


def main():
    print("="*60)
    print("  真实路网消融数据分析")
    print("="*60)

    # 收集所有站点的消融数据
    sites = ['ziyou', 'huangshanlu']
    all_data = {}

    for site in sites:
        entries = find_ablation_dirs(site)
        if not entries:
            print(f"  {site}: 未找到消融数据（请先运行项目）")
            continue

        curves_c = []
        curves_p = []
        names = []

        for name, csv_path in entries:
            rows = read_ablation_csv(csv_path)
            c_curve, p_curve = extract_decay_curve(rows)
            if len(c_curve) >= 2 and max(c_curve[:2]) > 0:
                curves_c.append(c_curve)
                curves_p.append(p_curve)
                names.append(name)

        if curves_c:
            all_data[site] = {
                'curves_c': curves_c, 'curves_p': curves_p, 'names': names,
                'count': len(curves_c),
            }
            print(f"  {site}: {len(curves_c)} 组消融数据")

    if not all_data:
        print("\n  没有找到消融数据。请先运行项目：")
        print("    echo \"1\" | python run.py --site ziyou --ablation-enable --max-frames 300")
        return

    # ═══════════════════════════════════════
    # 图 1: 各站点消融曲线
    # ═══════════════════════════════════════

    fig, axes = plt.subplots(1, len(all_data), figsize=(7*len(all_data), 5))
    if len(all_data) == 1:
        axes = [axes]

    for si, (site, data) in enumerate(all_data.items()):
        ax = axes[si]
        colors_curve = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_GOLD]

        # 绘制每条消融曲线
        for ci in range(min(len(data['curves_c']), 6)):
            c = data['curves_c'][ci]
            ks = list(range(len(c)))
            ax.plot(ks, c, 'o-', color=colors_curve[ci], linewidth=1.5,
                    markersize=3, alpha=0.6, label=f"Event {data['names'][ci][-6:]}")

        # 平均曲线
        if data['curves_c']:
            max_len = max(len(c) for c in data['curves_c'])
            padded = [c + [c[-1]] * (max_len - len(c)) for c in data['curves_c']]
            mean_c = np.mean(padded, axis=0)
            std_c = np.std(padded, axis=0)
            ks = list(range(len(mean_c)))
            ax.plot(ks, mean_c, 's-', color='black', linewidth=3, markersize=7, label='Mean ± 1σ')
            ax.fill_between(ks, mean_c-std_c, mean_c+std_c, alpha=0.1, color='black')

        ax.axhline(y=50, color=C_GRAY, linestyle='--', alpha=0.3)
        ax.set_xlabel('Ablation Level K', fontsize=11)
        ax.set_ylabel('Conflict Field Remaining (%)', fontsize=11)
        ax.set_title(f'{site}: {data["count"]} Events', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 110)

    fig.suptitle('Real-World Ablation: Conflict Field Decay via Top-K Removal',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = OUT / "real_ablation_curves.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  -> Saved {path.name}")

    # ═══════════════════════════════════════
    # 图 2: 冲突场 vs Phi 的消融对比
    # ═══════════════════════════════════════

    fig, ax = plt.subplots(figsize=(9, 5.5))

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    for si, (site, data) in enumerate(all_data.items()):
        for ci in range(min(len(data['curves_c']), 5)):
            c = data['curves_c'][ci]
            p = data['curves_p'][ci]
            ks = list(range(len(c)))
            ax.plot(ks, c, '-', color=[C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_GOLD][ci],
                    linewidth=1.5, alpha=0.5)
            # 用标记突出关键点
            ax.scatter(ks[:4], c[:4], marker=markers[ci % len(markers)],
                      color=[C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_GOLD][ci],
                      s=30, label=f'{site} Conflict (event {ci+1})', zorder=5)

            # Phi 衰减（虚线）
            if len(p) >= 4:
                ax.plot(ks[:4], p[:4], '--', color=[C_BLUE, C_GREEN, C_ORANGE, C_PURPLE, C_GOLD][ci],
                       linewidth=1, alpha=0.4)

    ax.set_xlabel('Ablation Level K', fontsize=11)
    ax.set_ylabel('Value Remaining (%)', fontsize=11)
    ax.set_title('Conflict Field vs Phi: Ablation Comparison Across Events', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    path = OUT / "real_ablation_conflict_vs_phi.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")

    # ═══════════════════════════════════════
    # 图 3: 各站点的平均消融柱状对比
    # ═══════════════════════════════════════

    fig, ax = plt.subplots(figsize=(9, 5))
    sites_list = list(all_data.keys())
    x = np.arange(len(sites_list))
    width = 0.2

    for ki, k_label in enumerate(['K=1', 'K=2', 'K=3']):
        k_vals = []
        for site in sites_list:
            data = all_data[site]
            vals_at_k = []
            for c in data['curves_c']:
                if len(c) > ki + 1:
                    vals_at_k.append(100 - c[ki + 1])
            k_vals.append(np.mean(vals_at_k) if vals_at_k else 0)

        ax.bar(x + ki * width - width, k_vals, width,
               label=f'{k_label}', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sites_list, fontsize=11)
    ax.set_ylabel('Conflict Field Reduction (%)', fontsize=11)
    ax.set_title('Average Conflict Reduction at K=1,2,3 by Site', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = OUT / "real_ablation_bar_summary.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")

    # ═══════════════════════════════════════
    # 图 4: 归因分数分布 — 每次移除的边际贡献
    # ═══════════════════════════════════════

    fig, ax = plt.subplots(figsize=(9, 5))
    all_marginals = {}

    for site, data in all_data.items():
        marginals = []
        for c in data['curves_c']:
            for ki in range(1, min(len(c), 6)):
                marginals.append(c[ki-1] - c[ki])
        if marginals:
            all_marginals[site] = marginals

    if all_marginals:
        sites_list = list(all_marginals.keys())
        bp = ax.boxplot([all_marginals[s] for s in sites_list], labels=sites_list,
                        patch_artist=True, widths=0.5)

        colors = [C_BLUE, C_ORANGE]
        for patch, color in zip(bp['boxes'], colors[:len(sites_list)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel('Marginal Conflict Drop per Removal (%)', fontsize=11)
        ax.set_title('Distribution of Per-Vehicle Conflict Contribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = OUT / "real_ablation_marginal_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")

    print("\n" + "="*60)
    print(f"  全部完成! 共 4 张图表生成到 {OUT}")
    print("="*60)


if __name__ == '__main__':
    main()
