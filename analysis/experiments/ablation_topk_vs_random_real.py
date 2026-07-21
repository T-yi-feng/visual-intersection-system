"""
归因引导 vs 随机消融 — 基于真实路网数据的对比

从项目运行结果的 ablation CSV/JSON 中提取归因引导消融曲线，
通过随机排列边际贡献模拟随机消融，绘制对比图。

用法:
  先运行项目采集数据:
    echo "1" | python run.py --site ziyou --ablation-enable --max-frames 300
    echo "1" | python run.py --site huangshanlu --ablation-enable --max-frames 300
  然后:
    python analysis/experiments/ablation_topk_vs_random_real.py
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
C_GRAY = '#999999'
C_GOLD = '#e2b96f'


def find_ablation_data(site_key: str) -> list[dict]:
    """返回所有 ablation JSON 数据"""
    results = []
    base = ROOT / "outputs" / site_key
    if not base.exists():
        return results
    for vedio_dir in base.iterdir():
        if not vedio_dir.is_dir():
            continue
        ev_dir = vedio_dir / "events"
        if not ev_dir.exists():
            continue
        for sub in ev_dir.iterdir():
            if sub.is_dir():
                json_path = sub / "ablation_results.json"
                if json_path.exists():
                    data = json.loads(json_path.read_text(encoding='utf-8'))
                    results.append({'site': site_key, 'event': sub.name, 'data': data})
    return results


def compute_curves(ablation_data: list[dict]):
    """
    从 ablation JSON 计算归因引导曲线和随机消融模拟曲线。

    对每个事件:
      - 归因引导: 直接从 JSON 的 conflict_field_before/after 读取
      - 随机模拟: 随机排列边际贡献 (level i 的 reduction = C_{i-1} - C_i)
                   多次随机排列取平均
    """
    attribution_curves = []  # 每个元素: [C0, C1, ..., Ck] 百分比
    random_curves = []       # 每个元素: 多次随机排列的平均值

    for ev in ablation_data:
        rows = ev['data']
        base_c = rows[0]['conflict_field_before']
        if base_c <= 0:
            continue

        # 归因引导曲线
        attr = [100.0]
        for row in rows:
            after = row['conflict_field_after']
            attr.append(after / base_c * 100)
        attribution_curves.append(attr)

        # 提取每级的边际降幅
        marginals = []
        prev = base_c
        for row in rows:
            after = row['conflict_field_after']
            marginals.append((prev - after) / base_c * 100)  # 百分比降幅
            prev = after

        # 随机排列模拟 (1000 次)
        n_levels = len(marginals)
        random_sims = []
        for _ in range(200):
            perm = np.random.permutation(marginals)
            cum = 100.0
            sim = [100.0]
            for m in perm:
                cum -= m
                sim.append(max(0, cum))
            random_sims.append(sim)

        # 取均值
        max_len = max(len(s) for s in random_sims)
        padded = [s + [s[-1]]*(max_len-len(s)) for s in random_sims]
        random_mean = np.mean(padded, axis=0).tolist()
        random_curves.append(random_mean)

    return attribution_curves, random_curves


def plot_comparison(all_data):
    """绘制归因引导 vs 随机消融对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 收集所有站点的曲线
    site_attr = {}
    site_random = {}
    for site, data_list in all_data.items():
        attr, rand = compute_curves(data_list)
        if attr:
            site_attr[site] = attr
            site_random[site] = rand

    # ── 图1: 各站点平均对比 ──
    ax1 = axes[0]
    for si, (site, attr_list) in enumerate(site_attr.items()):
        rand_list = site_random.get(site, [])

        if attr_list:
            max_len = max(len(a) for a in attr_list)
            padded = [a + [a[-1]]*(max_len-len(a)) for a in attr_list]
            mean_a = np.mean(padded, axis=0)
            std_a = np.std(padded, axis=0)
            ks = list(range(len(mean_a)))

            ax1.plot(ks, mean_a, '-', color=[C_RED, C_BLUE][si], linewidth=2.5,
                    label=f'{site} Attribution-guided')
            ax1.fill_between(ks, mean_a-std_a, mean_a+std_a, alpha=0.1,
                            color=[C_RED, C_BLUE][si])

        if rand_list:
            max_len = max(len(r) for r in rand_list)
            padded = [r + [r[-1]]*(max_len-len(r)) for r in rand_list]
            mean_r = np.mean(padded, axis=0)
            std_r = np.std(padded, axis=0)
            ks = list(range(len(mean_r)))

            ax1.plot(ks, mean_r, '--', color=[C_RED, C_BLUE][si], linewidth=1.5,
                    alpha=0.5, label=f'{site} Random (avg 200 trials)')
            ax1.fill_between(ks, mean_r-std_r, mean_r+std_r, alpha=0.05,
                            color=[C_RED, C_BLUE][si])

    ax1.axhline(y=50, color=C_GRAY, linestyle=':', alpha=0.3)
    ax1.set_xlabel('Ablation Level K', fontsize=11)
    ax1.set_ylabel('Conflict Field Remaining (%)', fontsize=11)
    ax1.set_title('Attribution-guided vs Random Ablation\n(Real data, averaged across events)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 110)

    # ── 图2: 归因引导的额外降幅 (差值) ──
    ax2 = axes[1]
    for si, (site, attr_list) in enumerate(site_attr.items()):
        rand_list = site_random.get(site, [])
        if not attr_list or not rand_list:
            continue
        max_len = max(max(len(a) for a in attr_list), max(len(r) for r in rand_list))
        p_a = [a + [a[-1]]*(max_len-len(a)) for a in attr_list]
        p_r = [r + [r[-1]]*(max_len-len(r)) for r in rand_list]
        mean_a = np.mean(p_a, axis=0)
        mean_r = np.mean(p_r, axis=0)
        diff = mean_r - mean_a  # 随机比归因多残留 = 归因额外降幅
        ks = list(range(len(diff)))
        ax2.bar(ks, diff, width=0.4, color=[C_RED, C_BLUE][si], alpha=0.6,
               label=f'{site}')
        ax2.plot(ks, diff, 'o-', color=[C_RED, C_BLUE][si], linewidth=1.5)

    ax2.axhline(y=0, color=C_GRAY, linestyle='-', alpha=0.4)
    ax2.set_xlabel('Ablation Level K', fontsize=11)
    ax2.set_ylabel('Extra Conflict Drop vs Random (%)', fontsize=11)
    ax2.set_title('Attribution-guided Advantage over Random\n(positive = attribution removes more conflict)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ── 图3: Top-3 的归因 vs 随机边际贡献排名 ──
    ax3 = axes[2]
    all_top3_attr = []
    all_top3_random = []
    for site, data_list in all_data.items():
        for ev in data_list:
            rows = ev['data']
            marginals = []
            prev = rows[0]['conflict_field_before']
            for row in rows:
                after = row['conflict_field_after']
                marginals.append((prev - after) / prev * 100 if prev > 0 else 0)
                prev = after

            # 归因引导的前3级（按重要顺序）
            top3_attr = marginals[:3]

            # 随机选3个
            all_random_top3 = []
            for _ in range(500):
                perm = np.random.permutation(marginals)
                all_random_top3.append(sum(perm[:3]))
            top3_random_mean = np.mean(all_random_top3)

            all_top3_attr.append(sum(top3_attr))
            all_top3_random.append(top3_random_mean)

    if all_top3_attr and all_top3_random:
        x = np.arange(len(all_top3_attr))
        w = 0.3
        ax3.bar(x - w/2, all_top3_attr, w, color=C_RED, alpha=0.7, label='Top-3 Attribution')
        ax3.bar(x + w/2, all_top3_random, w, color=C_GRAY, alpha=0.7, label='Random 3 vehicles avg')
        ax3.set_xlabel('Event Index', fontsize=11)
        ax3.set_ylabel('Conflict Reduction (%)', fontsize=11)
        ax3.set_title('Top-3 Vehicles: Attribution vs Random\n(per event)',
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.2, axis='y')
        # 标注均值
        ax3.axhline(y=np.mean(all_top3_attr), color=C_RED, linestyle='--', alpha=0.5, linewidth=1)
        ax3.axhline(y=np.mean(all_top3_random), color=C_GRAY, linestyle='--', alpha=0.5, linewidth=1)

    # ── 图4: 效率比 (归因引导 / 随机) ──
    ax4 = axes[3]
    if all_top3_attr and all_top3_random:
        ratios = [a/r if r > 0 else 1 for a, r in zip(all_top3_attr, all_top3_random)]
        ax4.bar(range(len(ratios)), ratios, color=[C_GREEN if r > 1 else C_GRAY for r in ratios],
                edgecolor='white', linewidth=0.5)
        ax4.axhline(y=1.0, color=C_GRAY, linestyle='--', alpha=0.5)
        ax4.axhline(y=np.mean(ratios), color=C_RED, linestyle='-', alpha=0.7, linewidth=1.5,
                   label=f'Mean ratio = {np.mean(ratios):.2f}x')
        ax4.set_xlabel('Event Index', fontsize=11)
        ax4.set_ylabel('Top-3 Reduction Ratio\n(Attribution / Random)', fontsize=11)
        ax4.set_title('Attribution Efficiency Ratio\n(>1 = attribution finds more impactful vehicles)',
                      fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Real-Data Comparison: Attribution-Guided vs Random Ablation',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = OUT / "ablation_topk_vs_random_real.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved {path.name}")

    # 打印关键数据
    if all_top3_attr and all_top3_random:
        print(f"\n  === 关键数据 ===")
        print(f"  Top-3 归因消融平均降幅: {np.mean(all_top3_attr):.1f}%")
        print(f"  Top-3 随机消融平均降幅: {np.mean(all_top3_random):.1f}%")
        print(f"  归因/随机 效率比: {np.mean(all_top3_attr)/np.mean(all_top3_random):.2f}x")
        print(f"  提升幅度: {(np.mean(all_top3_attr)-np.mean(all_top3_random))/np.mean(all_top3_random)*100:.1f}%")


def main():
    print("="*60)
    print("  归因引导 vs 随机消融 — 基于真实路网数据")
    print("="*60)

    all_data = {}
    for site in ['ziyou', 'huangshanlu']:
        data_list = find_ablation_data(site)
        if data_list:
            all_data[site] = data_list
            print(f"  {site}: {len(data_list)} 组消融数据")
        else:
            print(f"  {site}: 未找到数据（请先运行项目）")

    if not all_data:
        print("\n  错误: 没有找到消融数据。请先运行:")
        print("    echo \"1\" | python run.py --site ziyou --ablation-enable --max-frames 300")
        print("    echo \"1\" | python run.py --site huangshanlu --ablation-enable --max-frames 300")
        return

    plot_comparison(all_data)
    print("\n" + "="*60)
    print("  完成!")
    print("="*60)


if __name__ == '__main__':
    main()
