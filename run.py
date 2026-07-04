"""
Visual System - 交叉口监控视角下的快速鸟瞰映射和拥堵分析计算系统

主入口文件。支持三种运行模式：

1. 交互模式：python run.py
   → 列出所有路口站点 → 选择 → 列出视频 → 选择 → 运行

2. 站点模式：python run.py --site default
   → 指定站点 → 列出视频 → 选择 → 运行

3. 直接模式：python run.py data/videos/... --homography ... --risk-params ...
   → 直接运行（原方式）
"""

import argparse
import os
import sys
from pathlib import Path
from utils.config_loader import load_intersections

ROOT = Path(__file__).parent


# ============================================================
# 交互式选择
# ============================================================

def _load_intersections():
    """加载 intersections.json（使用 utils/config_loader.py 的统一实现）"""
    cfg_path = ROOT / "configs" / "intersections.json"
    if not cfg_path.exists():
        return None
    return load_intersections(cfg_path)


def list_videos(video_dir: str | Path) -> list[Path]:
    """列出视频目录中的视频文件"""
    d = ROOT / video_dir if isinstance(video_dir, str) else video_dir
    if not d.exists():
        return []
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    return sorted([f for f in d.iterdir() if f.suffix.lower() in exts])


def choose(prompt: str, items: list, default: int = 1) -> int:
    """交互选择"""
    print(f"\n{prompt}")
    for i, item in enumerate(items, 1):
        print(f"  [{i}] {item}")
    while True:
        raw = input(f"  选择 [1-{len(items)}] (默认 {default}): ").strip()
        if not raw:
            return default - 1
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return idx
        except ValueError:
            pass
        print(f"  请输入 1-{len(items)}")


def resolve_relative(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


# ============================================================
# 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual System - 交叉口拥堵分析系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 输入（改为可选，不传则进入交互模式）
    parser.add_argument('source', nargs='?', default=None, help='视频文件路径')
    parser.add_argument('--site', default=None, help='路口站点 key（交互模式用）')

    # 模型
    parser.add_argument('--model', default='data/models/yolo11s.pt', help='YOLO 模型路径')
    parser.add_argument('--imgsz', type=int, default=640, help='推理图像尺寸')
    parser.add_argument('--conf', type=float, default=0.15, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.40, help='IoU 阈值')
    parser.add_argument('--tracker', default='configs/bytetrack_stable.yaml', help='跟踪器配置')

    # 标定（改为可选，交互模式自动解析）
    parser.add_argument('--homography', default=None, help='单应性配置 JSON 路径')
    parser.add_argument('--risk-params', default=None, help='风险参数 JSON 路径')

    # 卷积参数
    parser.add_argument('--grid-size', type=int, default=64, help='BEV 网格边长')
    parser.add_argument('--world-width-m', type=float, default=40.0, help='世界坐标宽度 (m)')
    parser.add_argument('--world-height-m', type=float, default=40.0, help='世界坐标高度 (m)')

    # 显示
    parser.add_argument('--show-windows', action=argparse.BooleanOptionalAction, default=True, help='显示窗口')
    parser.add_argument('--display-scale', type=float, default=1.0, help='显示缩放')
    parser.add_argument('--frame-stride', type=int, default=1, help='帧跳步')
    parser.add_argument('--target-fps', type=float, default=20.0, help='目标处理帧率')

    # 轨迹
    parser.add_argument('--trail-seconds', type=float, default=10.0, help='轨迹保留秒数')

    # 分析
    parser.add_argument('--ablation-enable', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ablation-levels', type=int, default=3)
    parser.add_argument('--realtime-congestion-interval', type=int, default=2)
    parser.add_argument('--live-write-interval', type=int, default=2)
    parser.add_argument('--async-writer', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--async-writer-queue', type=int, default=4)

    # 输出
    parser.add_argument('--live-dir', help='实时预览输出目录')
    parser.add_argument('--events-dir', help='事件输出目录')
    parser.add_argument('--charts-dir', help='图表输出目录')
    parser.add_argument('--max-frames', type=int, default=0, help='最大处理帧数 (0=无限)')

    # 车辆配置
    parser.add_argument('--vehicle-size-path', default='data/vehicle/vehicle_size_m.json')
    parser.add_argument('--vehicle-arrow-style-path', default='data/vehicle/vehicle_arrow_style.json')

    # BEV 四边形
    parser.add_argument('--bev-quad',
                        default='0.38,0.24;0.72,0.30;0.64,0.80;0.16,0.62',
                        help='BEV 四边形 TL_x,TL_y;TR_x,TR_y;BR_x,BR_y;BL_x,BL_y')

    # 第三面板模式（Streamlit 控制台兼容）
    parser.add_argument('--third-panel-mode', default='quality',
                        choices=['quality', 'balanced', 'fast'],
                        help='第三面板渲染模式')

    # Backtest 调试模式
    parser.add_argument('--backtest', action=argparse.BooleanOptionalAction, default=False,
                        help='启用冲突检测调试窗口（可视化内部交织检测过程）')

    return parser.parse_args()


def ensure_runtime_ready():
    required = ['cv2', 'numpy', 'ultralytics']
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)


def main():
    ensure_runtime_ready()
    os.chdir(ROOT)
    args = parse_args()

    # ── 交互模式：选择站点和视频 ──────────────────────────
    if args.source is None or (args.homography is None and args.site):
        cfg = _load_intersections()
        if cfg is None or not cfg['sites']:
            print("[ERROR] configs/intersections.json 未找到或无站点")
            sys.exit(1)

        sites = cfg['sites']
        site_keys = list(sites.keys())

        # 选择站点
        if args.site and args.site in sites:
            site_key = args.site
        else:
            default_idx = max(0, site_keys.index(cfg['default_site']) if cfg['default_site'] in site_keys else 0)
            idx = choose(f"选择交叉口站点:", site_keys, default_idx + 1)
            site_key = site_keys[idx]

        site = sites[site_key]
        print(f"  站点: {site_key} ({site.get('display_name', '')})")

        # 列出该站视频
        video_dir = site.get('video_dir', '')
        videos = list_videos(ROOT / video_dir) if video_dir else []
        if not videos:
            # 也看默认 mp4 目录
            videos = list_videos(ROOT / 'mp4')
        if not videos:
            print(f"[ERROR] 站点 {site_key} 无视频文件")
            sys.exit(1)

        video_labels = [v.name for v in videos]
        vidx = choose("选择视频文件:", video_labels)
        source = str(videos[vidx])

        # 自动解析标定和风险参数
        homography = resolve_relative(site.get('homography', 'configs/homography_points_example.json'))
        risk_params = resolve_relative(site.get('risk_params', 'configs/traffic_risk_params.json'))

        if not homography.exists():
            print(f"[ERROR] 标定文件不存在: {homography}")
            sys.exit(1)
        if not risk_params.exists():
            print(f"[ERROR] 风险参数文件不存在: {risk_params}")
            sys.exit(1)

        # 覆盖 args
        args.source = source
        args.homography = str(homography)
        args.risk_params = str(risk_params)
        args.site_key = site_key

        # 从站点配置读取模型和推理尺寸
        site_model = site.get('model')
        if site_model:
            args.model = str(resolve_relative(site_model))
        site_runtime = site.get('runtime', {})
        # 站点配置作为默认值，命令行参数优先
        # 记录哪些参数是用户显式传入的（值 != 默认值）
        _cli_overrides = {}
        for key, default in [('imgsz', 640), ('frame-stride', 1), ('target-fps', 20.0)]:
            cli_key = key.replace('-', '_')
            if getattr(args, cli_key, None) != default:
                _cli_overrides[key] = True

        if 'imgsz' in site_runtime and 'imgsz' not in _cli_overrides:
            args.imgsz = int(site_runtime['imgsz'])
        if 'frame-stride' in site_runtime and 'frame-stride' not in _cli_overrides:
            args.frame_stride = int(site_runtime['frame-stride'])
        if 'target-process-fps' in site_runtime and 'target-fps' not in _cli_overrides:
            args.target_fps = float(site_runtime['target-process-fps'])

        # 自动设置输出目录
        if not args.live_dir:
            args.live_dir = f"outputs/{site_key}/{Path(source).stem}/live"
        if not args.events_dir:
            args.events_dir = f"outputs/{site_key}/{Path(source).stem}/events"

        print(f"  视频: {Path(source).name}")
        print(f"  标定: {Path(args.homography).name}")
        print(f"  输出: outputs/{site_key}/{Path(source).stem}/")
        print()

    # ── 直接模式 ──────────────────────────────────────────
    else:
        if not args.homography:
            print("[ERROR] direct mode 需要 --homography 参数")
            sys.exit(1)
        if not args.risk_params:
            print("[ERROR] direct mode 需要 --risk-params 参数")
            sys.exit(1)
        args.site_key = getattr(args, 'site_key', 'default')

        # 直接模式也设置默认输出目录
        source_stem = Path(args.source).stem if args.source else 'output'
        if not args.live_dir:
            args.live_dir = f"outputs/{args.site_key}/{source_stem}/live"
        if not args.events_dir:
            args.events_dir = f"outputs/{args.site_key}/{source_stem}/events"

    # 如果 source 是数字，转为摄像头索引
    try:
        args.source = int(args.source)
    except (ValueError, TypeError):
        pass

    # ── 运行 ──────────────────────────────────────────────
    from pipeline.engine import PipelineEngine, EngineConfig

    config = EngineConfig(args)
    engine = PipelineEngine(config)

    try:
        engine.run()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
        engine.stop()


if __name__ == '__main__':
    main()
