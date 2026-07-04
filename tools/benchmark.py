"""
网格化性能基准测试工具

对不同 imgsz / conf / frame_stride 组合运行 pipeline，收集 FPS 和各阶段耗时。
结果输出到 CSV 并打印 Top-K。

Usage
-----
python tools/benchmark.py --source data/videos/default/vedio_000.mp4 --max-frames 60
"""

import argparse
import csv
import itertools
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# 匹配 run.py 的 [PERF] 输出格式
# [PERF] 共 60 帧，总耗时 12.34s
FINAL_HEAD_RE = re.compile(r"\[PERF\]\s+共\s+(\d+)\s+帧，总耗时\s+([0-9.]+)s")
#   detect: 8.12s (65.8%)  avg 135.3ms/frame
PERF_ROW_RE = re.compile(r"^\s+([a-zA-Z_]+):\s+([0-9.]+)s\s+\(\s*([0-9.]+)%\)\s+avg\s+([0-9.]+)ms/frame")
#   处理FPS: 4.9
FPS_RE = re.compile(r"处理FPS:\s+([0-9.]+)")


def parse_num_list(text: str, cast_fn):
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(cast_fn(x))
    return out


def run_one(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(proc.returncode), str(proc.stdout)


def parse_perf(stdout: str) -> dict:
    lines = stdout.splitlines()
    frames = None
    total_time = None
    eff_fps = None
    metrics = {}

    for line in lines:
        line_s = line.strip()

        m = FINAL_HEAD_RE.search(line_s)
        if m:
            frames = int(m.group(1))
            total_time = float(m.group(2))

        m = FPS_RE.search(line_s)
        if m:
            eff_fps = float(m.group(1))

        m = PERF_ROW_RE.match(line_s)
        if m:
            key = m.group(1)
            ms = float(m.group(4))
            metrics[key] = ms

    ok = frames is not None and frames > 0
    detect_ms = metrics.get("detect", 0.0)

    return {
        "ok": ok,
        "frames": frames or "",
        "eff_fps": eff_fps or "",
        "total_time_s": total_time or "",
        "detect_ms": detect_ms,
        "traj_ms": metrics.get("traj", 0.0),
        "motion_ms": metrics.get("motion", 0.0),
        "phi_ms": metrics.get("phi", 0.0),
        "viz_c_ms": metrics.get("viz_c", 0.0),
        "write_ms": metrics.get("write", 0.0),
        "raw": "",
    }


def build_cmd(args, imgsz: int, conf: float, frame_stride: int) -> list[str]:
    cmd = [
        args.python,
        str(args.detector),
        args.source,
        "--homography",
        args.homography,
        "--risk-params",
        args.risk_params,
        "--model",
        args.model,
        "--imgsz",
        str(int(imgsz)),
        "--conf",
        f"{float(conf):.3f}",
        "--iou",
        f"{float(args.iou):.3f}",
        "--no-show-windows",
        "--max-frames",
        str(int(args.max_frames)),
        "--frame-stride",
        str(int(frame_stride)),
    ]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Grid benchmark for pipeline performance (uses run.py)"
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--detector", type=Path, default=Path("run.py"))
    parser.add_argument("--source", type=str, default="data/videos/default/vedio_000.mp4")
    parser.add_argument("--homography", type=str, default="configs/homography_points_example.json")
    parser.add_argument("--risk-params", type=str, default="configs/traffic_risk_params.json")
    parser.add_argument("--model", type=str, default="data/models/yolo11s.pt")
    parser.add_argument("--imgsz-list", type=str, default="640,512")
    parser.add_argument("--conf-list", type=str, default="0.22,0.25")
    parser.add_argument("--frame-stride-list", type=str, default="1,2,3")
    parser.add_argument("--iou", type=float, default=0.40)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/benchmarks/infer_grid_results.csv"))
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    detector_path = (workspace / args.detector).resolve()
    if not detector_path.exists():
        print(f"[ERROR] entry script not found: {detector_path}")
        return 2

    args.detector = detector_path
    imgsz_list = parse_num_list(args.imgsz_list, int)
    conf_list = parse_num_list(args.conf_list, float)
    frame_stride_list = parse_num_list(args.frame_stride_list, int)
    repeat = max(1, int(args.repeat))

    grid = list(itertools.product(imgsz_list, conf_list, frame_stride_list))
    if not grid:
        print("[ERROR] empty grid")
        return 2

    print(f"[benchmark] total combinations={len(grid)} repeat={repeat} total runs={len(grid) * repeat}")
    rows = []
    run_idx = 0
    total_runs = len(grid) * repeat
    for imgsz, conf, frame_stride in grid:
        for r in range(repeat):
            run_idx += 1
            cmd = build_cmd(args, imgsz, conf, frame_stride)
            print(
                f"[benchmark] run {run_idx}/{total_runs} | imgsz={imgsz} conf={conf:.3f} "
                f"frame_stride={frame_stride} rep={r+1}/{repeat}"
            )
            rc, out = run_one(cmd, cwd=workspace)
            perf = parse_perf(out)
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "imgsz": int(imgsz),
                "conf": float(conf),
                "frame_stride": int(frame_stride),
                "repeat": int(r + 1),
                "return_code": int(rc),
                "ok": bool(perf["ok"] and rc == 0),
                "frames": perf.get("frames", ""),
                "eff_fps": perf.get("eff_fps", ""),
                "total_time_s": perf.get("total_time_s", ""),
                "detect_ms": perf.get("detect_ms", ""),
                "traj_ms": perf.get("traj_ms", ""),
                "motion_ms": perf.get("motion_ms", ""),
                "phi_ms": perf.get("phi_ms", ""),
                "viz_c_ms": perf.get("viz_c_ms", ""),
                "write_ms": perf.get("write_ms", ""),
            }
            rows.append(row)

    out_csv = (workspace / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "imgsz", "conf", "frame_stride", "repeat",
        "return_code", "ok", "frames", "eff_fps", "total_time_s",
        "detect_ms", "traj_ms", "motion_ms", "phi_ms", "viz_c_ms", "write_ms",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    ok_rows = [r for r in rows if bool(r.get("ok"))]
    def _to_float(v, default=1e18):
        try:
            return float(v)
        except Exception:
            return float(default)

    ranked = sorted(
        ok_rows,
        key=lambda r: -_to_float(r.get("eff_fps"), default=-1e18),
    )
    top_k = max(1, int(args.top_k))
    print(f"\n[benchmark] wrote csv: {out_csv}")
    print(f"[benchmark] ok runs: {len(ok_rows)}/{len(rows)}")
    print("[benchmark] top results:")
    for i, r in enumerate(ranked[:top_k], start=1):
        print(
            f"  #{i:02d} fps={float(r['eff_fps']):.1f} detect={float(r['detect_ms']):.1f}ms "
            f"| imgsz={r['imgsz']} conf={float(r['conf']):.3f} stride={r['frame_stride']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
