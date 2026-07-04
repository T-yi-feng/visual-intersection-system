import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "intersections.json"


def load_intersections(cfg_path: Path) -> dict:
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    sites = raw.get("sites", {})
    if not isinstance(sites, dict) or len(sites) == 0:
        raise RuntimeError("Invalid intersections config: missing 'sites'")
    default_site = raw.get("default_site")
    if default_site not in sites:
        default_site = next(iter(sites.keys()))
    return {"sites": sites, "default_site": default_site}


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def list_site_videos(site_cfg: dict) -> list[Path]:
    video_dir = resolve_path(ROOT, str(site_cfg.get("video_dir", "")))
    if not video_dir.exists():
        return []
    rows = _list_site_videos_cached(str(video_dir.resolve()))
    return [Path(p) for p in rows]


@st.cache_data(show_spinner=False, ttl=2)
def _list_site_videos_cached(video_dir_text: str) -> list[str]:
    p = Path(video_dir_text)
    if not p.exists():
        return []
    return sorted([
        str(x)
        for x in p.iterdir()
        if x.is_file() and x.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    ])


def source_tag(source_token: str) -> str:
    if source_token.isdigit():
        return f"camera_{source_token}"
    stem = Path(source_token).stem
    return stem if stem else "source"


def peak_dir_for(site_key: str, source_token: str) -> Path:
    return ROOT / "outputs" / "phi_peak_arrow_analysis" / site_key / source_tag(source_token)


def run_dir_for(output_root: Path, site_key: str, source_token: str) -> Path:
    return output_root / "sites" / site_key / source_tag(source_token)


def peak_dir_for_root(output_root: Path, site_key: str, source_token: str) -> Path:
    return output_root / site_key / source_tag(source_token) / "events"


def rel_text(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def render_image_compat(image_data, caption: str | None = None) -> None:
    try:
        st.image(image_data, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image_data, caption=caption, use_column_width=True)
        except Exception:
            try:
                st.image(image_data, caption=caption)
            except Exception:
                st.write("Image preview unavailable in current Streamlit build.")


def _fmt_metric_float(v, digits: int = 3) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "--"


def _fmt_metric_int(v) -> str:
    if v is None:
        return "--"
    try:
        return str(int(v))
    except Exception:
        return "--"


def _load_site_config(site_key: str) -> dict:
    """加载 intersections.json 中指定站点的配置"""
    cfg_path = ROOT / "configs" / "intersections.json"
    try:
        data = json.loads(cfg_path.read_text(encoding='utf-8'))
        return data.get('sites', {}).get(site_key, {})
    except Exception:
        return {}


def _build_pipeline_cmd(
    site_key: str,
    source: str,
    model: str,
    imgsz: int,
    conf: float,
    iou: float,
    show_windows: int,
    ablation_enable: int,
    realtime_congestion_refresh_interval: int,
    live_preview_write_interval: int,
    async_live_writer: int,
    async_live_writer_queue: int,
    third_panel_mode: str,
    max_frames: int,
    session_output_root: str,
) -> list[str]:
    # 从站点配置读取 homography 和 risk_params 路径
    site_cfg = _load_site_config(site_key)
    homography_path = site_cfg.get('homography', 'configs/homography_points_example.json')
    risk_params_path = site_cfg.get('risk_params', 'configs/traffic_risk_params.json')

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run.py"),
        source,
        "--site-key",
        site_key,
        "--model",
        model,
        "--imgsz",
        str(imgsz),
        "--conf",
        f"{conf:.3f}",
        "--iou",
        f"{iou:.3f}",
        "--homography",
        homography_path,
        "--risk-params",
        risk_params_path,
    ]
    if int(show_windows):
        cmd.append("--show-windows")
    if int(ablation_enable):
        cmd.append("--ablation-enable")
    cmd.extend([
        "--realtime-congestion-interval",
        str(int(realtime_congestion_refresh_interval)),
        "--live-write-interval",
        str(int(live_preview_write_interval)),
    ])
    if int(async_live_writer):
        cmd.append("--async-writer")
    cmd.extend([
        "--async-writer-queue",
        str(int(async_live_writer_queue)),
        "--third-panel-mode",
        third_panel_mode if third_panel_mode in ("quality", "balanced", "fast") else "quality",
    ])
    if int(max_frames) > 0:
        cmd.extend(["--max-frames", str(int(max_frames))])
    return cmd


def _normalize_windows_return_code(rc: int) -> int:
    try:
        v = int(rc)
    except Exception:
        return rc
    if v >= (1 << 31):
        return v - (1 << 32)
    return v


def run_pipeline_async(
    site_key: str,
    source: str,
    model: str,
    imgsz: int,
    conf: float,
    iou: float,
    show_windows: int,
    ablation_enable: int,
    realtime_congestion_refresh_interval: int,
    live_preview_write_interval: int,
    async_live_writer: int,
    async_live_writer_queue: int,
    third_panel_mode: str,
    max_frames: int,
    session_output_root: str,
    run_log_file: Path,
) -> tuple[subprocess.Popen | None, str]:
    cmd = _build_pipeline_cmd(
        site_key=site_key,
        source=source,
        model=model,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        show_windows=show_windows,
        ablation_enable=ablation_enable,
        realtime_congestion_refresh_interval=realtime_congestion_refresh_interval,
        live_preview_write_interval=live_preview_write_interval,
        async_live_writer=async_live_writer,
        async_live_writer_queue=async_live_writer_queue,
        third_panel_mode=third_panel_mode,
        max_frames=max_frames,
        session_output_root=session_output_root,
    )
    try:
        run_log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(run_log_file, "w", encoding="utf-8", errors="replace", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Child keeps its own handle; parent closes immediately to avoid stale buffering/locks.
        f.close()
        return proc, ""
    except Exception as e:
        return None, str(e)


def launch_quick_calibration(
    site_key: str,
    top_m: float | None = None,
    right_m: float | None = None,
    bottom_m: float | None = None,
    left_m: float | None = None,
) -> tuple[bool, str]:
    script = ROOT / "tools" / "calibrate_homography.py"
    if not script.exists():
        return False, f"Missing script: {script}"

    cmd = [
        sys.executable,
        str(script),
        "--site",
        site_key,
    ]
    if top_m is not None and top_m > 0:
        cmd.extend(["--top-m", f"{float(top_m):.6f}"])
    if right_m is not None and right_m > 0:
        cmd.extend(["--right-m", f"{float(right_m):.6f}"])
    if bottom_m is not None and bottom_m > 0:
        cmd.extend(["--bottom-m", f"{float(bottom_m):.6f}"])
    if left_m is not None and left_m > 0:
        cmd.extend(["--left-m", f"{float(left_m):.6f}"])
    if all(v is not None and float(v) > 0 for v in [top_m, right_m, bottom_m, left_m]):
        cmd.append("--no-edge-prompt")
    try:
        kwargs = {"cwd": str(ROOT)}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(cmd, **kwargs)
        return True, ""
    except Exception as e:
        return False, str(e)


def clear_run_outputs(output_root_text: str, site_key: str, source: str) -> tuple[bool, str]:
    try:
        output_root = resolve_path(ROOT, output_root_text)
        run_dir = output_root / site_key / source_tag(source)
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        return True, ""
    except Exception as e:
        return False, str(e)


def terminate_stale_pipeline_processes() -> tuple[bool, str]:
    if not sys.platform.startswith("win"):
        return True, ""
    ps_cmd = (
        "$pattern='test0\\.py|detect_and_bev\\.py';"
        "$procs=Get-CimInstance Win32_Process | Where-Object { "
        "$_.Name -match 'python' -and $_.CommandLine -ne $null -and "
        "($_.CommandLine -match $pattern) };"
        "foreach($p in $procs){ try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {} };"
        "Write-Output ('killed=' + @($procs).Count)"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return True, out.strip()
    except Exception as e:
        return False, str(e)


def _read_live_snapshot(live_dir: Path, min_mtime: float | None = None) -> dict:
    snapshot = {
        "frames": {},
        "metrics": {},
        "phi_df": pd.DataFrame(),
    }

    frame_map = {
        "window_3_third_phi.jpg": live_dir / "window_3_third_phi.jpg",
        "window_4_realtime_highlight.jpg": live_dir / "window_4_realtime_highlight.jpg",
    }
    for k, p in frame_map.items():
        if p.exists():
            try:
                if min_mtime is not None and p.stat().st_mtime < float(min_mtime):
                    continue
                snapshot["frames"][k] = p.read_bytes()
            except Exception:
                pass

    metrics_file = live_dir / "live_metrics.json"
    if metrics_file.exists():
        try:
            if min_mtime is not None and metrics_file.stat().st_mtime < float(min_mtime):
                raise RuntimeError("stale metrics")
            snapshot["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    live_phi_csv = live_dir / "live_phi_timeline.csv"
    if live_phi_csv.exists():
        df_live_phi = read_csv_safe(live_phi_csv)
        if not df_live_phi.empty and "time_s" in df_live_phi.columns and "phi_t" in df_live_phi.columns:
            df_live_phi = df_live_phi.copy()
            df_live_phi["time_s"] = pd.to_numeric(df_live_phi["time_s"], errors="coerce")
            df_live_phi["phi_t"] = pd.to_numeric(df_live_phi["phi_t"], errors="coerce")
            df_live_phi = df_live_phi.dropna(subset=["time_s", "phi_t"]).sort_values("time_s")
            if min_mtime is None:
                snapshot["phi_df"] = df_live_phi
            else:
                try:
                    if live_phi_csv.stat().st_mtime >= float(min_mtime):
                        snapshot["phi_df"] = df_live_phi
                except Exception:
                    pass

    return snapshot


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def collect_event_dirs(peak_dir: Path) -> list[Path]:
    if not peak_dir.exists():
        return []
    return sorted([p for p in peak_dir.iterdir() if p.is_dir() and p.name.startswith("event_")], reverse=True)


def aggregate_vehicle_table(event_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for ev in event_dirs:
        for csv_path in sorted(ev.glob("*_arrow_intersections_by_id.csv")):
            df = read_csv_safe(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df["event_dir"] = ev.name
            df["file"] = csv_path.name
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = [
        "event_dir",
        "file",
        "track_id",
        "vehicle_type",
        "intersection_count",
        "intersected_vehicle_ids",
        "phi",
        "frame_index",
        "time_s",
        "rank_in_output",
        "selection_reason",
    ]
    return out[[c for c in cols if c in out.columns]]


def aggregate_group_table(event_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for ev in event_dirs:
        for csv_path in sorted(ev.glob("*_group_lambda_by_k.csv")):
            df = read_csv_safe(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df["event_dir"] = ev.name
            df["file"] = csv_path.name
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    cols = ["event_dir", "file", "group_k", "n_k", "gamma_ij_sum", "lambda_k", "vehicle_ids"]
    return out[[c for c in cols if c in out.columns]]


def aggregate_intersections_table(event_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for ev in event_dirs:
        detail_json = ev / "event_peak_analysis.json"
        if not detail_json.exists():
            continue
        try:
            detail = json.loads(detail_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        top2 = detail.get("top2", [])
        for item in top2:
            rank = item.get("rank")
            reason = item.get("selection_reason")
            phi = item.get("phi")
            frame_idx = item.get("frame_idx")
            time_s = item.get("time_s")
            for inter in item.get("intersections", []):
                rows.append(
                    {
                        "event_dir": ev.name,
                        "rank": rank,
                        "selection_reason": reason,
                        "phi": phi,
                        "frame_idx": frame_idx,
                        "time_s": time_s,
                        "id_a": inter.get("id_a"),
                        "id_b": inter.get("id_b"),
                        "x": inter.get("x"),
                        "y": inter.get("y"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    cols = ["event_dir", "rank", "selection_reason", "phi", "frame_idx", "time_s", "id_a", "id_b", "x", "y"]
    return out[[c for c in cols if c in out.columns]]


def export_phi_timeline_png(df_live_phi: pd.DataFrame, out_path: Path) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except Exception as e:
        return False, f"matplotlib unavailable: {e}"

    if len(df_live_phi) < 2:
        return False, "not enough timeline points"

    x = df_live_phi["time_s"].to_numpy(dtype=float)
    y = df_live_phi["phi_t"].to_numpy(dtype=float)
    points = np.column_stack((x, y))
    segments = np.stack([points[:-1], points[1:]], axis=1)
    seg_color = np.clip((y[:-1] + y[1:]) / 2.0, 0.0, 1.0)

    cmap = LinearSegmentedColormap.from_list("phi_cold_hot", ["#2E6BFF", "#FF3B30"])
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=180)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_color)
    lc.set_linewidth(1.0)
    lc.set_alpha(0.95)
    ax.add_collection(lc)

    sc = ax.scatter(x, y, c=np.clip(y, 0.0, 1.0), cmap=cmap, norm=norm, s=12, zorder=3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("综合拥堵指数")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1e-3
    y_pad = max(0.02, 0.06 * (y_max - y_min))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("时间线 (秒)")
    ax.set_ylabel("综合拥堵指数")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True, ""


def ensure_session_state_defaults() -> None:
    defaults = {
        "_boot_initialized": True,
        "live_frozen": False,
        "live_snapshot": lambda: {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()},
        "show_calib_confirm": False,
        "last_log": "",
        "last_params": lambda: {},
        "last_live_metrics": lambda: {},
        "ui_site_key": "--请选择交叉口--",
        "ui_video_name": "--请选择视频--",
        "has_run_current_session": False,
        "active_site_key": "",
        "active_source_value": "",
        "run_process": None,
        "run_log_file": "",
        "last_run_rc": None,
        "run_started_at": 0.0,
        "intro_hint_dismiss_until": 0.0,
        "splash_seen": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v() if callable(v) else v


def main() -> None:
    st.set_page_config(page_title="Traffic Risk Smart Console", page_icon="TS", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;700;900&family=Rajdhani:wght@500;700&display=swap');

        :root {
            --bg-deep: #060b14;
            --bg-mid: #0b1730;
            --bg-soft: #132746;
            --panel: rgba(14, 27, 49, 0.86);
            --panel-strong: rgba(10, 21, 40, 0.95);
            --border: rgba(154, 190, 233, 0.32);
            --text-main: #f2f7ff;
            --text-muted: #aec3de;
            --accent-cold: #58a6ff;
            --accent-hot: #ff6b5f;
            --accent-gold: #e6c36a;
            --accent-cyan: #49d9ff;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stDecoration"] {
            display: none;
        }
        .stApp {
            background:
                radial-gradient(900px 540px at 8% -10%, rgba(89, 154, 214, 0.20) 0%, rgba(89, 154, 214, 0.00) 72%),
                radial-gradient(760px 460px at 94% -6%, rgba(221, 164, 96, 0.16) 0%, rgba(221, 164, 96, 0.00) 68%),
                conic-gradient(from 210deg at 82% 22%, rgba(24, 54, 97, 0.22), rgba(12, 23, 43, 0.04), rgba(24, 54, 97, 0.22)),
                linear-gradient(140deg, var(--bg-deep) 0%, var(--bg-mid) 54%, var(--bg-soft) 100%);
            color: var(--text-main);
            font-family: "Rajdhani", "Segoe UI", sans-serif;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(170, 205, 240, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(170, 205, 240, 0.05) 1px, transparent 1px);
            background-size: 48px 48px;
            mask-image: radial-gradient(circle at 50% 35%, black 20%, rgba(0, 0, 0, 0.32) 72%, transparent 100%);
            z-index: 0;
        }
        .block-container {
            position: relative;
            z-index: 1;
            padding-top: 3.6rem;
        }
        .card {
            padding: 14px 16px;
            border-radius: 16px;
            background:
                linear-gradient(160deg, rgba(32, 58, 96, 0.24), rgba(17, 33, 58, 0.20)),
                var(--panel);
            border: 1px solid var(--border);
            box-shadow: 0 12px 34px rgba(3, 8, 18, 0.34), inset 0 1px 0 rgba(227, 240, 255, 0.08);
            backdrop-filter: blur(7px);
            animation: riseFade 460ms ease-out;
        }
        .hero-title {
            margin-top: 0.2rem;
            margin-bottom: 0.42rem;
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-size: 2.15rem;
            line-height: 1.22;
            letter-spacing: 0.03em;
            font-weight: 800;
            color: #f4f8ff;
            text-shadow: 0 6px 20px rgba(4, 10, 20, 0.52);
            animation: focusSlideIn 900ms cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 1.05s;
        }
        .hero-subtitle {
            max-width: 980px;
            margin-bottom: 1.08rem;
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-size: 1.04rem;
            line-height: 1.7;
            letter-spacing: 0.01em;
            color: #c8d7ec;
            opacity: 0.96;
            animation: focusSlideIn 980ms cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 1.12s;
        }
        .hero-subtitle .lead {
            color: #eaf4ff;
            font-weight: 700;
        }
        .title {
            font-size: 30px;
            font-weight: 750;
            letter-spacing: 0.6px;
            color: var(--text-main);
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
        }
        .muted {
            color: var(--text-muted);
        }
        .stSidebar {
            background:
                linear-gradient(180deg, rgba(8, 18, 34, 0.96), rgba(10, 21, 39, 0.96));
            border-right: 1px solid rgba(146, 183, 228, 0.24);
        }
        .stSidebar .stMarkdown,
        .stSidebar label,
        .stSidebar .stSelectbox,
        .stSidebar .stTextInput,
        .stSidebar .stNumberInput {
            color: var(--text-main) !important;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(158, 191, 236, 0.34);
            background: linear-gradient(140deg, rgba(32, 63, 110, 0.92), rgba(18, 38, 70, 0.92));
            color: #f6fbff;
            font-weight: 650;
            transition: transform 180ms cubic-bezier(0.2, 0.7, 0.2, 1), box-shadow 180ms ease, border-color 180ms ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px) scale(1.015);
            border-color: rgba(215, 176, 103, 0.50);
            box-shadow: 0 8px 18px rgba(4, 10, 22, 0.40);
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #9fd4ff, #6eb8ff 42%, #54a8f4 100%);
            color: #04152c !important;
            border: 1px solid rgba(173, 219, 255, 0.9);
            font-weight: 800;
            text-shadow: none;
            box-shadow: 0 10px 24px rgba(43, 110, 173, 0.35);
        }
        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, #b9e0ff, #7ac3ff 45%, #5db1fb 100%);
            color: #031226 !important;
            border-color: rgba(204, 235, 255, 0.98);
            box-shadow: 0 14px 30px rgba(58, 131, 199, 0.42);
        }
        .bubble {
            border-radius: 18px;
            padding: 15px 18px;
            border: 1px solid rgba(151, 194, 238, 0.28);
            background: linear-gradient(165deg, rgba(30, 56, 96, 0.34), rgba(15, 31, 58, 0.54));
            box-shadow: 0 16px 32px rgba(4, 10, 20, 0.32), inset 0 1px 0 rgba(220, 238, 255, 0.09);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
            animation: riseFade 500ms ease-out;
        }
        .bubble::after {
            content: "";
            position: absolute;
            width: 140px;
            height: 140px;
            right: -54px;
            top: -56px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(91, 200, 255, 0.25), rgba(91, 200, 255, 0));
            pointer-events: none;
        }
        .bubble.alert {
            background: linear-gradient(165deg, rgba(126, 33, 43, 0.44), rgba(53, 18, 25, 0.66));
            border-color: rgba(243, 132, 132, 0.32);
        }
        .bubble.dismiss {
            animation: bubbleOut 380ms ease-in forwards;
        }
        .bubble.info {
            background: linear-gradient(165deg, rgba(28, 58, 106, 0.42), rgba(17, 30, 59, 0.70));
        }
        .bubble-title {
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-size: 1.06rem;
            font-weight: 700;
            color: #eff5ff;
            margin-bottom: 0.35rem;
            letter-spacing: 0.02em;
        }
        .bubble-text {
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-size: 0.97rem;
            line-height: 1.64;
            color: #d8e5f7;
        }
        .status-capsule {
            border-radius: 18px;
            padding: 14px 16px;
            border: 1px solid rgba(151, 194, 238, 0.28);
            background: linear-gradient(160deg, rgba(24, 50, 88, 0.44), rgba(13, 27, 52, 0.74));
            box-shadow: 0 14px 30px rgba(4, 10, 20, 0.32), inset 0 1px 0 rgba(220, 238, 255, 0.10);
            position: relative;
            overflow: hidden;
        }
        .status-capsule .status-label {
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-weight: 700;
            color: #eef6ff;
            margin-bottom: 6px;
        }
        .status-capsule .status-row {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #d4e6ff;
            font-weight: 600;
            line-height: 1.4;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex: 0 0 10px;
            box-shadow: 0 0 0 0 rgba(73, 217, 255, 0.45);
            animation: breatheDot 2.1s ease-in-out infinite;
        }
        .status-capsule.live .status-dot {
            background: #56f0ff;
        }
        .status-capsule.frozen .status-dot {
            background: #ffb266;
            box-shadow: 0 0 0 0 rgba(255, 178, 102, 0.45);
        }
        .status-capsule::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            right: -80px;
            top: -86px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(86, 212, 255, 0.25), rgba(86, 212, 255, 0));
            animation: breatheGlow 2.8s ease-in-out infinite;
            pointer-events: none;
        }
        .status-capsule.frozen::after {
            background: radial-gradient(circle, rgba(255, 182, 111, 0.25), rgba(255, 182, 111, 0));
        }
        @keyframes riseFade {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes bubbleOut {
            from { opacity: 1; transform: translateY(0) scale(1); }
            to { opacity: 0; transform: translateY(-8px) scale(0.985); }
        }
        @keyframes splashRun {
            from { width: 8%; }
            to { width: 100%; }
        }
        @keyframes splashOut {
            to {
                opacity: 0;
                visibility: hidden;
            }
        }
        @keyframes breatheGlow {
            0% { opacity: 0.35; transform: scale(0.92); }
            50% { opacity: 0.7; transform: scale(1.02); }
            100% { opacity: 0.35; transform: scale(0.92); }
        }
        @keyframes breatheDot {
            0% { box-shadow: 0 0 0 0 rgba(73, 217, 255, 0.45); }
            70% { box-shadow: 0 0 0 8px rgba(73, 217, 255, 0.0); }
            100% { box-shadow: 0 0 0 0 rgba(73, 217, 255, 0.0); }
        }
        .stTabs [data-baseweb="tab"] {
            color: #d8e7fb;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: #fff9eb !important;
            border-bottom-color: var(--accent-gold) !important;
        }
        .stDataFrame {
            border: 1px solid rgba(151, 188, 230, 0.25);
            border-radius: 12px;
            overflow: hidden;
        }
        .window-title {
            margin-top: 0.48rem;
            margin-bottom: 0.14rem;
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-size: 1.08rem;
            font-weight: 700;
            color: #eef6ff;
            letter-spacing: 0.02em;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.36);
        }
        .window-note {
            margin-top: 0.14rem;
            margin-bottom: 0.42rem;
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid rgba(156, 198, 238, 0.30);
            background: linear-gradient(160deg, rgba(20, 43, 76, 0.62), rgba(13, 29, 56, 0.68));
            color: #d9e9fb;
            font-size: 0.92rem;
            line-height: 1.58;
        }
        .window-note b {
            color: #f4fbff;
            font-weight: 700;
        }
        .section-note {
            margin: 0.22rem 0 0.72rem 0;
            padding: 8px 12px;
            border-radius: 10px;
            border: 1px solid rgba(156, 198, 238, 0.26);
            background: linear-gradient(160deg, rgba(20, 43, 76, 0.56), rgba(13, 29, 56, 0.62));
            color: #d9e9fb;
            font-size: 0.9rem;
            line-height: 1.56;
            animation: focusSlideIn 780ms cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 0.08s;
        }
        .section-note b {
            color: #f4fbff;
        }
        .startup-splash {
            position: fixed;
            inset: 0;
            z-index: 9998;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(150deg, rgba(4, 10, 20, 0.95), rgba(8, 18, 35, 0.93));
            animation: splashOut 680ms ease 1.18s forwards;
            pointer-events: none;
        }
        .startup-splash .splash-core {
            width: min(520px, 74vw);
            text-align: center;
        }
        .startup-splash .splash-text {
            color: #dff1ff;
            font-family: "Noto Serif SC", "Microsoft YaHei", serif;
            font-size: 1.04rem;
            letter-spacing: 0.08em;
            margin-bottom: 14px;
            opacity: 0.94;
        }
        .startup-splash .splash-track {
            width: 100%;
            height: 2px;
            border-radius: 999px;
            background: rgba(127, 178, 230, 0.22);
            overflow: hidden;
        }
        .startup-splash .splash-bar {
            height: 100%;
            width: 36%;
            border-radius: 999px;
            background: linear-gradient(90deg, #88ceff, #44d8ff);
            animation: splashRun 1.1s cubic-bezier(0.2, 0.6, 0.2, 1) forwards;
            box-shadow: 0 0 10px rgba(86, 198, 255, 0.46);
        }
        @keyframes focusSlideIn {
            0% {
                opacity: 0;
                filter: blur(10px);
                transform: translateX(28px);
            }
            100% {
                opacity: 1;
                filter: blur(0px);
                transform: translateX(0);
            }
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 2.2rem;
            }
            .hero-title {
                font-size: 1.72rem;
            }
            .hero-subtitle {
                font-size: 0.94rem;
                line-height: 1.58;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cfg = load_intersections(CONFIG_PATH)
    sites = cfg["sites"]

    st.markdown('<div class="hero-title">Traffic Risk Smart Console</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle"><span class="lead">Unified run control, Phi/event analytics, and ID-level tables.</span><br/>'
        '以高可读排版承载实时风险计算，确保数据展示、交互反馈与视觉层次在同一界面中保持稳定与克制。</div>',
        unsafe_allow_html=True,
    )

    ensure_session_state_defaults()

    if not st.session_state.get("splash_seen", False):
        st.markdown(
            """
            <div class="startup-splash">
                <div class="splash-core">
                    <div class="splash-text">TRAFFIC RISK CONSOLE LOADING</div>
                    <div class="splash-track"><div class="splash-bar"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["splash_seen"] = True

    top_left, top_mid, top_calib, top_right = st.columns([2, 2, 3, 5])
    with top_left:
        if st.button("Stop (冻结输出)", use_container_width=True, key="global_stop_btn"):
            st.session_state["live_frozen"] = True
    with top_mid:
        if st.button("Resume (恢复更新)", use_container_width=True, key="global_resume_btn"):
            st.session_state["live_frozen"] = False
    with top_calib:
        if st.button("Click2Homo 快速标定", use_container_width=True, key="main_open_quick_calib"):
            st.session_state["show_calib_confirm"] = True
    with top_right:
        frozen = st.session_state.get("live_frozen", False)
        state_css = "frozen" if frozen else "live"
        state_txt = "已冻结：实时视频/图表/指标暂停更新" if frozen else "实时更新中"
        st.markdown(
            (
                f"<div class='status-capsule {state_css}'>"
                "<div class='status-label'>Live Status</div>"
                f"<div class='status-row'><span class='status-dot'></span><span>{state_txt}</span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    dismiss_until = float(st.session_state.get("intro_hint_dismiss_until", 0.0) or 0.0)
    now_ts = time.time()
    intro_is_dismissing = dismiss_until > now_ts
    show_intro_hints = (not st.session_state.get("has_run_current_session", False)) or intro_is_dismissing
    intro_bubble_cls = "bubble alert dismiss" if intro_is_dismissing else "bubble alert"
    if intro_is_dismissing:
        st_autorefresh(interval=200, key="intro_hint_dismiss_refresh")
    elif dismiss_until > 0:
        st.session_state["intro_hint_dismiss_until"] = 0.0

    if show_intro_hints:
        st.markdown(
            f"""
            <div class="{intro_bubble_cls}" style="margin-top:10px;">
                <div class="bubble-title">重要提示</div>
                <div class="bubble-text">页面初始不显示任何历史结果。请先选择交叉口和视频，再点击 Run Full Pipeline。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.get("show_calib_confirm", False):
        st.markdown(
            """
            <div class="bubble info" style="margin-top:10px;">
                <div class="bubble-title">Click2Homo 快速标定</div>
                <div class="bubble-text">确认后将启动独立大窗口标定程序，按提示点击 4 个角点并保存。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ui_site = st.session_state.get("ui_site_key", "")
        calib_default = ui_site if ui_site in sites else cfg["default_site"]
        calib_site_key = st.selectbox(
            "选择标定交叉口",
            options=list(sites.keys()),
            index=list(sites.keys()).index(calib_default) if calib_default in sites else 0,
            key="quick_calib_site_main",
        )
        cal_cfg = sites.get(calib_site_key, {}).get("calibration", {}) if isinstance(sites.get(calib_site_key, {}), dict) else {}
        world_edges = cal_cfg.get("world_edges_m", {}) if isinstance(cal_cfg, dict) else {}
        d_top = float(world_edges.get("top", 12.6))
        d_right = float(world_edges.get("right", 20.0))
        d_bottom = float(world_edges.get("bottom", d_top))
        d_left = float(world_edges.get("left", d_right))

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            top_m = st.number_input("Top(m)", min_value=0.1, value=float(d_top), step=0.1, key="calib_top_m")
        with e2:
            right_m = st.number_input("Right(m)", min_value=0.1, value=float(d_right), step=0.1, key="calib_right_m")
        with e3:
            bottom_m = st.number_input("Bottom(m)", min_value=0.1, value=float(d_bottom), step=0.1, key="calib_bottom_m")
        with e4:
            left_m = st.number_input("Left(m)", min_value=0.1, value=float(d_left), step=0.1, key="calib_left_m")
        c_ok, c_cancel = st.columns(2)
        with c_ok:
            if st.button("确定并启动标定窗口", use_container_width=True, key="quick_calib_confirm_main"):
                ok, err = launch_quick_calibration(
                    calib_site_key,
                    top_m=float(top_m),
                    right_m=float(right_m),
                    bottom_m=float(bottom_m),
                    left_m=float(left_m),
                )
                if ok:
                    st.success("快速标定窗口已启动，请在新窗口中完成点击与保存。")
                    st.session_state["show_calib_confirm"] = False
                else:
                    st.error(f"快速标定启动失败: {err}")
        with c_cancel:
            if st.button("取消", use_container_width=True, key="quick_calib_cancel_main"):
                st.session_state["show_calib_confirm"] = False

    with st.sidebar:
        st.header("Run Control")
        st.markdown(
            """
            <div class="bubble info">
                <div class="bubble-title">实时输出控制</div>
                <div class="bubble-text">Stop 会冻结实时视频、时间线和实时指标，Resume 恢复更新。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_stop, c_resume = st.columns(2)
        with c_stop:
            if st.button("Stop", use_container_width=True, key="sidebar_stop_btn"):
                st.session_state["live_frozen"] = True
        with c_resume:
            if st.button("Resume", use_container_width=True, key="sidebar_resume_btn"):
                st.session_state["live_frozen"] = False

        site_options = ["--请选择交叉口--", *list(sites.keys())]
        if st.session_state.get("ui_site_key") not in site_options:
            st.session_state["ui_site_key"] = "--请选择交叉口--"
        site_choice = st.selectbox("Site", options=site_options, key="ui_site_key")
        site_key = site_choice if site_choice in sites else ""

        videos = list_site_videos(sites[site_key]) if site_key else []
        video_map = {v.name: str(v) for v in videos}
        video_options = ["--请选择视频--", *list(video_map.keys())]
        if st.session_state.get("ui_video_name") not in video_options:
            st.session_state["ui_video_name"] = "--请选择视频--"
        video_choice = st.selectbox("Video", options=video_options, key="ui_video_name", disabled=not site_key)
        source_value = video_map.get(video_choice, "")

        c1, c2 = st.columns(2)
        with c1:
            model = st.text_input("Model", value="yolo11m.pt")
            imgsz = st.number_input("imgsz", min_value=640, max_value=1920, value=1280, step=32)
            conf = st.slider("conf", min_value=0.05, max_value=0.9, value=0.22, step=0.01)
        with c2:
            iou = st.slider("iou", min_value=0.05, max_value=0.9, value=0.40, step=0.01)
            show_windows = st.selectbox("Video Windows", options=[1, 0], index=1, format_func=lambda x: "ON" if x == 1 else "OFF")

        ablation_enable = st.selectbox(
            "Ablation Experiment",
            options=[1, 0],
            index=0,
            format_func=lambda x: "ON" if int(x) == 1 else "OFF",
            help="ON: 启用事件消融实验（剔除高贡献车辆并重算指标）；OFF: 关闭消融实验。",
        )

        third_mode = st.selectbox("Third Panel Mode", options=["quality", "balanced", "speed"], index=0)
        realtime_congestion_refresh_interval = st.number_input(
            "Realtime Congestion Refresh Interval",
            min_value=1,
            max_value=12,
            value=2,
            step=1,
            help="实时交织/贡献度着色每 N 帧刷新一次；仅影响显示频率，不改变检测精度。",
        )
        live_preview_write_interval = st.number_input(
            "Live Preview Write Interval",
            min_value=1,
            max_value=12,
            value=2,
            step=1,
            help="网页实时图片写盘间隔（N帧写一次）；增大可减少磁盘IO卡顿。",
        )
        async_live_writer = st.selectbox(
            "Async Live Writer",
            options=[1, 0],
            index=0,
            format_func=lambda x: "ON" if int(x) == 1 else "OFF",
            help="ON: 异步线程写实时图，减少主循环阻塞；OFF: 同步写图。",
        )
        async_live_writer_queue = st.number_input(
            "Async Live Writer Queue",
            min_value=1,
            max_value=8,
            value=2,
            step=1,
            help="异步写图最大排队任务数，建议 2~4。",
        )
        max_frames = st.number_input("max-frames (0=all)", min_value=0, max_value=200000, value=0, step=60)
        refresh_ms = st.slider("Realtime Refresh (ms)", min_value=300, max_value=5000, value=1000, step=100)

        if "session_output_root" not in st.session_state:
            st.session_state["session_output_root"] = "outputs/unified"
        session_output_root = st.text_input("Session Output Root", value=st.session_state["session_output_root"])
        st.session_state["session_output_root"] = session_output_root

        selection_ready = bool(site_key) and bool(source_value)
        if not selection_ready:
            st.info("请先选择交叉口和视频，再执行运行。")
        run_now = st.button(
            "Run Full Pipeline",
            use_container_width=True,
            type="primary",
            key="run_full_pipeline_btn",
            disabled=(not selection_ready),
        )

    run_proc = st.session_state.get("run_process")
    run_is_alive = False
    if run_proc is not None:
        try:
            run_is_alive = run_proc.poll() is None
        except Exception:
            run_is_alive = False

    active_site_for_refresh = st.session_state.get("active_site_key", "")
    active_source_for_refresh = st.session_state.get("active_source_value", "")
    if (
        (st.session_state.get("has_run_current_session", False) or run_is_alive)
        and bool(active_site_for_refresh)
        and bool(active_source_for_refresh)
        and (not st.session_state.get("live_frozen", False))
    ):
        st_autorefresh(interval=int(refresh_ms), key="live_refresh")

    if run_now:
        st.session_state["intro_hint_dismiss_until"] = time.time() + 0.42
        ok_kill, kill_msg = terminate_stale_pipeline_processes()
        if not ok_kill:
            st.error(f"终止旧后台进程失败，已阻止本次运行以避免输出冲突: {kill_msg}")
            st.session_state["has_run_current_session"] = False
            return
        elif kill_msg:
            st.caption(f"运行前清场: {kill_msg}")

        ok_clear, err_clear = clear_run_outputs(session_output_root, site_key, source_value)
        if not ok_clear:
            st.error(f"清理历史输出失败：{err_clear}")
            st.session_state["has_run_current_session"] = False
            return

        st.session_state["live_snapshot"] = {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()}
        st.session_state["last_live_metrics"] = {}
        st.session_state["last_log"] = ""

        active_run_dir = resolve_path(ROOT, session_output_root) / site_key / source_tag(source_value)
        run_log_file = active_run_dir / "live" / "web_run.log"
        proc, err = run_pipeline_async(
            site_key=site_key,
            source=source_value,
            model=model,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            show_windows=int(show_windows),
            ablation_enable=int(ablation_enable),
            realtime_congestion_refresh_interval=int(realtime_congestion_refresh_interval),
            live_preview_write_interval=int(live_preview_write_interval),
            async_live_writer=int(async_live_writer),
            async_live_writer_queue=int(async_live_writer_queue),
            third_panel_mode=third_mode,
            max_frames=int(max_frames),
            session_output_root=session_output_root,
            run_log_file=run_log_file,
        )
        if proc is None:
            st.error(f"启动运行失败：{err}")
            st.session_state["has_run_current_session"] = False
            return

        st.session_state["run_process"] = proc
        st.session_state["run_log_file"] = str(run_log_file)
        st.session_state["last_log"] = ""
        st.session_state["last_params"] = {
            "site": site_key,
            "source": source_value,
            "model": model,
            "imgsz": int(imgsz),
            "conf": float(conf),
            "iou": float(iou),
            "show_windows": int(show_windows),
            "ablation_enable": int(ablation_enable),
            "realtime_congestion_refresh_interval": int(realtime_congestion_refresh_interval),
            "live_preview_write_interval": int(live_preview_write_interval),
            "async_live_writer": int(async_live_writer),
            "async_live_writer_queue": int(async_live_writer_queue),
            "third_panel_mode": third_mode,
            "max_frames": int(max_frames),
            "session_output_root": session_output_root,
        }
        st.session_state["has_run_current_session"] = True
        st.session_state["active_site_key"] = site_key
        st.session_state["active_source_value"] = source_value
        st.session_state["last_run_rc"] = None
        st.session_state["run_started_at"] = time.time()
        st.success("任务已启动，正在后台运行。")

    run_proc = st.session_state.get("run_process")
    if run_proc is not None:
        rc = run_proc.poll()
        if rc is None:
            st.info("Running pipeline... 输出会在下方分类窗口实时出现。")
        else:
            log_path = Path(st.session_state.get("run_log_file", "")) if st.session_state.get("run_log_file") else None
            if log_path is not None and log_path.exists():
                try:
                    st.session_state["last_log"] = log_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            if int(rc) == 0:
                st.success("Run completed successfully.")
            else:
                rc_signed = _normalize_windows_return_code(int(rc))
                if int(rc_signed) != int(rc):
                    st.error(f"Run failed (exit code={rc}, signed={rc_signed}). 请展开日志查看原因。")
                else:
                    st.error(f"Run failed (exit code={rc}). 请展开日志查看原因。")
                # Keep current session context so user can still inspect generated artifacts/logs.
                st.session_state["has_run_current_session"] = True
            st.session_state["last_run_rc"] = int(rc)
            st.session_state["run_process"] = None

    if not st.session_state.get("has_run_current_session", False):
        init_cls = "bubble info dismiss" if intro_is_dismissing else "bubble info"
        st.markdown(
            f"""
            <div class="{init_cls}" style="margin-top:10px;">
                <div class="bubble-title">初始化状态</div>
                <div class="bubble-text">当前未进入运行态（不会显示历史结果）。请选择交叉口与视频后，再点击 Run Full Pipeline。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.get("last_log", "").strip():
            with st.expander("最近一次运行日志", expanded=False):
                st.code(st.session_state.get("last_log", ""), language="text")
        return

    active_site_key = st.session_state.get("active_site_key", "")
    active_source_value = st.session_state.get("active_source_value", "")
    if not (bool(active_site_key) and bool(active_source_value)):
        st.warning("当前会话没有有效运行上下文，请重新选择并点击 Run。")
        return

    output_root = resolve_path(ROOT, session_output_root)
    run_dir = output_root / active_site_key / source_tag(active_source_value)
    peak_dir = peak_dir_for_root(output_root, active_site_key, active_source_value)
    event_dirs = collect_event_dirs(peak_dir)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown('<div class="card"><b>Peak Output Dir</b><br/>' + rel_text(peak_dir) + "</div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="card"><b>Event Count</b><br/>{len(event_dirs)}</div>', unsafe_allow_html=True)
    with k3:
        src_short = Path(active_source_value).name if not active_source_value.isdigit() else active_source_value
        st.markdown(f'<div class="card"><b>Source</b><br/>{src_short}</div>', unsafe_allow_html=True)

    st.subheader("Embedded Video Windows")
    st.markdown(
        "<div class='section-note'><b>模块说明</b>：保留两个核心可视化窗口：风险主视图与实时高亮贡献视图。</div>",
        unsafe_allow_html=True,
    )
    live_dir = run_dir / "live"
    preview_dir = run_dir / "preview_frames"
    run_started_at = float(st.session_state.get("run_started_at", 0.0) or 0.0)
    min_mtime = run_started_at - 1.0 if run_started_at > 0 else None

    live_jpg_count = len(list(live_dir.glob("*.jpg"))) if live_dir.exists() else 0
    preview_count = len(list(preview_dir.glob("*.jpg"))) if preview_dir.exists() else 0
    st.caption(
        f"Artifacts: live_jpg={live_jpg_count}, preview_frames={preview_count}"
    )

    # Sync snapshot only when not frozen; frozen mode keeps all panes/timeline/metrics static.
    if not st.session_state.get("live_frozen", False):
        st.session_state["live_snapshot"] = _read_live_snapshot(live_dir, min_mtime=min_mtime)
    live_snapshot = st.session_state.get("live_snapshot", {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()})

    video_items = [
        ("Third Output + Phi", "window_3_third_phi.jpg"),
        ("Realtime Contribution Highlight", "window_4_realtime_highlight.jpg"),
    ]
    video_notes = {
        3: "<b>窗口3</b> 展示风险计算主视图与Phi趋势，是观察事件触发与阈值关系的核心窗口。",
        4: "<b>窗口4</b> 展示实时拥堵贡献高亮结果，用于观察关键车辆影响范围。",
    }
    for idx, (title, frame_key) in enumerate(video_items, start=3):
        st.markdown(f"<div class='window-title'>Window {idx}: {title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='window-note'>{video_notes.get(idx, '')}</div>", unsafe_allow_html=True)
        frame_bytes = live_snapshot.get("frames", {}).get(frame_key)
        frame_path = live_dir / frame_key
        if frame_bytes is None and frame_path.exists():
            try:
                if min_mtime is None or frame_path.stat().st_mtime >= min_mtime:
                    frame_bytes = frame_path.read_bytes()
            except Exception:
                frame_bytes = None
        if frame_bytes is not None:
            render_image_compat(frame_bytes)
        else:
            st.info(f"No live frame yet: {rel_text(frame_path)}")

    if "last_live_metrics" not in st.session_state:
        st.session_state["last_live_metrics"] = {}

    snap_metrics = live_snapshot.get("metrics", {})
    if isinstance(snap_metrics, dict) and len(snap_metrics) > 0:
        st.session_state["last_live_metrics"] = snap_metrics
    else:
        metrics_file = live_dir / "live_metrics.json"
        if metrics_file.exists():
            try:
                if min_mtime is None or metrics_file.stat().st_mtime >= min_mtime:
                    st.session_state["last_live_metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    m = st.session_state.get("last_live_metrics", {})
    phi_txt = _fmt_metric_float(m.get("phi_t"), digits=3)
    total_txt = _fmt_metric_int(m.get("vehicle_total"))
    parked_txt = _fmt_metric_int(m.get("parked_count"))
    speed_txt = _fmt_metric_float(m.get("avg_speed_mps"), digits=2)
    st.markdown(
        (
            "<div style='position:fixed;right:16px;bottom:16px;z-index:9999;"
            "background:rgba(10,20,35,0.92);border:1px solid rgba(120,170,230,0.45);"
            "border-radius:12px;padding:10px 12px;min-width:230px;'>"
            "<div style='font-weight:700;color:#d8e8ff;margin-bottom:6px;'>实时指标</div>"
            f"<div style='color:#9fc1ef'>综合拥堵指数: <b style='color:#ffffff'>{phi_txt}</b></div>"
            f"<div style='color:#9fc1ef'>车辆总数: <b style='color:#ffffff'>{total_txt}</b></div>"
            f"<div style='color:#9fc1ef'>停车数: <b style='color:#ffffff'>{parked_txt}</b></div>"
            f"<div style='color:#9fc1ef'>平均速度(m/s): <b style='color:#ffffff'>{speed_txt}</b></div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Phi Timeline (Realtime)")
    st.markdown(
        "<div class='section-note'><b>模块说明</b>：时间线以冷热线渐变展示综合拥堵指数随时间变化，红色区段表示更接近或超过风险阈值。</div>",
        unsafe_allow_html=True,
    )
    df_live_phi = live_snapshot.get("phi_df", pd.DataFrame())
    if not isinstance(df_live_phi, pd.DataFrame):
        df_live_phi = pd.DataFrame()
    if df_live_phi.empty:
        live_phi_csv_fallback = live_dir / "live_phi_timeline.csv"
        if live_phi_csv_fallback.exists() and (min_mtime is None or live_phi_csv_fallback.stat().st_mtime >= min_mtime):
            tmp = read_csv_safe(live_phi_csv_fallback)
            if not tmp.empty and "time_s" in tmp.columns and "phi_t" in tmp.columns:
                tmp = tmp.copy()
                tmp["time_s"] = pd.to_numeric(tmp["time_s"], errors="coerce")
                tmp["phi_t"] = pd.to_numeric(tmp["phi_t"], errors="coerce")
                tmp = tmp.dropna(subset=["time_s", "phi_t"]).sort_values("time_s")
                df_live_phi = tmp
    if not df_live_phi.empty and "time_s" in df_live_phi.columns and "phi_t" in df_live_phi.columns:
        if len(df_live_phi) >= 2:
                y_min = float(df_live_phi["phi_t"].min())
                y_max = float(df_live_phi["phi_t"].max())
                if abs(y_max - y_min) < 1e-9:
                    y_max = y_min + 1e-3

                charts_dir = run_dir / "charts"
                realtime_chart_png = charts_dir / "phi_realtime_timeline.png"
                ok_export, err_export = export_phi_timeline_png(df_live_phi, realtime_chart_png)
                if not ok_export and "matplotlib unavailable" in err_export.lower():
                    st.warning("无法导出 realtime 时间线 PNG：当前环境缺少 matplotlib。")

                # Build consecutive segments so each segment can be color-mapped while keeping a visible line.
                df_segments = df_live_phi.copy()
                df_segments["time_s_next"] = df_segments["time_s"].shift(-1)
                df_segments["phi_t_next"] = df_segments["phi_t"].shift(-1)
                df_segments["phi_color"] = (df_segments["phi_t"] + df_segments["phi_t_next"]) / 2.0
                df_segments = df_segments.dropna(subset=["time_s_next", "phi_t_next", "phi_color"])

                line = (
                    alt.Chart(df_segments)
                    .mark_rule(strokeWidth=1.0, opacity=0.9)
                    .encode(
                        x=alt.X("time_s:Q", title="time_s"),
                        x2=alt.X2("time_s_next:Q"),
                        y=alt.Y("phi_t:Q", title="phi_t", scale=alt.Scale(domain=[y_min, y_max])),
                        y2=alt.Y2("phi_t_next:Q"),
                        color=alt.Color(
                            "phi_color:Q",
                            title="phi_t",
                            scale=alt.Scale(domain=[0.0, 1.0], range=["#2E6BFF", "#FF3B30"], clamp=True),
                        ),
                    )
                )
                points = (
                    alt.Chart(df_live_phi)
                    .mark_circle(size=12)
                    .encode(
                        x=alt.X("time_s:Q", title="time_s"),
                        y=alt.Y("phi_t:Q", title="phi_t", scale=alt.Scale(domain=[y_min, y_max])),
                        color=alt.Color(
                            "phi_t:Q",
                            title="phi_t",
                            scale=alt.Scale(domain=[0.0, 1.0], range=["#2E6BFF", "#FF3B30"], clamp=True),
                        ),
                        tooltip=[
                            alt.Tooltip("time_s:Q", format=".3f"),
                            alt.Tooltip("phi_t:Q", format=".6f"),
                        ],
                    )
                )
                st.altair_chart((line + points).interactive(), use_container_width=True)
                if realtime_chart_png.exists():
                    st.caption(f"已导出图表: {rel_text(realtime_chart_png)}")
                    st.download_button(
                        "Download Realtime Timeline PNG",
                        data=realtime_chart_png.read_bytes(),
                        file_name=realtime_chart_png.name,
                        mime="image/png",
                        key="dl_realtime_timeline_png",
                    )
        elif len(df_live_phi) == 1:
            st.dataframe(df_live_phi, use_container_width=True, height=120)
        else:
            st.info("Realtime phi_t 暂无有效数据。")
    else:
        st.info("No realtime phi_t timeline yet. Please run pipeline first.")

    t1, t2, t3 = st.tabs(["Event Dashboard", "Tables", "Parameters"])

    with t1:
        st.markdown(
            "<div class='section-note'><b>Event Dashboard</b>：展示事件级摘要表。关键事件图片暂存展示已移除。</div>",
            unsafe_allow_html=True,
        )
        if len(event_dirs) == 0:
            st.info("No event outputs found yet. Run the pipeline and ensure high-Phi events exist.")
        else:
            ev = st.selectbox("Event Folder", options=event_dirs, format_func=lambda p: p.name)

            summary_csv = ev / "phi_max_and_max_intersections_summary.csv"
            if summary_csv.exists():
                st.subheader("Phi/Event Summary")
                df_summary = read_csv_safe(summary_csv)
                st.dataframe(df_summary, use_container_width=True, height=220)

    with t2:
        st.markdown(
            "<div class='section-note'><b>Tables</b>：汇总交点、车辆ID关系和分组强度，适合导出做离线分析或论文表格。</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Intersection Points Table")
        df_inter = aggregate_intersections_table(event_dirs)
        if df_inter.empty:
            st.info("No intersections parsed from event_peak_analysis.json.")
        else:
            st.dataframe(df_inter, use_container_width=True, height=280)
            st.download_button(
                "Download Intersections CSV",
                data=df_inter.to_csv(index=False).encode("utf-8"),
                file_name="intersections_all_events.csv",
                mime="text/csv",
            )

        st.subheader("Vehicle-ID Tables")
        df_vehicle = aggregate_vehicle_table(event_dirs)
        if df_vehicle.empty:
            st.info("No *_arrow_intersections_by_id.csv found.")
        else:
            st.dataframe(df_vehicle, use_container_width=True, height=360)
            st.download_button(
                "Download Vehicle Table CSV",
                data=df_vehicle.to_csv(index=False).encode("utf-8"),
                file_name="vehicle_intersections_all_events.csv",
                mime="text/csv",
            )

        st.subheader("Group Lambda Tables")
        df_group = aggregate_group_table(event_dirs)
        if df_group.empty:
            st.info("No *_group_lambda_by_k.csv found.")
        else:
            st.dataframe(df_group, use_container_width=True, height=360)
            st.download_button(
                "Download Group Lambda CSV",
                data=df_group.to_csv(index=False).encode("utf-8"),
                file_name="group_lambda_all_events.csv",
                mime="text/csv",
            )

    with t3:
        st.markdown(
            "<div class='section-note'><b>Parameters</b>：记录本次运行配置与原始日志，便于复现实验与定位问题。</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Run Parameters")
        p = st.session_state.get("last_params", {})
        if len(p) == 0:
            st.info("No run parameters yet. Execute one run from sidebar.")
        else:
            df_params = pd.DataFrame([{"parameter": k, "value": v} for k, v in p.items()])
            st.dataframe(df_params, use_container_width=True, height=360)

        with st.expander("Raw Console Output", expanded=False):
            st.code(st.session_state.get("last_log", "No run log yet."), language="text")


if __name__ == "__main__":
    main()
