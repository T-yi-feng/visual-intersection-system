"""
Traffic Risk Smart Console v2 — 站点可视化选择 + 一键启动 + 实时监控

启动方式:
    streamlit run app/smart_console.py
    或
    python -m streamlit run app/smart_console.py
"""

import base64
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "intersections.json"

# ═══════════════════════════════════════════════════════════════
# Backend helpers (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=2)
def _list_site_videos_cached(video_dir_text: str) -> list[str]:
    p = Path(video_dir_text)
    if not p.exists():
        return []
    return sorted([
        str(x) for x in p.iterdir()
        if x.is_file() and x.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    ])


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


def source_tag(source_token: str) -> str:
    if source_token.isdigit():
        return f"camera_{source_token}"
    stem = Path(source_token).stem
    return stem if stem else "source"


def peak_dir_for_root(output_root: Path, site_key: str, source_token: str) -> Path:
    return output_root / site_key / source_tag(source_token) / "events"


def rel_text(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# Site card helpers
# ═══════════════════════════════════════════════════════════════

def _get_calib_image_base64(site_cfg: dict) -> str | None:
    """读取标定图像并返回 base64 编码"""
    calib_rel = site_cfg.get("calibration_image", "")
    if not calib_rel:
        return None
    calib_path = resolve_path(ROOT, calib_rel)
    if not calib_path.exists():
        return None
    try:
        data = calib_path.read_bytes()
        return base64.b64encode(data).decode()
    except Exception:
        return None


def _get_site_display(site_key: str, site_cfg: dict) -> dict:
    """构建站点卡片信息"""
    videos = list_site_videos(site_cfg)
    calib_b64 = _get_calib_image_base64(site_cfg)
    return {
        "key": site_key,
        "name": site_cfg.get("display_name", site_key),
        "video_count": len(videos),
        "calib_b64": calib_b64,
        "videos": videos,
    }


# ═══════════════════════════════════════════════════════════════
# Pipeline launcher (unchanged core)
# ═══════════════════════════════════════════════════════════════

def _load_site_config(site_key: str) -> dict:
    cfg_path = ROOT / "configs" / "intersections.json"
    try:
        data = json.loads(cfg_path.read_text(encoding='utf-8'))
        return data.get('sites', {}).get(site_key, {})
    except Exception:
        return {}


def _build_pipeline_cmd(
    site_key: str, source: str, model: str, imgsz: int, conf: float, iou: float,
    show_windows: int, ablation_enable: int,
    realtime_interval: int, live_interval: int,
    async_writer: int, async_queue: int,
    third_mode: str, max_frames: int, session_output_root: str,
) -> list[str]:
    site_cfg = _load_site_config(site_key)
    homography_path = site_cfg.get('homography', 'configs/homography_points_example.json')
    risk_params_path = site_cfg.get('risk_params', 'configs/traffic_risk_params.json')

    cmd = [
        sys.executable, "-u", str(ROOT / "run.py"),
        source,
        "--site-key", site_key,
        "--model", model,
        "--imgsz", str(imgsz),
        "--conf", f"{conf:.3f}",
        "--iou", f"{iou:.3f}",
        "--homography", homography_path,
        "--risk-params", risk_params_path,
        "--realtime-congestion-interval", str(int(realtime_interval)),
        "--live-write-interval", str(int(live_interval)),
        "--async-writer-queue", str(int(async_queue)),
        "--third-panel-mode", third_mode if third_mode in ("quality", "balanced", "fast") else "quality",
    ]
    if int(show_windows):
        cmd.append("--show-windows")
    if int(ablation_enable):
        cmd.append("--ablation-enable")
    if int(async_writer):
        cmd.append("--async-writer")
    if int(max_frames) > 0:
        cmd.extend(["--max-frames", str(int(max_frames))])
    return cmd


def run_pipeline_async(
    site_key: str, source: str, model: str, imgsz: int, conf: float, iou: float,
    show_windows: int, ablation_enable: int,
    realtime_interval: int, live_interval: int,
    async_writer: int, async_queue: int,
    third_mode: str, max_frames: int, session_output_root: str,
    run_log_file: Path,
) -> tuple[subprocess.Popen | None, str]:
    cmd = _build_pipeline_cmd(
        site_key=site_key, source=source, model=model, imgsz=imgsz,
        conf=conf, iou=iou, show_windows=show_windows,
        ablation_enable=ablation_enable,
        realtime_interval=realtime_interval, live_interval=live_interval,
        async_writer=async_writer, async_queue=async_queue,
        third_mode=third_mode, max_frames=max_frames,
        session_output_root=session_output_root,
    )
    try:
        run_log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(run_log_file, "w", encoding="utf-8", errors="replace", buffering=1)
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        f.close()
        return proc, ""
    except Exception as e:
        return None, str(e)


def clear_run_outputs(output_root_text: str, site_key: str, source: str) -> tuple[bool, str]:
    try:
        output_root = resolve_path(ROOT, output_root_text)
        run_dir = output_root / site_key / source_tag(source)
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        return True, ""
    except Exception as e:
        return False, str(e)


def launch_quick_calibration(site_key: str, top_m=None, right_m=None, bottom_m=None, left_m=None) -> tuple[bool, str]:
    script = ROOT / "tools" / "calibrate_homography.py"
    if not script.exists():
        return False, f"Missing script: {script}"
    cmd = [sys.executable, str(script), "--site", site_key]
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


def _normalize_windows_return_code(rc: int) -> int:
    try:
        v = int(rc)
    except Exception:
        return rc
    if v >= (1 << 31):
        return v - (1 << 32)
    return v


# ═══════════════════════════════════════════════════════════════
# Live monitoring helpers
# ═══════════════════════════════════════════════════════════════

def _read_live_snapshot(live_dir: Path, min_mtime: float | None = None) -> dict:
    snapshot = {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()}
    frame_map = {"window_3_third_phi.jpg": live_dir / "window_3_third_phi.jpg"}
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
                raise RuntimeError("stale")
            snapshot["metrics"] = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    live_phi_csv = live_dir / "live_phi_timeline.csv"
    if live_phi_csv.exists():
        df_tmp = read_csv_safe(live_phi_csv)
        if not df_tmp.empty and "time_s" in df_tmp.columns and "phi_t" in df_tmp.columns:
            df_tmp = df_tmp.copy()
            df_tmp["time_s"] = pd.to_numeric(df_tmp["time_s"], errors="coerce")
            df_tmp["phi_t"] = pd.to_numeric(df_tmp["phi_t"], errors="coerce")
            df_tmp = df_tmp.dropna(subset=["time_s", "phi_t"]).sort_values("time_s")
            snapshot["phi_df"] = df_tmp
    return snapshot


def collect_event_dirs(peak_dir: Path) -> list[Path]:
    if not peak_dir.exists():
        return []
    return sorted([p for p in peak_dir.iterdir() if p.is_dir() and p.name.startswith("event_")], reverse=True)


def export_phi_timeline_png(df_live_phi: pd.DataFrame, out_path: Path) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        mpl.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False
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
    lc.set_array(seg_color); lc.set_linewidth(1.0); lc.set_alpha(0.95)
    ax.add_collection(lc)
    sc = ax.scatter(x, y, c=np.clip(y, 0.0, 1.0), cmap=cmap, norm=norm, s=12, zorder=3)
    fig.colorbar(sc, ax=ax).set_label("Phi")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if abs(x_max - x_min) < 1e-9: x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-9: y_max = y_min + 1e-3
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min - 0.02, y_max + 0.02)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Phi")
    ax.grid(True, alpha=0.25); fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return True, ""


def _fmt_float(v, d=3): return f"{float(v):.{d}f}" if v is not None else "--"
def _fmt_int(v): return str(int(v)) if v is not None else "--"


# ═══════════════════════════════════════════════════════════════
# Session state init
# ═══════════════════════════════════════════════════════════════

PRESETS = {
    "🚀 Quick Start (balanced)": {"imgsz": 1280, "conf": 0.22, "iou": 0.40, "ablation": 1, "third": "balanced", "refresh": 2},
    "🎯 High Quality (paper)": {"imgsz": 1600, "conf": 0.15, "iou": 0.35, "ablation": 1, "third": "quality", "refresh": 1},
    "⚡ Fast Preview (low-res)": {"imgsz": 960, "conf": 0.30, "iou": 0.45, "ablation": 0, "third": "fast", "refresh": 4},
}


def ensure_session():
    defaults = {
        "selected_site_key": "", "selected_video_path": "", "selected_video_name": "",
        "mode": "quick",  # quick | advanced
        "preset_key": "🚀 Quick Start (balanced)",
        "run_proc": None, "run_log_file": "", "last_run_rc": None,
        "run_started_at": 0.0, "has_run": False,
        "live_frozen": False, "live_snapshot": {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()},
        "last_params": {}, "last_metrics": {}, "last_log": "",
        "show_calib": False, "session_output_root": "outputs/unified",
        "run_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# CSS — Production-grade dark theme
# ═══════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b0e11; --surface: #14171b; --surface2: #1a1e23;
    --border: #2a2e35; --border-active: #4a5568;
    --text: #e8eaed; --text-muted: #8b9199; --text-dim: #5a5f66;
    --blue: #5b9bd5; --blue-bg: rgba(91,155,213,0.12);
    --green: #6baf6b; --green-bg: rgba(107,175,107,0.12);
    --orange: #d4956b; --orange-bg: rgba(212,149,107,0.12);
    --red: #d55b5b; --red-bg: rgba(213,91,91,0.12);
    --accent: #e2b96f;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Buttons */
.stButton > button {
    border-radius: 8px; border: 1px solid var(--border);
    background: var(--surface2); color: var(--text);
    font-weight: 600; font-size: 0.9rem;
    transition: all 0.15s ease;
    padding: 0.5rem 1rem;
}
.stButton > button:hover {
    border-color: var(--border-active);
    background: #20252b;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #5b9bd5, #3d7ec5);
    border-color: #5b9bd5; color: #fff; font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6babf0, #4d8ed5);
}

/* Select box */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Cards */
.site-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 0; overflow: hidden;
    cursor: pointer; transition: all 0.2s ease;
}
.site-card:hover { border-color: var(--blue); transform: translateY(-2px); box-shadow: 0 4px 20px rgba(91,155,213,0.15); }
.site-card.selected { border: 2px solid var(--blue); box-shadow: 0 0 0 3px rgba(91,155,213,0.15); }
.site-card .card-img { width: 100%; height: 140px; object-fit: cover; background: var(--surface2); }
.site-card .card-body { padding: 12px 14px; }
.site-card .card-title { font-weight: 700; font-size: 0.95rem; color: var(--text); }
.site-card .card-meta { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }
.site-card .card-badge {
    display: inline-block; background: var(--blue-bg); color: var(--blue);
    border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; font-weight: 600;
}

/* Preset chips */
.preset-chip {
    display: inline-block; padding: 8px 16px; border-radius: 20px;
    border: 1px solid var(--border); background: var(--surface);
    cursor: pointer; font-size: 0.85rem; font-weight: 500;
    transition: all 0.15s ease; margin: 4px;
}
.preset-chip:hover { border-color: var(--accent); }
.preset-chip.active { border-color: var(--accent); background: rgba(226,185,111,0.12); color: var(--accent); font-weight: 700; }

/* Status pill */
.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem;
}
.status-pill.running { background: rgba(107,175,107,0.15); color: var(--green); }
.status-pill.idle { background: var(--surface2); color: var(--text-muted); }
.status-pill.error { background: var(--red-bg); color: var(--red); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-dot.running { background: var(--green); animation: pulse 1.5s infinite; }
.status-dot.idle { background: var(--text-dim); }
.status-dot.error { background: var(--red); }

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Metric tile */
.metric-tile {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 800; color: var(--text); }
.metric-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }

/* Section title */
.section-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 1.2rem 0 0.6rem; }
.section-desc { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.8rem; }

/* Expander override */
.streamlit-expanderHeader { font-weight: 600; }

/* Number input */
.stNumberInput input { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; }
.stSlider > div > div > div { background: var(--blue) !important; }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Traffic Intersection Monitor", page_icon="🚦", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_session()

    cfg = load_intersections(CONFIG_PATH)
    sites = cfg["sites"]

    # ── Header ──
    c_title, c_status = st.columns([3, 1])
    with c_title:
        st.markdown("## 🚦 Traffic Intersection Analysis Console")
        st.caption("Visual site selection · One-click pipeline launch · Real-time Phi monitoring")
    with c_status:
        _render_status_pill()

    # ── Tab: Launch | Monitor | History ──
    tab_launch, tab_monitor, tab_history = st.tabs(["🚀 Launch", "📊 Live Monitor", "📁 History"])

    # ═══════════════════ LAUNCH TAB ═══════════════════
    with tab_launch:
        _render_launch_tab(sites)

    # ═══════════════════ MONITOR TAB ═══════════════════
    with tab_monitor:
        _render_monitor_tab()

    # ═══════════════════ HISTORY TAB ═══════════════════
    with tab_history:
        _render_history_tab()


# ═══════════════════════════════════════════════════════════════
# Launch Tab
# ═══════════════════════════════════════════════════════════════

def _render_launch_tab(sites):
    # ── Step 1: Select site (visual cards) ──
    st.markdown('<div class="section-title">Step 1 — Select Intersection</div>', unsafe_allow_html=True)

    site_infos = {k: _get_site_display(k, v) for k, v in sites.items()}
    cols = st.columns(min(len(site_infos), 5))
    for i, (sk, info) in enumerate(site_infos.items()):
        with cols[i % len(cols)]:
            selected = st.session_state["selected_site_key"] == sk
            card_cls = "site-card selected" if selected else "site-card"

            # Card with calibration image
            card_html = f'<div class="{card_cls}">'
            if info["calib_b64"]:
                card_html += f'<img class="card-img" src="data:image/jpeg;base64,{info["calib_b64"]}" alt="{sk}"/>'
            else:
                card_html += f'<div class="card-img" style="display:flex;align-items:center;justify-content:center;color:var(--text-dim);font-size:2rem;">🚦</div>'
            card_html += f'<div class="card-body">'
            card_html += f'<div class="card-title">{info["name"]}</div>'
            card_html += f'<div class="card-meta">{info["video_count"]} video(s) available</div>'
            card_html += f'<span class="card-badge">{sk}</span>'
            card_html += '</div></div>'
            st.markdown(card_html, unsafe_allow_html=True)

            # Hidden button for click behavior
            if st.button(f"Select {sk}", key=f"sel_{sk}", use_container_width=True,
                         type="primary" if not selected else "secondary"):
                st.session_state["selected_site_key"] = sk
                st.session_state["selected_video_path"] = ""
                st.session_state["selected_video_name"] = ""
                st.rerun()

    # ── Step 2: Select video ──
    site_key = st.session_state["selected_site_key"]
    if site_key and site_key in sites:
        st.markdown('<div class="section-title">Step 2 — Select Video</div>', unsafe_allow_html=True)
        info = site_infos[site_key]
        videos = info["videos"]

        if videos:
            vcols = st.columns(min(len(videos), 4))
            for i, vp in enumerate(videos):
                with vcols[i % len(vcols)]:
                    vselected = st.session_state["selected_video_path"] == str(vp)
                    vcard_cls = "site-card selected" if vselected else "site-card"
                    st.markdown(
                        f'<div class="{vcard_cls}"><div class="card-body">'
                        f'<div class="card-title" style="font-size:0.85rem;">🎬 {vp.name}</div>'
                        f'<div class="card-meta">{(vp.stat().st_size / 1024 / 1024):.1f} MB</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Pick {vp.name}", key=f"vid_{i}", use_container_width=True):
                        st.session_state["selected_video_path"] = str(vp)
                        st.session_state["selected_video_name"] = vp.name
                        st.rerun()
        else:
            st.warning("No videos found for this site.")

    # ── Step 3: Preset & params ──
    if st.session_state["selected_video_path"]:
        st.markdown('<div class="section-title">Step 3 — Configure & Launch</div>', unsafe_allow_html=True)

        # Preset chips
        presets = list(PRESETS.keys())
        ccols = st.columns(len(presets))
        for i, pk in enumerate(presets):
            with ccols[i]:
                is_active = st.session_state["preset_key"] == pk
                if st.button(pk, key=f"preset_{i}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state["preset_key"] = pk
                    st.rerun()

        preset = PRESETS[st.session_state["preset_key"]]

        with st.expander("⚙️ Advanced Settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                model = st.text_input("Model", value="yolo11m.pt")
                imgsz = st.number_input("imgsz", value=preset["imgsz"], min_value=640, max_value=1920, step=32)
                conf = st.slider("conf", 0.05, 0.90, value=preset["conf"], step=0.01)
            with c2:
                iou = st.slider("iou", 0.05, 0.90, value=preset["iou"], step=0.01)
                ablation = st.selectbox("Ablation", [1, 0], index=1 if preset["ablation"] else 1, format_func=lambda x: "ON" if x==1 else "OFF")
                third_mode = st.selectbox("Render Quality", ["quality","balanced","fast"], index=["quality","balanced","fast"].index(preset["third"]))
            with c3:
                refresh = st.number_input("Refresh Interval", value=preset["refresh"], min_value=1, max_value=12)
                live_write = st.number_input("Live Write Interval", value=preset["refresh"], min_value=1, max_value=12)
                max_frames = st.number_input("Max Frames (0=all)", value=0, min_value=0, max_value=200000, step=60)
                show_win = st.selectbox("Video Windows", [0, 1], index=0, format_func=lambda x: "ON" if x==1 else "OFF")
                async_w = st.selectbox("Async Writer", [1, 0], index=0, format_func=lambda x: "ON" if x==1 else "OFF")

        # Launch button
        can_launch = bool(site_key) and bool(st.session_state["selected_video_path"])
        run_proc = st.session_state.get("run_proc")
        is_running = run_proc is not None and run_proc.poll() is None

        if is_running:
            st.info("⏳ Pipeline is currently running... Switch to **Live Monitor** tab to watch progress.")
            if st.button("⏹ Stop Pipeline", type="secondary"):
                try:
                    run_proc.terminate()
                except Exception:
                    pass
                st.session_state["run_proc"] = None
                st.rerun()
        else:
            if st.button("▶  Launch Pipeline", type="primary", use_container_width=True, disabled=not can_launch):
                source = st.session_state["selected_video_path"]
                session_root = st.session_state["session_output_root"]
                active_dir = resolve_path(ROOT, session_root) / site_key / source_tag(source)
                log_file = active_dir / "live" / "web_run.log"

                clear_run_outputs(session_root, site_key, source)

                proc, err = run_pipeline_async(
                    site_key=site_key, source=source,
                    model=model, imgsz=int(imgsz), conf=float(conf), iou=float(iou),
                    show_windows=int(show_win), ablation_enable=int(ablation),
                    realtime_interval=int(refresh), live_interval=int(live_write),
                    async_writer=int(async_w), async_queue=2,
                    third_mode=third_mode, max_frames=int(max_frames),
                    session_output_root=session_root, run_log_file=log_file,
                )
                if proc is None:
                    st.error(f"Failed to start: {err}")
                else:
                    st.session_state["run_proc"] = proc
                    st.session_state["run_log_file"] = str(log_file)
                    st.session_state["run_started_at"] = time.time()
                    st.session_state["has_run"] = True
                    st.session_state["live_snapshot"] = {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()}
                    st.session_state["last_params"] = {"site": site_key, "source": source, "preset": st.session_state["preset_key"]}
                    st.session_state["active_site_key"] = site_key
                    st.session_state["active_source_value"] = source
                    # Add to history
                    st.session_state["run_history"].insert(0, {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "site": site_key,
                        "video": Path(source).name,
                        "preset": st.session_state["preset_key"],
                        "status": "started",
                    })
                    st.success("✅ Pipeline launched! Switch to **Live Monitor** tab.")
                    time.sleep(0.5)
                    st.rerun()


# ═══════════════════════════════════════════════════════════════
# Monitor Tab
# ═══════════════════════════════════════════════════════════════

def _render_monitor_tab():
    if not st.session_state.get("has_run", False):
        st.info("👈 No pipeline has been launched yet. Go to the **Launch** tab to start one.")
        return

    run_proc = st.session_state.get("run_proc")
    is_running = run_proc is not None and run_proc.poll() is None
    frozen = st.session_state.get("live_frozen", False)

    # Control bar
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    with c1:
        status = "running" if is_running else "idle"
        dot = "running" if is_running else "idle"
        st.markdown(f'<div class="status-pill {status}"><span class="status-dot {dot}"></span>{"Running" if is_running else "Idle"}</div>', unsafe_allow_html=True)
    with c2:
        if st.button("🧊 Freeze" if not frozen else "▶ Resume", use_container_width=True):
            st.session_state["live_frozen"] = not frozen
            st.rerun()
    with c3:
        st.caption(f"Site: **{st.session_state.get('active_site_key','?')}**")
    with c4:
        st.caption(f"Video: **{st.session_state.get('active_source_value','?')}**")

    if is_running:
        st_autorefresh(interval=1000, key="monitor_refresh")

    # Read live data
    site_key = st.session_state.get("active_site_key", "")
    source = st.session_state.get("active_source_value", "")
    if site_key and source:
        session_root = st.session_state["session_output_root"]
        live_dir = resolve_path(ROOT, session_root) / site_key / source_tag(source) / "live"
        run_started = float(st.session_state.get("run_started_at", 0) or 0)
        min_mtime = run_started - 1.0 if run_started > 0 else None

        if not frozen:
            st.session_state["live_snapshot"] = _read_live_snapshot(live_dir, min_mtime=min_mtime)

    snap = st.session_state.get("live_snapshot", {"frames": {}, "metrics": {}, "phi_df": pd.DataFrame()})

    # Metrics row
    m = snap.get("metrics", {})
    if m:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            phi_val = float(m.get("phi_t", 0))
            phi_color = "#6baf6b" if phi_val < 0.3 else ("#e2b96f" if phi_val < 0.55 else ("#d4956b" if phi_val < 0.75 else "#d55b5b"))
            st.markdown(f'<div class="metric-tile"><div class="metric-value" style="color:{phi_color}">{phi_val:.4f}</div><div class="metric-label">Phi Index</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{_fmt_int(m.get("vehicle_total"))}</div><div class="metric-label">Vehicles</div></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{_fmt_float(m.get("avg_speed_mps"),2)}</div><div class="metric-label">Avg Speed (m/s)</div></div>', unsafe_allow_html=True)
        with mc4:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{_fmt_int(m.get("parked_count"))}</div><div class="metric-label">Parked</div></div>', unsafe_allow_html=True)

    # Live preview image
    st.markdown('<div class="section-title">Live Output</div>', unsafe_allow_html=True)
    frame_bytes = snap.get("frames", {}).get("window_3_third_phi.jpg")
    if frame_bytes:
        st.image(frame_bytes, caption="Real-time Composite View (960×540)", use_container_width=True)
    else:
        img_path = live_dir / "window_3_third_phi.jpg" if site_key else None
        if img_path and img_path.exists():
            st.image(img_path.read_bytes(), caption="Cached Output", use_container_width=True)
        else:
            st.info("Waiting for first output frame... (may take a few seconds)")

    # Phi timeline
    st.markdown('<div class="section-title">Phi Timeline</div>', unsafe_allow_html=True)
    df = snap.get("phi_df", pd.DataFrame())
    if isinstance(df, pd.DataFrame) and not df.empty:
        y_min, y_max = float(df["phi_t"].min()), float(df["phi_t"].max())
        if abs(y_max - y_min) < 1e-9: y_max = y_min + 1e-3
        df_seg = df.copy()
        df_seg["t2"] = df_seg["time_s"].shift(-1); df_seg["p2"] = df_seg["phi_t"].shift(-1)
        df_seg["pc"] = (df_seg["phi_t"] + df_seg["p2"]) / 2.0
        df_seg = df_seg.dropna(subset=["t2","p2","pc"])
        chart = (
            alt.Chart(df_seg).mark_rule(strokeWidth=1.2, opacity=0.85)
            .encode(x=alt.X("time_s:Q", title="Time (s)"), x2=alt.X2("t2:Q"),
                    y=alt.Y("phi_t:Q", title="Phi", scale=alt.Scale(domain=[y_min, y_max])),
                    y2=alt.Y2("p2:Q"),
                    color=alt.Color("pc:Q", scale=alt.Scale(domain=[0,1], range=["#2E6BFF","#FF3B30"]), legend=None))
            + alt.Chart(df).mark_circle(size=12).encode(x="time_s:Q", y="phi_t:Q",
                  color=alt.Color("phi_t:Q", scale=alt.Scale(domain=[0,1], range=["#2E6BFF","#FF3B30"]), legend=None),
                  tooltip=[alt.Tooltip("time_s:Q",format=".3f"), alt.Tooltip("phi_t:Q",format=".4f")])
        )
        st.altair_chart(chart.interactive(), use_container_width=True)
    else:
        st.info("No Phi timeline data yet.")

    # Event summary
    if site_key and source:
        peak_dir = peak_dir_for_root(resolve_path(ROOT, st.session_state["session_output_root"]), site_key, source)
        ev_dirs = collect_event_dirs(peak_dir)
        if ev_dirs:
            st.markdown(f'<div class="section-title">Events Detected ({len(ev_dirs)})</div>', unsafe_allow_html=True)
            for ev in ev_dirs[:5]:
                st.caption(f"📌 {ev.name}")
        else:
            st.caption("No congestion events detected yet.")


# ═══════════════════════════════════════════════════════════════
# History Tab
# ═══════════════════════════════════════════════════════════════

def _render_history_tab():
    history = st.session_state.get("run_history", [])
    if not history:
        st.info("No runs yet. Launch a pipeline from the **Launch** tab.")
        return
    for h in history:
        st.markdown(
            f'<div class="site-card" style="padding:10px 16px;margin:6px 0;">'
            f'<span style="font-weight:600;">{h["time"]}</span> — '
            f'Site: <b>{h["site"]}</b> | Video: {h["video"]} | '
            f'Preset: {h["preset"]} | Status: {h["status"]}</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# Status pill helper
# ═══════════════════════════════════════════════════════════════

def _render_status_pill():
    run_proc = st.session_state.get("run_proc")
    is_running = run_proc is not None and run_proc.poll() is None
    rc = st.session_state.get("last_run_rc")

    if is_running:
        cls, dot, label = "running", "running", "Pipeline Running"
    elif rc is not None and rc != 0:
        cls, dot, label = "error", "error", f"Error (code={rc})"
    elif st.session_state.get("has_run", False):
        cls, dot, label = "idle", "idle", "Ready"
    else:
        cls, dot, label = "idle", "idle", "No runs yet"

    st.markdown(
        f'<div class="status-pill {cls}"><span class="status-dot {dot}"></span>{label}</div>'
        f'<div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">'
        f'Site: {st.session_state.get("active_site_key","—")} | '
        f'{st.session_state.get("active_source_value","—")}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
