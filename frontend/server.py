"""
Visual Traffic System — Web Demo Backend Server

Bridges the web frontend to the pipeline engine.
Provides REST API for:
  - Listing sites, videos, models
  - Launching/stopping pipeline runs
  - Live metrics streaming (SSE)
  - Event history browsing
"""

from __future__ import annotations

import io
import json
import os
import queue
import sys

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import subprocess
import sys
import threading
import time
import csv
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, request, send_file, Response, stream_with_context

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "intersections.json"
FRONTEND_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(FRONTEND_DIR / "static"), static_url_path="/static")

# ── Python executable detection ──────────────────────────────
_CANDIDATE_PYTHONS = [
    r"E:\Anaconda\python.exe",
    sys.executable,
    r"C:\Users\21495\AppData\Local\Programs\Python\Python312\python.exe",
    "python", "python3",
]
PYTHON_EXE = None
for _p in _CANDIDATE_PYTHONS:
    try:
        r = subprocess.run(
            [_p, "-c", "import lap._lapjv; print('ok')"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            PYTHON_EXE = _p
            break
    except Exception:
        continue
PYTHON_EXE = PYTHON_EXE or sys.executable

# ── Global state ─────────────────────────────────────────────
_pipeline_proc: subprocess.Popen | None = None
_pipeline_output_dir: str = ""
_live_metrics: dict = {"phi_t": 0, "vehicle_total": 0, "parked_count": 0, "avg_speed_mps": 0}
_event_queue: queue.Queue = queue.Queue()
_run_log: list[str] = []

PRESETS = {
    "quick": {"imgsz": 1280, "conf": 0.22, "iou": 0.40, "ablation": True, "quality": "balanced"},
    "quality": {"imgsz": 1600, "conf": 0.15, "iou": 0.35, "ablation": True, "quality": "quality"},
    "fast": {"imgsz": 960, "conf": 0.30, "iou": 0.45, "ablation": False, "quality": "fast"},
}


# ═══════════════════════════════════════════════════════════════
# API Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main demo page."""
    return send_file(str(FRONTEND_DIR / "static" / "index.html"))


@app.route("/api/sites")
def api_sites():
    """List all configured sites with metadata."""
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        sites = cfg.get("sites", {})
        result = []
        for key, site in sites.items():
            video_dir = ROOT / site.get("video_dir", "")
            videos = []
            if video_dir.exists():
                videos = sorted([
                    {"name": p.name, "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
                     "path": str(p)}
                    for p in video_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
                ], key=lambda v: v["name"])
            calib_img = site.get("calibration_image", "")
            calib_exists = (ROOT / calib_img).exists() if calib_img else False
            result.append({
                "key": key,
                "display_name": site.get("display_name", key),
                "video_count": len(videos),
                "videos": videos,
                "calibration_image": calib_img,
                "calibration_exists": calib_exists,
                "homography": site.get("homography", ""),
                "risk_params": site.get("risk_params", ""),
                "model": site.get("model", ""),
            })
        return jsonify({"sites": result, "default_site": cfg.get("default_site", "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models")
def api_models():
    """List available YOLO models."""
    models_dir = ROOT / "data" / "models"
    models = []
    if models_dir.exists():
        for p in sorted(models_dir.glob("*.pt")):
            models.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(p.stat().st_size / 1024 / 1024, 1),
            })
    return jsonify({"models": models})


@app.route("/api/presets")
def api_presets():
    """List launch presets."""
    return jsonify({"presets": {
        k: {"imgsz": v["imgsz"], "conf": v["conf"], "iou": v["iou"],
            "ablation": v["ablation"], "quality": v["quality"]}
        for k, v in PRESETS.items()
    }})


@app.route("/api/launch", methods=["POST"])
def api_launch():
    """Launch the pipeline with given parameters."""
    global _pipeline_proc, _pipeline_output_dir, _run_log, _live_metrics

    if _pipeline_proc and _pipeline_proc.poll() is None:
        return jsonify({"error": "Pipeline already running"}), 409

    data = request.get_json() or {}
    site_key = data.get("site_key", "")
    video_path = data.get("video_path", "")
    model_path = data.get("model_path", "")
    preset_key = data.get("preset", "quick")
    stride = data.get("stride", 1)
    show_windows = data.get("show_windows", True)
    max_frames = data.get("max_frames", 0)

    if not site_key or not video_path:
        return jsonify({"error": "Missing site_key or video_path"}), 400

    preset = PRESETS.get(preset_key, PRESETS["quick"])

    # Determine homography & risk params from site config
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    site_cfg = cfg.get("sites", {}).get(site_key, {})
    homography = site_cfg.get("homography", "configs/homography_points_example.json")
    risk_params = site_cfg.get("risk_params", "configs/traffic_risk_params.json")

    source_tag = Path(video_path).stem
    output_live = str(ROOT / "outputs" / site_key / source_tag / "live")
    output_events = str(ROOT / "outputs" / site_key / source_tag / "events")

    cmd = [
        PYTHON_EXE, "-u", str(ROOT / "run.py"),
        video_path,
        "--site", site_key,
        "--model", model_path or site_cfg.get("model", "data/models/yolo11m.pt"),
        "--imgsz", str(preset["imgsz"]),
        "--conf", f"{preset['conf']:.3f}",
        "--iou", f"{preset['iou']:.3f}",
        "--homography", homography,
        "--risk-params", risk_params,
        "--frame-stride", str(stride),
        "--realtime-congestion-interval", "2",
        "--live-write-interval", "2",
        "--live-dir", output_live,
        "--events-dir", output_events,
        "--async-writer",
    ]
    if show_windows:
        cmd.append("--show-windows")
    else:
        cmd.append("--no-show-windows")
    if preset["ablation"]:
        cmd.append("--ablation-enable")
    else:
        cmd.append("--no-ablation-enable")
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])

    _run_log = [f"[{time.strftime('%H:%M:%S')}] CMD: {' '.join(cmd[:8])}..."]

    try:
        # Ensure output directories exist
        log_dir = Path(output_live)
        log_dir.mkdir(parents=True, exist_ok=True)
        Path(output_events).mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / "web_run.log", "w", encoding="utf-8", errors="replace")

        _pipeline_proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        _pipeline_output_dir = output_live
        _live_metrics = {"phi_t": 0, "vehicle_total": 0, "parked_count": 0, "avg_speed_mps": 0}

        return jsonify({
            "status": "launched",
            "site_key": site_key,
            "video": Path(video_path).name,
            "preset": preset_key,
            "log_dir": str(log_dir),
        })
    except Exception as e:
        _run_log.append(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the running pipeline."""
    global _pipeline_proc, _pipeline_output_dir
    if _pipeline_proc and _pipeline_proc.poll() is None:
        _pipeline_proc.terminate()
        try:
            _pipeline_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _pipeline_proc.kill()
        _pipeline_proc = None
        _pipeline_output_dir = ""
        return jsonify({"status": "stopped"})
    _pipeline_proc = None
    _pipeline_output_dir = ""
    return jsonify({"status": "not_running"})


@app.route("/api/status")
def api_status():
    """Get pipeline status and live metrics."""
    global _pipeline_proc, _pipeline_output_dir, _live_metrics
    is_running = _pipeline_proc is not None and _pipeline_proc.poll() is None
    rc = _pipeline_proc.poll() if _pipeline_proc else None

    # Read live metrics from the pipeline's output directory
    if _pipeline_output_dir:
        metrics_file = Path(_pipeline_output_dir) / "live_metrics.json"
        if metrics_file.exists():
            try:
                _live_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    return jsonify({
        "running": is_running,
        "exit_code": rc,
        "metrics": _live_metrics,
        "log": _run_log[-50:] if _run_log else [],
    })


@app.route("/api/metrics/stream")
def api_metrics_stream():
    """SSE endpoint for live metrics streaming."""
    def generate():
        while True:
            global _pipeline_proc
            # Check if pipeline is still alive
            is_running = _pipeline_proc is not None and _pipeline_proc.poll() is None

            # Try to read live metrics from the outputs directory
            # We scan for the most recent live_metrics.json
            outputs_dir = ROOT / "outputs"
            latest_metrics = {}
            if outputs_dir.exists():
                for site_dir in sorted(outputs_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True):
                    if not site_dir.is_dir():
                        continue
                    for video_dir in sorted(site_dir.iterdir(), key=lambda p: p.stat().st_mtime if p.is_dir() else 0, reverse=True):
                        if not video_dir.is_dir():
                            continue
                        live_dir = video_dir / "live"
                        metrics_file = live_dir / "live_metrics.json"
                        if metrics_file.exists():
                            try:
                                latest_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                                # Also try to read the live screenshot as base64
                                screenshot = live_dir / "window_3_third_phi.jpg"
                                if screenshot.exists():
                                    import base64
                                    latest_metrics["screenshot"] = base64.b64encode(
                                        screenshot.read_bytes()
                                    ).decode()
                            except Exception:
                                pass
                            break
                    if latest_metrics:
                        break

            yield f"data: {json.dumps({'running': is_running, 'metrics': latest_metrics})}\n\n"
            if not is_running:
                break
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/events")
def api_events():
    """List event summaries from output directories."""
    events = []
    outputs_dir = ROOT / "outputs"
    if outputs_dir.exists():
        for site_dir in outputs_dir.iterdir():
            if not site_dir.is_dir():
                continue
            for video_dir in site_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                events_dir = video_dir / "events"
                if not events_dir.exists():
                    continue
                for ev_dir in sorted(events_dir.iterdir(), reverse=True):
                    if not ev_dir.is_dir():
                        continue
                    summary_file = ev_dir / "event_summary.json"
                    if summary_file.exists():
                        try:
                            summary = json.loads(summary_file.read_text(encoding="utf-8"))
                            events.append({
                                "site": site_dir.name,
                                "video": video_dir.name,
                                "event": ev_dir.name,
                                "peak_phi": summary.get("peak_phi", 0),
                                "duration_s": summary.get("duration_s", 0),
                                "start_time": summary.get("start_time", 0),
                            })
                        except Exception:
                            pass
    return jsonify({"events": events[:50]})


@app.route("/api/calibration-image/<site_key>")
def api_calibration_image(site_key: str):
    """Serve calibration image for a site."""
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    site = cfg.get("sites", {}).get(site_key, {})
    calib_path = site.get("calibration_image", "")
    if calib_path:
        full_path = ROOT / calib_path
        if full_path.exists():
            return send_file(str(full_path), mimetype="image/png")
    return jsonify({"error": "Not found"}), 404


@app.route("/api/architecture")
def api_architecture():
    """Return project architecture metadata for the frontend visualization."""
    return jsonify({
        "name": "Visual Recognition Intersection Information Collection System",
        "short_name": "Visual Traffic System",
        "version": "2.0",
        "pipeline_stages": [
            {"id": "detect", "name": "Detection & Tracking", "icon": "🎯",
             "description": "YOLO11m + ByteTrack, 6 vehicle classes"},
            {"id": "bev", "name": "BEV Transform", "icon": "🗺️",
             "description": "Homography warp, pixel→world coordinates"},
            {"id": "trajectory", "name": "Trajectory Management", "icon": "📈",
             "description": "EMA smoothing, deque-based trail history"},
            {"id": "motion", "name": "Motion Analysis", "icon": "🏃",
             "description": "Speed calculation, moving/stationary/parked"},
            {"id": "phi", "name": "Phi Index", "icon": "📊",
             "description": "Φ = wρ·N/Nsat + wv·(1−v/vref)"},
            {"id": "conflict", "name": "Directional Field Convolution", "icon": "🌊",
             "description": "O(G²) grid convolution, 24 conflict pairs"},
            {"id": "attribution", "name": "Vehicle Attribution", "icon": "🎯",
             "description": "Per-vehicle influence scoring"},
            {"id": "root_cause", "name": "Root Cause Tracing", "icon": "💧",
             "description": "Water Drop propagation, sparse matrix iteration"},
            {"id": "visualization", "name": "Visualization", "icon": "🎨",
             "description": "1920×1080 3-row layout, heatmap overlay"},
        ],
        "algorithm": {
            "name": "Directional Field Convolution",
            "complexity": "O(G²)",
            "grid_size": 64,
            "direction_bins": 12,
            "conflict_pairs": 24,
            "steps": [
                "Scatter vehicles to G×G grid",
                "Soft direction binning (12 bins, Gaussian weights)",
                "Anisotropic kernel construction",
                "Influence convolution (cv2.filter2D)",
                "Conflict field = Σ R_a × R_b",
                "Per-vehicle attribution scoring",
            ],
        },
        "phi_formula": "Φ = w_ρ × min(1, N/N_sat) + w_v × max(0, 1 − v_avg/v_ref)",
        "phi_levels": [
            {"range": "0.00–0.30", "label": "Free Flow", "color": "#6baf6b"},
            {"range": "0.30–0.55", "label": "Moderate", "color": "#e2b96f"},
            {"range": "0.55–0.75", "label": "Heavy", "color": "#d4956b"},
            {"range": "0.75–1.00", "label": "Severe", "color": "#d55b5b"},
        ],
        "sites_count": 5,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  Visual Traffic System -- Web Demo Server")
    print("=" * 60)
    print(f"  Python: {PYTHON_EXE}")
    print(f"  Project root: {ROOT}")
    print()
    print("  Opening http://localhost:5000 ...")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # Open browser automatically
    import webbrowser
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
