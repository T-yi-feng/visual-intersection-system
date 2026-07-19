"""
Traffic Intersection Analysis — Native Desktop Launcher
零额外依赖 (tkinter + PIL 内置)，双击即用。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "intersections.json"
SETTINGS_PATH = Path.home() / ".visual_traffic_settings.json"

# ── Detect the correct Python (must be able to import lap) ──
_CANDIDATE_PYTHONS = [
    r"E:\Anaconda\python.exe",                # Anaconda (known working)
    sys.executable,                           # current
    r"C:\Users\21495\AppData\Local\Programs\Python\Python312\python.exe",
    r"C:\Users\21495\AppData\Local\Programs\Python\Python311\python.exe",
    "python", "python3",
]
PYTHON_EXE = None
for _p in _CANDIDATE_PYTHONS:
    try:
        # Must pass both lap AND its C extension lap._lapjv
        r = subprocess.run(
            [_p, "-c", "import lap._lapjv; print('ok')"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            PYTHON_EXE = _p
            break
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        continue

if PYTHON_EXE is None:
    PYTHON_EXE = r"E:\Anaconda\python.exe"  # known-good fallback

with open(CONFIG_PATH, encoding="utf-8") as f:
    _raw = json.load(f)
SITES = _raw["sites"]
DEFAULT_SITE = _raw.get("default_site", list(SITES.keys())[0])

PRESETS = {
    "Quick Start (balanced)":    {"imgsz": 1280, "conf": 0.22, "ablation": 1, "quality": "balanced"},
    "High Quality (paper)":      {"imgsz": 1600, "conf": 0.15, "ablation": 1, "quality": "quality"},
    "Fast Preview (low-res)":    {"imgsz": 960,  "conf": 0.30, "ablation": 0, "quality": "fast"},
}
STRIDE_OPTIONS = [1, 2, 3, 5, 8, 10]

THEMES = {
    "Dark (default)":   {"bg":"#0f1117","surface":"#14171b","text":"#e8eaed","muted":"#8b9199","dim":"#5a5f66","accent":"#5b9bd5","green":"#6baf6b","orange":"#e2b96f","red":"#d55b5b"},
    "Midnight Blue":    {"bg":"#0a0e1a","surface":"#111827","text":"#e0e4f0","muted":"#8892a8","dim":"#5a6078","accent":"#6b9bd5","green":"#5baf7b","orange":"#d4a56b","red":"#d55b5b"},
    "Slate":            {"bg":"#1a1d23","surface":"#22262e","text":"#e2e4e8","muted":"#999da4","dim":"#6b6f76","accent":"#6ea8d5","green":"#6ea86e","orange":"#d4a06b","red":"#d56b6b"},
}


# ═══════════ Settings persistence ═══════════

def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_settings(data: dict):
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ═══════════ Pipeline ═══════════

def scan_videos(site_key: str) -> list[Path]:
    site = SITES.get(site_key, {})
    vd = ROOT / site.get("video_dir", "")
    return sorted([p for p in vd.iterdir() if p.is_file() and p.suffix.lower() in {".mp4",".avi",".mov",".mkv",".wmv"}]) if vd.exists() else []


def launch_pipeline(site_key, video_path, preset, model_path, stride):
    site = SITES[site_key]
    cmd = [
        PYTHON_EXE, "-u", str(ROOT / "run.py"),
        str(video_path), "--site", site_key, "--model", model_path,
        "--imgsz", str(preset["imgsz"]), "--conf", f"{preset['conf']:.3f}", "--iou", "0.40",
        "--homography", site.get("homography","configs/homography_points_example.json"),
        "--risk-params", site.get("risk_params","configs/traffic_risk_params.json"),
        "--show-windows", "--frame-stride", str(stride),
        "--ablation-enable" if preset["ablation"] else "--no-ablation-enable",
        "--third-panel-mode", preset["quality"],
    ]
    log_dir = ROOT / "outputs" / site_key / video_path.stem / "live"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "launcher.log", "w", encoding="utf-8", errors="replace")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)


# ═══════════ UI ═══════════

class LauncherApp:
    def __init__(self):
        self.settings = load_settings()
        theme_name = self.settings.get("theme", "Dark (default)")
        self.theme = THEMES.get(theme_name, THEMES["Dark (default)"])

        self.root = tk.Tk()
        self.root.title("Traffic Intersection Analysis — Launcher")
        self.root.geometry("920x740")
        self.root.minsize(860, 680)

        self._bg_image: ImageTk.PhotoImage | None = None
        self._bg_label: tk.Label | None = None

        self.site_var   = tk.StringVar(value=DEFAULT_SITE)
        self.video_var  = tk.StringVar(value="")
        self.model_var  = tk.StringVar(value="")
        self.preset_var = tk.StringVar(value=list(PRESETS.keys())[0])
        self.stride_var = tk.IntVar(value=1)
        self.video_paths: dict[str, Path] = {}
        self._preview_id = None
        self._anim_job   = None

        self._apply_theme()
        self._build_ui()
        self._on_site_change(animate=False)

    # ── Theme ────────────────────────────────────

    def _apply_theme(self):
        T = self.theme
        self.root.configure(bg=T["bg"])
        bg_path = self.settings.get("bg_image", "")
        if bg_path and Path(bg_path).exists():
            try:
                img = Image.open(bg_path)
                self._bg_image = ImageTk.PhotoImage(img)
                if self._bg_label:
                    self._bg_label.destroy()
                self._bg_label = tk.Label(self.root, image=self._bg_image)
                self._bg_label.place(x=0, y=0, relwidth=1, relheight=1)
                self._bg_label.lower()
            except Exception:
                pass

    def _build_ui(self):
        T = self.theme

        header = tk.Frame(self.root, bg=T["bg"])
        header.pack(fill=tk.X, padx=24, pady=(16, 0))
        tk.Label(header, text="Traffic Intersection Analysis",
                 font=("Segoe UI", 18, "bold"), fg=T["text"], bg=T["bg"]).pack(side=tk.LEFT)
        tk.Button(header, text="Settings", font=("Segoe UI", 9),
                  bg=T["surface"], fg=T["muted"], activebackground=T["surface"],
                  activeforeground=T["text"], relief=tk.FLAT, padx=12, pady=4,
                  cursor="hand2", command=self._open_settings,
                  ).pack(side=tk.RIGHT)

        tk.Label(self.root, text="Select a site, pick a video, and launch.",
                 font=("Segoe UI", 10), fg=T["muted"], bg=T["bg"]
                 ).pack(anchor=tk.W, padx=24, pady=(2, 12))

        main = tk.Frame(self.root, bg=T["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 8))

        # ── Left ──
        left = tk.Frame(main, bg=T["surface"], width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Intersection", font=("Segoe UI", 11, "bold"),
                 fg=T["text"], bg=T["surface"]).pack(anchor=tk.W, padx=14, pady=(12, 6))
        self.site_frame = tk.Frame(left, bg=T["surface"])
        self.site_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # ── Right ──
        right = tk.Frame(main, bg=T["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(
            right, bg=T["surface"], height=250,
            highlightthickness=0, bd=0,
        )
        self.preview_canvas.pack(fill=tk.X, pady=(0, 8))
        self._preview_placeholder()

        # Scrollable controls area
        ctrl_canvas = tk.Canvas(right, bg=T["bg"], highlightthickness=0, bd=0)
        ctrl_scroll = tk.Frame(ctrl_canvas, bg=T["bg"])
        scrollbar = tk.Scrollbar(right, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # scrollbar not packed → 完全不可见，仍支持鼠标滚轮滚动
        ctrl_canvas.create_window((0, 0), window=ctrl_scroll, anchor=tk.NW, tags="inner")
        ctrl_scroll.bind("<Configure>",
                         lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        def _on_mw(event):
            ctrl_canvas.yview_scroll(int(-1 * event.delta / 120), "units")
        ctrl_canvas.bind_all("<MouseWheel>", _on_mw)
        ctrl = ctrl_scroll

        # Video
        tk.Label(ctrl, text="Video", font=("Segoe UI", 10, "bold"),
                 fg=T["text"], bg=T["bg"]).pack(anchor=tk.W)
        self.video_frame = tk.Frame(ctrl, bg=T["bg"])
        self.video_frame.pack(fill=tk.X, pady=(0, 6))

        # Model + Stride
        row2 = tk.Frame(ctrl, bg=T["bg"])
        row2.pack(fill=tk.X, pady=(2, 4))
        mcol = tk.Frame(row2, bg=T["bg"])
        mcol.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        scol = tk.Frame(row2, bg=T["bg"], width=150)
        scol.pack(side=tk.RIGHT, fill=tk.Y)
        scol.pack_propagate(False)

        tk.Label(mcol, text="Model", font=("Segoe UI", 10, "bold"),
                 fg=T["text"], bg=T["bg"]).pack(anchor=tk.W)
        self.model_frame = tk.Frame(mcol, bg=T["bg"])
        self.model_frame.pack(fill=tk.X)

        tk.Label(scol, text="Stride", font=("Segoe UI", 10, "bold"),
                 fg=T["text"], bg=T["bg"]).pack(anchor=tk.W)
        self.stride_frame = tk.Frame(scol, bg=T["bg"])
        self.stride_frame.pack(fill=tk.X)

        # Preset
        tk.Label(ctrl, text="Preset", font=("Segoe UI", 10, "bold"),
                 fg=T["text"], bg=T["bg"]).pack(anchor=tk.W, pady=(6, 2))
        self.preset_frame = tk.Frame(ctrl, bg=T["bg"])
        self.preset_frame.pack(fill=tk.X, pady=(0, 6))

        # Launch
        # Button bar (fixed at bottom, outside scrollable area)
        btn_bar = tk.Frame(right, bg=T["bg"])
        btn_bar.pack(fill=tk.X, side=tk.BOTTOM)

        calib_row = tk.Frame(btn_bar, bg=T["bg"])
        calib_row.pack(fill=tk.X)
        tk.Button(calib_row, text="⚙ Calibration",
                  font=("Segoe UI", 9), bg=T["surface"], fg=T["muted"],
                  activebackground="#1a1e23", activeforeground=T["text"],
                  relief=tk.FLAT, padx=8, pady=4, cursor="hand2",
                  command=self._on_calib,
                  ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(calib_row, text="Set 4 corner points",
                 font=("Segoe UI", 7), fg=T["dim"], bg=T["bg"]).pack(side=tk.LEFT)

        self.launch_btn = tk.Button(
            btn_bar, text="▶  Launch Pipeline",
            font=("Segoe UI", 14, "bold"),
            bg=T["accent"], fg="#ffffff", activebackground="#4d8ed5",
            activeforeground="#ffffff", relief=tk.FLAT,
            padx=16, pady=10, cursor="hand2",
            command=self._on_launch,
        )
        self.launch_btn.pack(fill=tk.X, pady=(4, 0))

        self.status_label = tk.Label(
            btn_bar, text="Ready", font=("Segoe UI", 9), fg=T["dim"], bg=T["bg"],
        )
        self.status_label.pack()

        bottom = tk.Frame(self.root, bg=T["bg"])
        bottom.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=(0, 8))
        tk.Label(bottom, text="Python: " + PYTHON_EXE,
                 font=("Segoe UI", 8), fg=T["dim"], bg=T["bg"]).pack(side=tk.LEFT)

    # ── Preview ──────────────────────────────────

    def _preview_placeholder(self):
        T = self.theme
        self.preview_canvas.delete("all")
        cw = self.preview_canvas.winfo_width() or 600
        ch = self.preview_canvas.winfo_height() or 250
        self.preview_canvas.create_text(
            cw // 2, ch // 2,
            text="Select an intersection\nto preview calibration image",
            font=("Segoe UI", 11), fill=T["dim"], justify=tk.CENTER,
        )

    def _show_preview(self, site_key: str):
        """Simple slide-in from right — no fade, clean and fast."""
        if self._anim_job:
            self.root.after_cancel(self._anim_job)

        T = self.theme
        cw = self.preview_canvas.winfo_width() or 600
        ch = self.preview_canvas.winfo_height() or 250

        site = SITES.get(site_key, {})
        calib_rel = site.get("calibration_image", "")
        img = None
        if calib_rel:
            path = ROOT / calib_rel
            if path.exists():
                try:
                    pil = Image.open(path)
                    pil.thumbnail((cw - 20, ch - 20), Image.LANCZOS)
                    img = ImageTk.PhotoImage(pil)
                except Exception:
                    pass

        # Clear canvas
        self.preview_canvas.delete("all")

        if img:
            new_id = self.preview_canvas.create_image(
                cw + 200, ch // 2, image=img, anchor=tk.CENTER)
            # Keep reference
            self._preview_tk = img
            self._slide_in(new_id, cw // 2, ch // 2)
        else:
            self.preview_canvas.create_text(
                cw // 2, ch // 2,
                text="No calibration image\nfor this site",
                font=("Segoe UI", 11), fill=T["dim"], justify=tk.CENTER)

    def _slide_in(self, item_id, tx, ty):
        """Animate item sliding from current x to target x."""
        cx, cy = self.preview_canvas.coords(item_id)
        dx = (tx - cx) / 5
        if abs(dx) < 0.5:
            self.preview_canvas.coords(item_id, tx, ty)
            return
        self.preview_canvas.coords(item_id, cx + dx, ty)
        self._anim_job = self.root.after(16, lambda: self._slide_in(item_id, tx, ty))

    # ── Settings ─────────────────────────────────

    def _open_settings(self):
        T = self.theme
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.geometry("400x350")
        dlg.configure(bg=T["bg"])
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="Settings", font=("Segoe UI", 14, "bold"),
                 fg=T["text"], bg=T["bg"]).pack(pady=(16, 10))

        # Theme
        tk.Label(dlg, text="Color Theme", font=("Segoe UI", 10),
                 fg=T["muted"], bg=T["bg"]).pack(anchor=tk.W, padx=24)
        theme_var = tk.StringVar(value=self.settings.get("theme", "Dark (default)"))
        for name in THEMES:
            tk.Radiobutton(
                dlg, text=name, variable=theme_var, value=name,
                font=("Segoe UI", 9), fg=T["muted"], bg=T["bg"],
                selectcolor=T["bg"], activebackground=T["surface"],
                activeforeground=T["text"], anchor=tk.W, padx=4, pady=2,
            ).pack(fill=tk.X, padx=32)

        # Background
        tk.Label(dlg, text="Background Image (optional)", font=("Segoe UI", 10),
                 fg=T["muted"], bg=T["bg"]).pack(anchor=tk.W, padx=24, pady=(16, 4))
        bg_var = tk.StringVar(value=self.settings.get("bg_image", ""))
        bg_row = tk.Frame(dlg, bg=T["bg"])
        bg_row.pack(fill=tk.X, padx=24)
        tk.Entry(bg_row, textvariable=bg_var, font=("Segoe UI", 9),
                 bg=T["surface"], fg=T["text"], relief=tk.FLAT, width=30
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(bg_row, text="Browse", font=("Segoe UI", 8),
                  bg=T["surface"], fg=T["muted"], relief=tk.FLAT, padx=8,
                  command=lambda: self._browse_bg(dlg, bg_var)
                  ).pack(side=tk.RIGHT, padx=(4, 0))
        if bg_var.get():
            tk.Button(bg_row, text="Clear", font=("Segoe UI", 8),
                      bg=T["surface"], fg=T["muted"], relief=tk.FLAT, padx=8,
                      command=lambda: bg_var.set("")
                      ).pack(side=tk.RIGHT, padx=(2, 0))

        # Apply
        btns = tk.Frame(dlg, bg=T["bg"])
        btns.pack(fill=tk.X, padx=24, pady=(20, 16))
        tk.Button(btns, text="Apply & Restart", font=("Segoe UI", 11, "bold"),
                  bg=T["accent"], fg="#fff", relief=tk.FLAT, padx=20, pady=6,
                  command=lambda: self._apply_settings(dlg, theme_var.get(), bg_var.get()),
                  ).pack(side=tk.RIGHT)
        tk.Button(btns, text="Cancel", font=("Segoe UI", 11),
                  bg=T["surface"], fg=T["muted"], relief=tk.FLAT, padx=16, pady=6,
                  command=dlg.destroy,
                  ).pack(side=tk.RIGHT, padx=(8, 0))

    def _browse_bg(self, parent, var):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=parent, title="Select Background Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _apply_settings(self, dlg, theme_name, bg_path):
        self.settings["theme"] = theme_name
        self.settings["bg_image"] = bg_path
        save_settings(self.settings)
        dlg.destroy()
        # Restart the app using detected Python
        self.root.destroy()
        subprocess.Popen([PYTHON_EXE, __file__])
        sys.exit(0)

    # ── Site change ──────────────────────────────

    def _on_site_change(self, *_args, animate=True):
        T = self.theme
        sk = self.site_var.get()
        site = SITES.get(sk, {})

        # Videos
        videos = scan_videos(sk)
        self.video_paths = {v.name: v for v in videos}
        first = list(self.video_paths.keys())[0] if self.video_paths else ""
        self.video_var.set(first)
        for w in self.video_frame.winfo_children():
            w.destroy()
        if videos:
            for v in videos:
                tk.Radiobutton(
                    self.video_frame, text=f"  {v.name}  ({(v.stat().st_size/1024/1024):.0f} MB)",
                    variable=self.video_var, value=v.name,
                    font=("Segoe UI", 9), fg=T["muted"], bg=T["bg"],
                    selectcolor=T["bg"], activebackground=T["surface"],
                    activeforeground=T["text"], anchor=tk.W, padx=2, pady=1,
                ).pack(fill=tk.X)
        else:
            tk.Label(self.video_frame, text="  No videos found",
                     font=("Segoe UI", 9), fg=T["dim"], bg=T["bg"]).pack(anchor=tk.W)

        # Models
        site_model = site.get("model", "")
        models_dir = ROOT / "data" / "models"
        available = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []
        for w in self.model_frame.winfo_children():
            w.destroy()
        for mp in available:
            tag = "  ← site default" if (str(mp) == site_model or mp.name == site_model) else ""
            tk.Radiobutton(
                self.model_frame, text=f"  {mp.name}  ({(mp.stat().st_size/1024/1024):.0f} MB){tag}",
                variable=self.model_var, value=str(mp),
                font=("Segoe UI", 9), fg=T["muted"], bg=T["bg"],
                selectcolor=T["bg"], activebackground=T["surface"],
                activeforeground=T["text"], anchor=tk.W, padx=2, pady=1,
            ).pack(fill=tk.X)
        default_m = str(ROOT / site_model) if site_model else ""
        if default_m and Path(default_m).exists():
            self.model_var.set(default_m)
        elif available:
            self.model_var.set(str(available[0]))

        # Stride
        for w in self.stride_frame.winfo_children():
            w.destroy()
        for v in STRIDE_OPTIONS:
            tk.Radiobutton(
                self.stride_frame, text=f" {v}", variable=self.stride_var, value=v,
                font=("Segoe UI", 9), fg=T["muted"], bg=T["bg"],
                selectcolor=T["bg"], activebackground=T["surface"],
                activeforeground=T["text"], anchor=tk.W,
            ).pack(fill=tk.X)

        # Presets
        for w in self.preset_frame.winfo_children():
            w.destroy()
        for name, p in PRESETS.items():
            tk.Radiobutton(
                self.preset_frame, text=f"  {name}  (imgsz={p['imgsz']}, conf={p['conf']})",
                variable=self.preset_var, value=name,
                font=("Segoe UI", 9), fg=T["muted"], bg=T["bg"],
                selectcolor=T["bg"], activebackground=T["surface"],
                activeforeground=T["text"], anchor=tk.W, padx=2, pady=1,
            ).pack(fill=tk.X)

        # Preview
        if animate:
            self._show_preview(sk)

    def _build_site_radios(self):
        T = self.theme
        for sk, site in SITES.items():
            frame = tk.Frame(self.site_frame, bg=T["surface"])
            frame.pack(fill=tk.X)
            tk.Radiobutton(
                frame, text=f"  {site.get('display_name', sk)}",
                variable=self.site_var, value=sk,
                font=("Segoe UI", 10), fg=T["muted"], bg=T["surface"],
                selectcolor=T["surface"], activebackground="#1a1e23",
                activeforeground=T["text"],
                anchor=tk.W, padx=8, pady=5,
                command=self._on_site_change,
            ).pack(fill=tk.X)
            calib = site.get("calibration_image", "")
            if calib:
                tk.Label(frame, text=f"       calib: {calib}",
                         font=("Segoe UI", 7), fg=T["dim"], bg=T["surface"]
                         ).pack(anchor=tk.W, padx=8)

    # ── Launch ───────────────────────────────────

    def _on_launch(self):
        T = self.theme
        sk = self.site_var.get()
        vn = self.video_var.get()
        vp = self.video_paths.get(vn)
        preset = PRESETS.get(self.preset_var.get(), list(PRESETS.values())[0])
        model = self.model_var.get()
        stride = self.stride_var.get()

        if not vp:
            messagebox.showwarning("No Video", "Select a video first.")
            return
        if not model:
            messagebox.showwarning("No Model", "Select a model first.")
            return

        self.status_label.config(text=f"Launching: {sk} / {vn} ...", fg=T["orange"])
        self.launch_btn.config(state=tk.DISABLED, text="Starting...")
        self.root.update()

        try:
            proc = launch_pipeline(sk, vp, preset, model, stride)
        except Exception as e:
            messagebox.showerror("Launch Failed", str(e))
            self.launch_btn.config(state=tk.NORMAL, text="▶  Launch Pipeline")
            self.status_label.config(text="Ready")
            return

        self.root.after(3000, lambda: self._check(proc, sk, vn))

    def _check(self, proc, sk, vn):
        T = self.theme
        rc = proc.poll()
        if rc is None:
            self.status_label.config(text=f"Running — OpenCV windows open", fg=T["green"])
            self.root.after(5000, lambda: self._check(proc, sk, vn))
        elif rc == 0:
            self.status_label.config(text="Complete.", fg=T["muted"])
            self.launch_btn.config(state=tk.NORMAL, text="▶  Launch Pipeline")
        else:
            vp = self.video_paths.get(vn)
            log = ROOT / "outputs" / sk / (vp.stem if vp else "unknown") / "live" / "launcher.log"
            err = log.read_text(encoding="utf-8", errors="replace")[-1200:] if log.exists() else "No log."
            self.status_label.config(text=f"Error (code={rc})", fg=T["red"])
            self.launch_btn.config(state=tk.NORMAL, text="▶  Launch Pipeline")
            messagebox.showerror("Pipeline Error", f"Exit code {rc}.\n\n{err}")

    # ── Calibration ──────────────────────────────

    def _on_calib(self):
        """启动标定工具 — 拖拽 4 个角点，Enter 保存"""
        T = self.theme
        sk = self.site_var.get()
        # 使用原版 calibrate_homography.py（已验证过逻辑正确）
        tool = ROOT / "tools" / "calibrate_homography.py"
        if not tool.exists():
            messagebox.showerror("Error", f"Calibration tool not found:\n{tool}")
            return
        cmd = [PYTHON_EXE, str(tool), "--site", sk]
        try:
            kwargs = {"cwd": str(ROOT)}
            if sys.platform.startswith("win"):
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(cmd, **kwargs)
            self.status_label.config(
                text=f"Calibration launched for '{sk}' (OpenCV window)", fg=T["orange"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch calibration:\n{e}")

    def run(self):
        self._build_site_radios()
        self.root.after(200, lambda: self._on_site_change(animate=True))
        self.root.mainloop()


if __name__ == "__main__":
    LauncherApp().run()
