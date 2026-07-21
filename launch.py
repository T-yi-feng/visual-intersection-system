"""
Visual Traffic Analysis — Unified Launcher

双击此文件或执行:
    E:\Anaconda\python.exe
    python launch.py

自动启动 Streamlit 智能控制台，提供:
  - 可视化站点卡片选择
  - 一键预设启动
  - 实时 Phi 监控
  - 事件历史浏览
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    console_script = root / "app" / "smart_console.py"

    if not console_script.exists():
        print(f"[ERROR] Console script not found: {console_script}")
        print("Please run from the project root directory.")
        sys.exit(1)

    print("=" * 60)
    print("  Traffic Intersection Analysis System")
    print("  Launching Smart Console...")
    print("=" * 60)
    print()
    print("  The console will open in your browser.")
    print("  If it doesn't open automatically, visit:")
    print("  http://localhost:8501")
    print()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(console_script),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
    ]

    try:
        subprocess.run(cmd, cwd=str(root))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch: {e}")
        print("\nTry running manually:")
        print(f"  streamlit run app/smart_console.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
