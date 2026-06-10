"""Replay IPS entrypoint kept at repo root for compatibility."""

import os
from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
if LOCAL_VENV_PYTHON.exists() and Path(sys.executable).resolve() != LOCAL_VENV_PYTHON.resolve():
    os.execv(str(LOCAL_VENV_PYTHON), [str(LOCAL_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

SRC_DIR = Path(__file__).resolve().parent / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    runpy.run_module("ml_ddos.replay_ips", run_name="__main__")
