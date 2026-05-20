"""Compatibility wrapper for ml_ddos.data_loader."""

from pathlib import Path
import runpy
import sys


SRC_DIR = Path(__file__).resolve().parent / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_ddos.data_loader import *  # noqa: E402,F401,F403


if __name__ == "__main__":
    runpy.run_module("ml_ddos.data_loader", run_name="__main__")
