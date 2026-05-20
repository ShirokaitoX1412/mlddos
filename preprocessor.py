"""Compatibility wrapper for ml_ddos.preprocessor."""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_ddos.preprocessor import *  # noqa: E402,F401,F403
