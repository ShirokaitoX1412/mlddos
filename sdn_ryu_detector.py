"""Ryu entrypoint exposed at repo root.

Ryu-manager imports this file and discovers MLDDoSRyuController in globals.
"""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_ddos.sdn_ryu_detector import MLDDoSRyuController as _BaseController  # noqa: E402
from ml_ddos.sdn_ryu_detector import main  # noqa: E402


class MLDDoSRyuController(_BaseController):
    """Expose the Ryu app class in this module for ryu-manager discovery."""


if __name__ == "__main__":
    main()
