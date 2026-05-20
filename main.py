"""Training pipeline entrypoint kept at repo root for compatibility."""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_ddos.main import main  # noqa: E402


if __name__ == "__main__":
    main()
