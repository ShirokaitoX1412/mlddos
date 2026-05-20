"""Root Streamlit wrapper. Preferred command: streamlit run frontend/app.py."""

from pathlib import Path
import runpy


FRONTEND_APP = Path(__file__).resolve().parent / "frontend" / "app.py"
runpy.run_path(str(FRONTEND_APP), run_name="__main__")
