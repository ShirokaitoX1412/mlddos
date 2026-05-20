"""Vercel Python entrypoint.

Vercel expects a top-level variable named `app` in a Python file under `api/`.
The actual backend implementation lives in `backend/api/index.py`.
"""

from backend.api.index import app  # noqa: F401
