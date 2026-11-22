"""
==============================================================================
MODULE: databricks_app/server.py
------------------------------------------------------------------------------
- HOW THIS RELATES TO THE LAST FILE: this is the Databricks Apps entrypoint.
  It imports the existing FastAPI app from modeling/api/app.py unchanged
  (same /health and /recommend routes, same model logic) and adds exactly
  one thing on top: serving the built frontend (dist/) from the same
  process, so the whole app runs as a single Databricks App instead of a
  separate Vercel frontend + a separately hosted API.
- WHAT THIS FILE DOES: mounts dist/assets as static files, and serves
  dist/index.html for every other path (SPA fallback) so client-side
  routes and a hard refresh on any URL both work.
- WHY A SEPARATE FILE INSTEAD OF EDITING modeling/api/app.py: that file is
  also used standalone (e.g. behind Render, or any other ASGI host per its
  own header comment) without a bundled frontend. Keeping it untouched
  means that deployment path still works; this file only adds the
  Databricks-specific serving behavior on top.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from modeling.api.app import app  # noqa: E402

_DIST = _REPO_ROOT / "dist"
_INDEX_HTML = _DIST / "index.html"

assert _INDEX_HTML.exists(), (
    f"{_INDEX_HTML} not found — run 'npm run build' before deploying "
    "(the Databricks Apps runtime does not run a frontend build step)."
)

app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="frontend-assets")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str) -> FileResponse:
    """Serve the built SPA for any path not already matched above.

    Registered last so it never shadows /health or /recommend: FastAPI
    matches routes in registration order, and those two are added first
    inside modeling/api/app.py, which runs at import time, above.
    """
    return FileResponse(_INDEX_HTML)
