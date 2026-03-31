"""
Cross-platform Jupyter Lab startup script.
Starts Jupyter Lab in the background if not already running.
"""

import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PORT = 8888
TOKEN = "mm2026"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = REPO_ROOT / ".jupyter_cache" / "jupyter_server.log"


def is_jupyter_running() -> bool:
    try:
        url = f"http://localhost:{PORT}/api/status?token={TOKEN}"
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_jupyter() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "jupyter", "lab",
        f"--port={PORT}",
        "--ip=0.0.0.0",
        "--no-browser",
        f"--IdentityProvider.token={TOKEN}",
        f"--SQLiteYStore.db_path={REPO_ROOT / '.jupyter_cache' / 'jupyter_ystore.db'}",
    ]

    with open(LOG_FILE, "w") as log:
        subprocess.Popen(
            cmd,
            stdout=log,
            stderr=log,
            cwd=REPO_ROOT,
            start_new_session=True,
        )

    print("Waiting for Jupyter Lab to become ready...", file=sys.stderr)
    for _ in range(60):
        time.sleep(1)
        if is_jupyter_running():
            print("Jupyter Lab is ready.", file=sys.stderr)
            return

    print(f"Error: Jupyter Lab failed to start. Check {LOG_FILE}", file=sys.stderr)
    sys.exit(1)


if is_jupyter_running():
    print(f"Jupyter Lab is already running on port {PORT}.", file=sys.stderr)
else:
    print(f"Jupyter Lab not detected on port {PORT}. Starting...", file=sys.stderr)
    start_jupyter()
