"""The showcase site builds from the committed result files and embeds valid, complete data."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_site_builds_with_all_data_blocks():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, str(ROOT / "demo" / "build_site.py")], check=True, cwd=ROOT, env=env)
    html = (ROOT / "docs" / "demo" / "index.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html
    data = json.loads(re.search(r"const D = (\{.*?\});\s*const \$ =", html, re.S).group(1))
    for block in ("kernel", "campaign", "contention", "sweep", "mpi", "replay"):
        assert data[block], f"missing data block: {block}"
    assert {"steady", "bursty", "mixed_length"} <= set(data["replay"])
    assert all(r["pct_change"] > 0 for r in data["campaign"]["h1"])     # the negative result is on the page
    assert data["kernel"]["maxerr"] < 5e-3                              # correctness figure shown on the page
    assert "cdn" not in html.lower()                                    # fully offline: no external scripts
