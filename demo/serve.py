#!/usr/bin/env python
"""
Serve the showcase site locally (rebuilds it from docs/results first).

    python demo/serve.py            # http://127.0.0.1:8765
    python demo/build_site.py       # just (re)build docs/demo/index.html

Append ?static to the URL for a motion-free view (useful for screenshots), ?theme=light|dark to force a theme.
"""
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--serve"] + sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_site import main  # noqa: E402

main()
