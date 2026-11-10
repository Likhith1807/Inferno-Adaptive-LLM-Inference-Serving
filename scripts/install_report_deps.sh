#!/usr/bin/env bash
# Extra packages for `make pdf` (the technical report), installed into the project venv.
set -euo pipefail
PY="${HOME}/.venvs/inferno/bin/python"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" reportlab pillow
else
  "$PY" -m pip install reportlab pillow
fi
"$PY" -c "import reportlab; print('reportlab', reportlab.Version)"
