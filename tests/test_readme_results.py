"""README results injection: fills only from data, keeps surrounding text, refuses when data is missing."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from .test_report import _write_run


def _load_script():
    spec = importlib.util.spec_from_file_location("update_readme_results", ROOT / "scripts" / "update_readme_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_block_is_filled_from_run_and_surroundings_preserved(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    run = _write_run(tmp_path / "run", 0.7)
    readme = tmp_path / "README.md"
    readme.write_text("# T\nintro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n", encoding="utf-8")
    mod = _load_script()
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--readme", str(readme),
                                      "--figures", str(tmp_path / "figs")])
    mod.main()
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# T\nintro\n") and text.endswith("outro\n")
    assert "old" not in text and "adaptive better" in text and "steady" in text and "bursty" in text
    assert (tmp_path / "figs" / "p99_by_pattern_light.png").exists()


def test_refuses_without_markers(tmp_path, monkeypatch):
    (tmp_path / "run").mkdir()
    run = _write_run(tmp_path / "run", 0.7)
    readme = tmp_path / "README.md"
    readme.write_text("no markers here", encoding="utf-8")
    mod = _load_script()
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--readme", str(readme)])
    with pytest.raises(SystemExit):
        mod.main()
