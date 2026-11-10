"""Report generators for the layer matrix and the Marlin correctness probe (no GPU needed)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(us, spread=0.5):
    return {"us_median": us, "us_min": us - spread, "us_max": us + spread, "trials": 3, "copies": 10}


def test_layer_matrix_report_winners(tmp_path):
    batches = [1, 16, 32]
    shapes = {}
    for name, (c1, m1) in {"qkv": (10.0, 12.0), "o_proj": (12.0, 12.2), "gate_up": (90.0, 95.0), "down": (50.0, 45.0)}.items():
        cells = {"custom/B1": _cell(c1), "custom/B16": _cell(c1 * 6), "custom/B32": None,
                 "marlin/B1": _cell(m1), "marlin/B16": _cell(m1 * 1.1), "marlin/B32": _cell(m1 * 1.6),
                 "awq/B1": _cell(m1 * 1.2), "awq/B16": _cell(m1 * 1.5), "awq/B32": _cell(m1 * 2.5),
                 "fp16/B1": _cell(m1 * 3), "fp16/B16": _cell(m1 * 3), "fp16/B32": _cell(m1 * 3)}
        shapes[name] = {"N": 1, "K": 1, "cells": cells,
                        "conversion_ms": {"custom": 1.0, "marlin": 2.0, "awq": 0.0, "fp16": 1.0},
                        "max_rel_err": {"custom": 1e-4, "marlin": 1e-4, "awq": 1e-4, "fp16": 1e-4}}
    src = tmp_path / "lm.json"
    src.write_text(json.dumps({"gpu": "test", "vllm": "0", "torch": "0", "group": 128, "method": "m", "rotate_mb": 1,
                               "reps": 1, "trials": 3, "batches": batches, "shapes": shapes}))
    mod = _load("report_layer_matrix")
    import sys
    sys.argv = ["x", str(src), str(tmp_path / "out")]
    mod.main()
    md = (tmp_path / "out" / "report.md").read_text()
    assert "| qkv (1x1) | **custom** 1.20x" in md          # clear win, non-overlapping ranges
    assert "tie (custom/marlin)" in md                        # o_proj: overlapping ranges -> tie
    assert "**marlin**" in md and "n/a" in md


def test_marlin_probe_parser():
    mod = _load("summarize_marlin_fault")
    txt = ("N=1536 K=8960 alloc=default gpu=NVIDIA Test GPU SMs=24 workspace=(24,)\n"
           "B=   1 plain    bad= 0/40 worst_rel_err=3.09e-04\n"
           "B=  32 plain    bad=19/40 worst_rel_err=1.11e-01\n")
    out = mod.parse(txt)
    assert out["gpu"] == "NVIDIA Test GPU" and out["sms"] == 24
    s = out["shapes"][0]
    assert s["name"] == "down" and len(s["rows"]) == 2
    assert s["rows"][1] == {"B": 32, "variant": "plain", "bad": 19, "calls": 40, "worst_rel_err": 0.111}
