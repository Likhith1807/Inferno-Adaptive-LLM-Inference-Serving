#!/usr/bin/env python
"""
Copy the small, commit-ready kernel-study result files from results/ (raw, gitignored) into docs/results/ (what the
report generators read). Raw per-request logs and profiler dumps stay in results/.

    python scripts/sync_docs_results.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "results", ROOT / "docs" / "results"

FILES = ["kernel_bench.json", "kernel_bench.txt", "kernel_comparison.md"]
DIRS = {"kernel_bench_runs": "*.json", "kernel_batch_runs": "*.json", "flush_sweep": "*.json", "insitu": "*.json"}
SUPERSEDED = ["vllm_awq_marlin_decode.json", "vllm_awq_decode.json", "vllm_marlin_decode.json"]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (SRC / f).exists():
            shutil.copy(SRC / f, DST / f)
            print("copied", f)
    for d, pat in DIRS.items():
        if (SRC / d).is_dir():
            (DST / d).mkdir(exist_ok=True)
            for p in sorted((SRC / d).glob(pat)):
                shutil.copy(p, DST / d / p.name)
            print("copied", d)
    # The pre-correction in-situ timings (an invalid Marlin measurement) must not stay next to the valid ones.
    old = SRC / "superseded"
    for f in SUPERSEDED:
        for base in (SRC, DST):
            p = base / f
            if p.exists():
                old.mkdir(exist_ok=True)
                shutil.move(str(p), str(old / f)) if base is SRC else p.unlink()
                print("retired", base.name + "/" + f)


if __name__ == "__main__":
    main()
