#!/usr/bin/env python
"""
Merge repeated in-situ kernel profiles (results/insitu/<path>_<mode>_rep<r>.json, one fresh process each) into
results/insitu/<path>_<mode>.json. A single process can differ from another by ~10% (GPU clock state), so the
reported per-role time is the median across processes, with the min and max kept.

    python scripts/aggregate_insitu.py results/insitu
"""
import json
import re
import statistics
import sys
from pathlib import Path


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/insitu")
    groups: dict = {}
    for p in sorted(d.glob("*_rep*.json")):
        m = re.match(r"(.+)_rep(\d+)\.json$", p.name)
        groups.setdefault(m.group(1), []).append(json.loads(p.read_text()))
    for name, runs in groups.items():
        base = dict(runs[0])
        ok = [r for r in runs if r.get("roles")]
        if not ok:
            base["roles"] = None
            base["warning"] = "no repetition produced per-role attribution"
        else:
            roles = {}
            for role in ok[0]["roles"]:
                v = [r["roles"][role]["us_median"] for r in ok]
                roles[role] = {"us_median": statistics.median(v), "us_min": min(v), "us_max": max(v), "n": len(v),
                               "spread_pct": 100 * (max(v) - min(v)) / statistics.median(v)}
            base["roles"] = roles
            base.pop("warning", None)
        base["n_process_reps"] = len(runs)
        # A GPU-wide slowdown (throttling, another workload) makes every kernel slower, FlashAttention included; flag such processes.
        fl = []
        for r in runs:
            t = next((k for k in r.get("top_kernels_by_total_us", []) if "flash" in k["name"].lower()), None)
            fl.append(t["total_us"] / t["calls"] if t and t["calls"] else None)
        ref = statistics.median(v for v in fl if v) if any(fl) else None
        base["flash_us_per_call"] = fl
        base["slow_process_reps"] = [i + 1 for i, v in enumerate(fl) if ref and v and v > 1.5 * ref]
        (d / f"{name}.json").write_text(json.dumps(base, indent=2))
        print(name, {r: round(v["us_median"], 1) for r, v in (base["roles"] or {}).items()} or base.get("warning"))


if __name__ == "__main__":
    main()
