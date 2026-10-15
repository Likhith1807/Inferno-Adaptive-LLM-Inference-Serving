#!/usr/bin/env python
"""
Post-hoc headroom analysis of the adaptive-routing campaign (uses only the existing campaign data; no new runs).

Question: how much could ANY policy that routes each pattern to its better fixed tier gain over the best single fixed
tier, on this shared-GPU pool? If the two fixed tiers are nearly equal, an adaptive policy has almost nothing to win,
which explains why H1 could not be confirmed independently of the co-residency cost of the policy actually tested.

    python scripts/headroom_analysis.py            # writes docs/results/campaign/headroom.json and headroom.md
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results" / "campaign"
FIXED = ["fixed-fast", "fixed-bulk"]


def main() -> None:
    df = pd.read_csv(RES / "summary.csv")
    patterns = sorted(df.pattern.unique())
    mean = {(p, c): float(df[(df.pattern == p) & (df.config == c)].e2e_p99.mean()) for p in patterns for c in df.config.unique()}
    rows = []
    for p in patterns:
        f = {c: mean[(p, c)] for c in FIXED}
        best = min(f, key=f.get)
        rows.append({"pattern": p, "p99_fast": f["fixed-fast"], "p99_bulk": f["fixed-bulk"], "oracle_tier": best,
                     "oracle_p99": f[best], "static_lb_p99": mean[(p, "static-lb")], "adaptive_p99": mean[(p, "adaptive")]})
    # best single fixed tier = the one with the smaller worst-case regret (regret = p99 / per-pattern oracle p99)
    regret = {c: max(mean[(p, c)] / min(mean[(p, c2)] for c2 in FIXED) for p in patterns) for c in FIXED}
    single = min(regret, key=regret.get)
    for r in rows:
        r["single_fixed_regret"] = r[f"p99_{single.split('-')[1]}"] / r["oracle_p99"] - 1.0
        r["oracle_gain_vs_single_pct"] = max(0.0, -100 * (r["oracle_p99"] / r[f"p99_{single.split('-')[1]}"] - 1.0))
        r["adaptive_vs_oracle_pct"] = 100 * (r["adaptive_p99"] / r["oracle_p99"] - 1.0)
    out = {"single_fixed_tier": single, "worst_case_regret": regret, "rows": rows,
           "max_headroom_pct": max(r["oracle_gain_vs_single_pct"] for r in rows)}
    (RES / "headroom.json").write_text(json.dumps(out, indent=2))
    L = [f"Best single fixed tier by worst-case regret: **{single}**.\n",
         "| pattern | fixed-fast p99 (s) | fixed-bulk p99 (s) | per-pattern oracle | oracle gain over " + single + " | adaptive vs oracle |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['pattern']} | {r['p99_fast']:.2f} | {r['p99_bulk']:.2f} | {r['oracle_tier']} | {max(0.0, r['oracle_gain_vs_single_pct']):.1f}% | {r['adaptive_vs_oracle_pct']:+.0f}% |")
    L.append(f"\nMaximum headroom for any pattern-aware choice between the two fixed tiers: {out['max_headroom_pct']:.1f}% of p99.")
    (RES / "headroom.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
