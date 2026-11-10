"""
Turn results/layer_matrix/layer_matrix.json into docs/results/layer_matrix/report.md: which int4 implementation wins
each (Qwen layer shape, batch size) cell, by how much, and how much of a decode step each shape accounts for.

A cell counts as a win only if the winner's slowest trial beats the runner-up's fastest trial (non-overlapping
ranges); otherwise it is reported as a tie. fp16 is shown as a reference but is not a deployable int4 candidate.

    python scripts/report_layer_matrix.py [results/layer_matrix/layer_matrix.json] [docs/results/layer_matrix]
"""
import json
import shutil
import sys
from pathlib import Path

INT4 = ["custom", "marlin", "awq"]


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/layer_matrix/layer_matrix.json")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/results/layer_matrix")
    r = json.loads(src.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out_dir / "layer_matrix.json")
    B = r["batches"]
    L = []
    L.append("# Qwen2.5-1.5B int4 layers: which implementation wins? (generated)\n")
    L.append(f"GPU {r['gpu']}, vLLM {r['vllm']}, torch {r['torch']}, group {r['group']}. {r['method']}; "
             f"rotation >= {r['rotate_mb']:.0f} MB, {r['trials']} trials x {r['reps']} graph replays per cell. "
             "Times are median microseconds per call. A win requires non-overlapping trial ranges, otherwise `tie`. "
             "`n/a`: the custom kernel is not instantiated for that batch (the deployed path falls back).\n")

    # win table
    L.append("## Winner per cell (int4 implementations only)\n")
    L.append("| shape (N x K) | " + " | ".join(f"B={b}" for b in B) + " |")
    L.append("|---|" + "---|" * len(B))
    wins = {}
    for name, s in r["shapes"].items():
        row = []
        for b in B:
            cells = {i: s["cells"].get(f"{i}/B{b}") for i in INT4}
            cells = {i: c for i, c in cells.items() if c}
            order = sorted(cells, key=lambda i: cells[i]["us_median"])
            best, second = order[0], order[1]
            ratio = cells[second]["us_median"] / cells[best]["us_median"]
            clear = cells[best]["us_max"] < cells[second]["us_min"]
            wins[(name, b)] = best if clear else "tie"
            row.append(f"**{best}** {ratio:.2f}x" if clear else f"tie ({best}/{second})")
        L.append(f"| {name} ({s['N']}x{s['K']}) | " + " | ".join(row) + " |")
    L.append("\nRatio = runner-up time / winner time.\n")

    # full timing table
    L.append("## Median us per call\n")
    L.append("| shape | impl | " + " | ".join(f"B={b}" for b in B) + " |")
    L.append("|---|---|" + "---|" * len(B))
    for name, s in r["shapes"].items():
        for i in INT4 + ["fp16"]:
            vals = []
            for b in B:
                c = s["cells"].get(f"{i}/B{b}")
                vals.append("n/a" if not c else f"{c['us_median']:.1f}")
            L.append(f"| {name} | {i} | " + " | ".join(vals) + " |")

    # custom vs marlin directly
    L.append("\n## Custom kernel vs Marlin (vLLM's default int4 path), time ratio custom/marlin\n")
    L.append("| shape | " + " | ".join(f"B={b}" for b in B) + " |")
    L.append("|---|" + "---|" * len(B))
    for name, s in r["shapes"].items():
        vals = []
        for b in B:
            c, m = s["cells"].get(f"custom/B{b}"), s["cells"].get(f"marlin/B{b}")
            vals.append("n/a" if not (c and m) else f"{c['us_median'] / m['us_median']:.2f}")
        L.append(f"| {name} | " + " | ".join(vals) + " |")
    L.append("\n< 1 means the custom kernel is faster.\n")

    # decode-time share (each shape appears once per transformer layer, so its share of the quantized-GEMM time
    # in a decode step is its time over the sum across shapes)
    L.append("## Share of quantized-GEMM time per decode step (using Marlin times)\n")
    L.append("| batch | " + " | ".join(r["shapes"]) + " |")
    L.append("|---|" + "---|" * len(r["shapes"]))
    for b in B:
        t = {n: s["cells"][f"marlin/B{b}"]["us_median"] for n, s in r["shapes"].items()}
        tot = sum(t.values())
        L.append(f"| {b} | " + " | ".join(f"{100 * v / tot:.0f}%" for v in t.values()) + " |")

    # one-time costs and correctness
    L.append("\n## One-time layout conversion (ms, paid once at model load) and correctness\n")
    L.append("| shape | " + " | ".join(f"{i} convert ms" for i in INT4) + " | " +
             " | ".join(f"{i} max rel err" for i in INT4) + " |")
    L.append("|---|" + "---|" * (2 * len(INT4)))
    for name, s in r["shapes"].items():
        L.append(f"| {name} | " + " | ".join(f"{s['conversion_ms'][i]:.2f}" for i in INT4) + " | " +
                 " | ".join(f"{s['max_rel_err'][i]:.1e}" for i in INT4) + " |")
    L.append("\n`awq` needs no conversion (it runs on the checkpoint layout). Errors are relative max errors vs "
             "awq_dequantize + fp32 matmul.\n")

    n_custom = sum(1 for v in wins.values() if v == "custom")
    n_cells = sum(1 for (n, b) in wins if f"custom/B{b}" in r["shapes"][n]["cells"]
                  and r["shapes"][n]["cells"][f"custom/B{b}"])
    L.append(f"## Verdict\n\nThe custom kernel clearly wins {n_custom} of the {n_cells} cells where it runs. "
             "Cell winners: " + ", ".join(f"{k}: {sum(1 for v in wins.values() if v == k)}"
                                           for k in INT4 + ["tie"]) + ".\n")
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
