#!/usr/bin/env python
"""
Build data/sharegpt_lengths.csv: real (prompt_tokens, output_tokens) pairs from ShareGPT,
tokenized with the study model's own tokenizer.

    python scripts/prepare_trace.py                       # downloads ShareGPT (~640 MB) then tokenizes
    python scripts/prepare_trace.py --json path/to.json   # use a local copy

Then set `request_length.distribution: "empirical"` in configs/workload_config.yaml.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inferno.workload.traces import build_sharegpt_trace  # noqa: E402

URL = ("https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/"
       "ShareGPT_V3_unfiltered_cleaned_split.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="data/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--out", default="data/sharegpt_lengths.csv")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, default=20000, help="max pairs to keep")
    args = ap.parse_args()

    src = Path(args.json)
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} -> {src}")
        urllib.request.urlretrieve(URL, src)

    n = build_sharegpt_trace(src, args.out, args.tokenizer, limit=args.limit)
    print(f"Wrote {n} (prompt, output) pairs to {args.out}")


if __name__ == "__main__":
    main()
