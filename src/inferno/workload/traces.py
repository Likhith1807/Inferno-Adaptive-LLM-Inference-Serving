"""
Real request-length traces (ShareGPT-style), replacing the synthetic lognormal
assumption with an empirical distribution of (prompt_tokens, output_tokens) pairs.

Trace file format: CSV with header `prompt_tokens,output_tokens`.
Build one with `python scripts/prepare_trace.py` (downloads ShareGPT and tokenizes).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def load_length_trace(path: str | Path) -> np.ndarray:
    """Load an (N, 2) int array of (prompt_tokens, output_tokens) from a CSV trace."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["prompt_tokens"]), int(row["output_tokens"])))
    if not rows:
        raise ValueError(f"Empty length trace: {path}")
    return np.asarray(rows, dtype=np.int64)


def empirical_sampler(pairs: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n (prompt, output) pairs with replacement. Pairs stay joint, which
    preserves the prompt/output correlation present in real conversations."""
    idx = rng.integers(0, len(pairs), size=n)
    return pairs[idx]


def build_sharegpt_trace(
    sharegpt_json: str | Path,
    out_csv: str | Path,
    tokenizer_name: str,
    max_prompt: int = 1536,
    max_output: int = 512,
    limit: int | None = None,
) -> int:
    """
    Turn a ShareGPT dump into (prompt_tokens, output_tokens) pairs.

    For each conversation, the first human turn is the prompt and the first
    assistant reply is the output. Pairs outside [4, max_*] are dropped, the
    same filter vLLM's own benchmark uses to remove degenerate samples.
    Returns the number of pairs written.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    with open(sharegpt_json, encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_tokens", "output_tokens"])
        for conv in data:
            turns = conv.get("conversations", [])
            if len(turns) < 2 or turns[0].get("from") != "human" or turns[1].get("from") != "gpt":
                continue
            p = len(tok(turns[0]["value"]).input_ids)
            o = len(tok(turns[1]["value"]).input_ids)
            if not (4 <= p <= max_prompt and 4 <= o <= max_output):
                continue
            w.writerow([p, o])
            written += 1
            if limit and written >= limit:
                break
    return written
