"""End-to-end matrix aggregation: means with intervals, paired ratio to the reference, text-agreement prefix."""
from inferno.experiments import report_e2e


def _run(tok1, tok4, ids):
    return {"config": "x", "results": {"1": {"tok_per_s_median": tok1}, "4": {"tok_per_s_median": tok4}},
            "first_prompt_token_ids": ids, "startup_sec": 10.0, "layer_check": None, "kernel_calls": None}


def _runs():
    return {
        "awq_marlin": {1: _run(140.0, 500.0, [1, 2, 3, 4]), 2: _run(142.0, 520.0, [1, 2, 3, 4]), 3: _run(144.0, 510.0, [1, 2, 3, 4])},
        "inferno_b8": {1: _run(140.0, 250.0, [1, 2, 9, 4]), 2: _run(142.0, 260.0, [1, 2, 9, 4]), 3: _run(144.0, 255.0, [1, 2, 9, 4])},
        "awq_classic": {1: _run(130.0, 480.0, [1, 2, 9, 4])},
    }


def test_means_and_paired_ratio_to_reference():
    s = report_e2e.summarize(_runs())
    b = s["configs"]["inferno_b8"]["by_concurrency"]
    assert b["1"]["mean"] == 142.0 and b["1"]["ratio_to_reference"]["mean"] == 1.0
    assert 0.48 < b["4"]["ratio_to_reference"]["mean"] < 0.51        # paired per repetition, not ratio of means
    assert b["4"]["ratio_to_reference"]["min"] <= b["4"]["ratio_to_reference"]["max"]
    assert s["configs"]["awq_classic"]["by_concurrency"]["1"]["ci95"] != s["configs"]["awq_classic"]["by_concurrency"]["1"]["ci95"]  # NaN with n = 1


def test_common_prefix_and_agreement_table():
    assert report_e2e.common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert report_e2e.common_prefix([1, 2], [1, 2]) == 2
    s = report_e2e.summarize(_runs())
    ag = s["text_agreement"]
    assert ag["inferno_b8 vs awq_marlin"]["identical_prefix"] == 2 and ag["inferno_b8 vs awq_marlin"]["identical_total"] == 0
    assert ag["inferno_b8 vs awq_classic"]["identical_total"] == 1


def test_markdown_states_the_reference():
    md = report_e2e.markdown(report_e2e.summarize(_runs()))
    assert "vLLM Marlin" in md and "paired ratio" in md
