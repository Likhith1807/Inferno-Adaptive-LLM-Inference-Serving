"""Power sampler and hardware capture. GPU-dependent parts skip when nvidia-smi is absent."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from inferno.hardware import capture_hardware
from inferno.power import PowerSampler, nvidia_smi_available


def test_capture_hardware_has_core_fields():
    hw = capture_hardware()
    assert {"timestamp_utc", "platform", "python", "gpus", "packages"} <= set(hw)


def test_power_summary_with_too_few_samples():
    assert PowerSampler().summary()["n_power_samples"] == 0


def test_energy_integration_is_trapezoidal():
    ps = PowerSampler()
    ps.samples = [(0.0, 100.0, 1000.0, 50.0), (1.0, 200.0, 2000.0, 60.0), (2.0, 200.0, 3000.0, 70.0)]
    s = ps.summary(output_tokens=1000)
    assert s["energy_j"] == pytest.approx(150.0 + 200.0)     # 150 J then 200 J
    assert s["tokens_per_joule"] == pytest.approx(1000 / 350.0)
    assert s["peak_mem_mib"] == 3000.0


@pytest.mark.skipif(not nvidia_smi_available(), reason="needs nvidia-smi")
def test_live_sampling_collects_samples():
    with PowerSampler(interval_ms=100) as ps:
        time.sleep(1.5)
    assert len(ps.samples) >= 3


def test_multi_gpu_ticks_are_summed():
    ps = PowerSampler(gpu_indices=[0, 1])
    assert ps._n_gpus == 2
    for t, lines in ((1.0, ["50.0, 1000, 40", "70.0, 3000, 60"]), (2.0, ["60.0, 1000, 50", "[N/A], x, y", "80.0, 3000, 70"])):
        for line in lines:
            ps._ingest(line, t)
    assert len(ps.samples) == 2
    assert ps.samples[0] == (1.0, 120.0, 4000.0, 50.0)     # power/memory summed, utilization averaged
    assert ps.samples[1][1] == 140.0                       # malformed line skipped, tick still completes
