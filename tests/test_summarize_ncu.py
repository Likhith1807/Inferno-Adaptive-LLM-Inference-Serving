"""ncu CSV summarizer: parses the header/units/value layout and converts units."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("summarize_ncu", ROOT / "scripts" / "summarize_ncu.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HEADER = ["ID", "Kernel Name", "gpu__time_duration.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
          "sm__warps_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum",
          "launch__registers_per_thread", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
          "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"]
UNITS = ["", "", "nsecond", "%", "%", "inst", "register/thread", "", ""]
VALUES = ["0", "gemv_int4_vec", "31,250", "87.5", "61.2", "4,000,000", "40", "900", "1,000"]


def _csv_text():
    q = lambda row: ",".join(f'"{c}"' for c in row)
    return "\n".join([q(HEADER), q(UNITS), q(VALUES)]) + "\n"


def test_units_and_ratio(tmp_path, monkeypatch, capsys):
    (tmp_path / "int4_v2_vec_shape0.csv").write_text(_csv_text())
    m = _mod()
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path)])
    m.main()
    out = capsys.readouterr().out
    row = [l for l in out.splitlines() if l.startswith("| int4_v2_vec")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "31.25"          # 31,250 ns -> 31.25 us
    assert cells[3] == "87.5" and cells[4] == "0.90" and cells[6] == "4.00" and cells[7] == "40"
    assert (tmp_path / "summary.md").exists()
