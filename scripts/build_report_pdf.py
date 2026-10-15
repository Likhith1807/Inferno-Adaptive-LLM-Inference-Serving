"""Builds docs/Inferno_Technical_Report.pdf from the measured result files. Run from the repository root:

    python scripts/build_report_pdf.py [--out docs/Inferno_Technical_Report.pdf]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reportlab.platypus import KeepTogether  # noqa: E402

from report_pdf import figures, sec_back, sec_intro, sec_lit, sec_method, sec_results, style  # noqa: E402


def _flat(items):
    out = []
    for x in items:
        out += _flat(x) if isinstance(x, list) else [x]
    return out


def _group_headings(story):
    """keepWithNext cannot chain a heading to a KeepTogether (table or figure), so do the grouping here."""
    out, i = [], 0
    while i < len(story):
        f = story[i]
        if isinstance(f, style.Heading) and f.level >= 2:
            group = [f]
            j = i + 1
            while j < len(story) and isinstance(story[j], style.Heading) and story[j].level >= 2:
                group.append(story[j])
                j += 1
            if j < len(story):
                nxt = story[j]
                group += list(nxt._content) if isinstance(nxt, KeepTogether) else [nxt]
                j += 1
            out.append(KeepTogether(group))
            i = j
        else:
            out.append(f)
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "Inferno_Technical_Report.pdf"))
    ap.add_argument("--figs", default=str(ROOT / "build_report" / "figs"))
    a = ap.parse_args()
    style.register_fonts()
    Path(a.figs).mkdir(parents=True, exist_ok=True)
    figs = figures.build_all(a.figs)

    story = []
    story += sec_intro.cover()
    story += sec_intro.abstract()
    story += sec_intro.contents()
    for chapter in (sec_intro.chapter_intro(), sec_intro.chapter_problem(), sec_intro.chapter_objectives(),
                    sec_lit.chapter_literature(figs), sec_lit.chapter_previous_work(),
                    sec_method.chapter_design(figs), sec_method.chapter_methodology(figs),
                    sec_results.chapter_implementation(), sec_results.chapter_results(), sec_results.chapter_discussion(),
                    sec_results.chapter_limits(), sec_results.chapter_conclusion(),
                    sec_back.appendices(), sec_back.references()):
        story += _flat(chapter)
    story = _group_headings(story)
    doc = style.ReportDoc(a.out, "Inferno: Adaptive LLM Inference Serving, Measured End to End", "Inferno technical report")
    doc.multiBuild(story)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
