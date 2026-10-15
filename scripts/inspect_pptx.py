"""Dump the structure of a .pptx (slides, layouts, shapes, text, positions) so a deck can be built to match it exactly."""
import sys

from pptx import Presentation
from pptx.util import Emu

p = Presentation(sys.argv[1])
print("slide size", p.slide_width, p.slide_height, Emu(p.slide_width).inches, Emu(p.slide_height).inches)
print("layouts:", [(i, l.name) for i, l in enumerate(p.slide_layouts)])
for i, s in enumerate(p.slides, 1):
    print(f"\n=== slide {i} layout={s.slide_layout.name!r} shapes={len(s.shapes)}")
    for sh in s.shapes:
        t = sh.text_frame.text.replace("\n", " | ")[:200] if sh.has_text_frame else ""
        print(f"  [{sh.shape_type}] name={sh.name!r} ph={sh.placeholder_format.type if sh.is_placeholder else None} "
              f"pos=({Emu(sh.left).inches:.2f},{Emu(sh.top).inches:.2f}) size=({Emu(sh.width).inches:.2f},{Emu(sh.height).inches:.2f}) text={t!r}")
    if s.has_notes_slide:
        print("  notes:", s.notes_slide.notes_text_frame.text[:200])
