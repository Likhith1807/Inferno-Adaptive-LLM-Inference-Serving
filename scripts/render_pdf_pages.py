"""Render selected PDF pages to PNG for visual review.   python scripts/render_pdf_pages.py <pdf> <outdir> 2 43 44"""
import sys
from pathlib import Path

import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for n in map(int, sys.argv[3:]):
    img = pdf[n - 1].render(scale=1.6).to_pil()
    img.save(out / f"page_{n:03d}.png")
    print("wrote", out / f"page_{n:03d}.png")
