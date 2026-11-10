"""Print the page numbers of a PDF that contain each search phrase.   python scripts/find_pdf_pages.py <pdf> <phrase>..."""
import sys

from pypdf import PdfReader

r = PdfReader(sys.argv[1])
print("pages:", len(r.pages))
for phrase in sys.argv[2:]:
    hits = [i + 1 for i, p in enumerate(r.pages) if phrase.lower() in (p.extract_text() or "").lower().replace("\n", " ")]
    print(f"{phrase!r}: {hits}")
