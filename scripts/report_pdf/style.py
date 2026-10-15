"""
Typesetting foundation for the technical report: fonts, paragraph styles, numbered headings with TOC and PDF
outline, figure/table captions with automatic numbering, callout boxes, and LaTeX-style equations (rendered with
matplotlib mathtext, so no LaTeX installation is required).
"""
from __future__ import annotations

import hashlib
import io
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem,  # noqa: E402
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

PAGE = A4
LM = RM = 22 * mm
TM, BM = 24 * mm, 22 * mm
TEXT_W = PAGE[0] - LM - RM

INK = colors.HexColor("#14171c")
INK2 = colors.HexColor("#4a4f57")
MUTED = colors.HexColor("#7a808a")
ACCENT = colors.HexColor("#1f6fd0")
ACCENT2 = colors.HexColor("#17916a")
RULE = colors.HexColor("#d5d8de")
BAND = colors.HexColor("#f2f4f7")
WARN = colors.HexColor("#b4691a")
BAD = colors.HexColor("#c0392b")

_FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"


def register_fonts() -> None:
    for name, f in (("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"), ("Serif-I", "DejaVuSerif-Italic.ttf"),
                    ("Serif-BI", "DejaVuSerif-BoldItalic.ttf"), ("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
                    ("Sans-I", "DejaVuSans-Oblique.ttf"), ("Sans-BI", "DejaVuSans-BoldOblique.ttf"),
                    ("Mono", "DejaVuSansMono.ttf"), ("Mono-B", "DejaVuSansMono-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(_FONT_DIR / f)))
    pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-BI")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-B", italic="Mono", boldItalic="Mono-B")


register_fonts()

S = {
    "body": ParagraphStyle("body", fontName="Serif", fontSize=9.6, leading=14.2, alignment=TA_JUSTIFY, textColor=INK,
                           spaceAfter=6.5, hyphenationLang=None),
    "small": ParagraphStyle("small", fontName="Serif", fontSize=8.4, leading=11.6, alignment=TA_JUSTIFY, textColor=INK2,
                            spaceAfter=4),
    "h1": ParagraphStyle("h1", keepWithNext=1, fontName="Sans-B", fontSize=20, leading=25, textColor=INK, spaceBefore=4, spaceAfter=12),
    "h2": ParagraphStyle("h2", keepWithNext=1, fontName="Sans-B", fontSize=13.2, leading=17, textColor=INK, spaceBefore=14, spaceAfter=6),
    "h3": ParagraphStyle("h3", keepWithNext=1, fontName="Sans-B", fontSize=10.6, leading=14, textColor=ACCENT, spaceBefore=10, spaceAfter=4),
    "h4": ParagraphStyle("h4", fontName="Sans-B", fontSize=9.4, leading=12.5, textColor=INK2, spaceBefore=6, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.1, leading=11, textColor=INK2, alignment=TA_LEFT, spaceBefore=3,
                          spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=7.9, leading=10.2, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Sans-B", fontSize=7.9, leading=10.2, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10.2, textColor=INK, leftIndent=6, rightIndent=6,
                           backColor=BAND, borderPadding=(5, 6, 5, 6), spaceBefore=4, spaceAfter=9),
    "call": ParagraphStyle("call", fontName="Serif", fontSize=9.2, leading=13.4, alignment=TA_LEFT, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="Sans-B", fontSize=10, leading=15, leftIndent=0, spaceBefore=4),
    "toc2": ParagraphStyle("toc2", fontName="Sans", fontSize=8.8, leading=12.4, leftIndent=14),
    "toc3": ParagraphStyle("toc3", fontName="Sans", fontSize=8.2, leading=11.4, leftIndent=28, textColor=INK2),
    "ref": ParagraphStyle("ref", fontName="Serif", fontSize=8.5, leading=11.8, leftIndent=16, firstLineIndent=-16,
                          alignment=TA_LEFT, spaceAfter=3.2),
    "center": ParagraphStyle("center", fontName="Serif", fontSize=9.6, leading=14, alignment=TA_CENTER),
}


# --------------------------------------------------------------------------- numbering state
class Counters:
    def __init__(self):
        self.h = [0, 0, 0]
        self.fig = 0
        self.tab = 0
        self.eq = 0
        self.appendix = False

    def heading(self, level: int) -> str:
        self.h[level - 1] += 1
        for i in range(level, 3):
            self.h[i] = 0
        if self.appendix:
            letter = chr(ord("A") + self.h[0] - 1)
            return ".".join([letter] + [str(x) for x in self.h[1:level]])
        return ".".join(str(x) for x in self.h[:level])


C = Counters()


class Heading(Paragraph):
    """Paragraph that carries its level and text for the TOC and PDF outline."""

    def __init__(self, text, style, level, number):
        super().__init__(f"{number}&nbsp;&nbsp;{text}" if (level < 4 and number) else text, style)
        self.level, self.plain, self.number = level, re.sub(r"<[^>]+>", "", text), number


def H(level: int, text: str, page_break: bool | None = None):
    num = C.heading(level) if level < 4 else ""
    out = []
    if level == 1 and (page_break is None or page_break):
        out.append(PageBreak())
    out.append(Heading(text, S[f"h{level}"], level, num))
    return out


def UH(text: str, page_break: bool = True):
    """Unnumbered level-1 heading that still appears in the contents and the PDF outline."""
    out = [PageBreak()] if page_break else []
    out.append(Heading(text, S["h1"], 1, ""))
    return out


def P(text: str, style: str = "body"):
    return Paragraph(text, S[style])


def CODE(text: str):
    body = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>").replace(" ", "&nbsp;")
    return Paragraph(body, S["code"])


def BUL(items, style="body", bullet="•"):
    return ListFlowable([ListItem(Paragraph(i, S[style]), leftIndent=14, bulletColor=ACCENT) for i in items],
                        bulletType="bullet", start=bullet, leftIndent=14, bulletFontSize=8, spaceAfter=4)


def NUM(items, style="body"):
    return ListFlowable([ListItem(Paragraph(i, S[style]), leftIndent=18) for i in items], bulletType="1",
                        bulletFormat="%s.", leftIndent=18, bulletFontName="Sans-B", bulletFontSize=8.5, spaceAfter=4)


def CALLOUT(title: str, text: str, color=ACCENT2):
    inner = [Paragraph(f"<font name='Sans-B' color='{color.hexval().replace('0x', '#')}'>{title}</font>", S["call"]),
             Paragraph(text, S["call"])]
    t = Table([[inner]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND), ("LINEBEFORE", (0, 0), (0, -1), 2.6, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return [t, Spacer(1, 8)]


# --------------------------------------------------------------------------- tables and figures
def TABLE(rows, widths=None, caption: str | None = None, header=True, align=None, zebra=True, font=None):
    """rows: list of lists of str; first row is the header. widths as fractions of the text width or absolute points."""
    n = len(rows[0])
    if widths is None:
        widths = [1 / n] * n
    widths = [w * TEXT_W if w <= 1.0 else w for w in widths]
    cell, cellh = S["cell"], S["cellh"]
    if font:
        cell = ParagraphStyle("c2", parent=cell, fontSize=font, leading=font + 2.4)
        cellh = ParagraphStyle("h2", parent=cellh, fontSize=font, leading=font + 2.4)
    data = []
    for ri, r in enumerate(rows):
        data.append([Paragraph(str(c), cellh if (header and ri == 0) else cell) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4.5), ("RIGHTPADDING", (0, 0), (-1, -1), 4.5),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
          ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), INK)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if (i % 2 == 0):
                st.append(("BACKGROUND", (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(st))
    out = [t]
    if caption:
        C.tab += 1
        out = [Paragraph(f"<b>Table {C.tab}.</b> {caption}", S["cap"]), t, Spacer(1, 10)]
        out = [KeepTogether(out[:2])] + out[2:]
    else:
        out.append(Spacer(1, 8))
    return out


def FIG(path, caption: str, width: float = 1.0, max_h: float | None = None):
    from PIL import Image as PILImage

    w_pt = width * TEXT_W
    with PILImage.open(path) as im:
        iw, ih = im.size
    h_pt = w_pt * ih / iw
    if max_h and h_pt > max_h:
        w_pt, h_pt = w_pt * max_h / h_pt, max_h
    C.fig += 1
    img = Image(str(path), width=w_pt, height=h_pt)
    img.hAlign = "CENTER"
    return [KeepTogether([img, Paragraph(f"<b>Figure {C.fig}.</b> {caption}", S["cap"])])]


_EQ_CACHE = Path(tempfile.gettempdir()) / "inferno_eq_cache"
_EQ_CACHE.mkdir(exist_ok=True)


def _render_eq(latex: str, size: float = 12.5) -> tuple[Path, float, float]:
    key = hashlib.md5(f"{latex}|{size}".encode()).hexdigest()
    path = _EQ_CACHE / f"{key}.png"
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${latex}$", fontsize=size, color="#14171c")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=330, bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    from PIL import Image as PILImage

    buf.seek(0)
    with PILImage.open(buf) as im:
        w, h = im.size
        im.save(path)
    return path, w * 72 / 330, h * 72 / 330


def EQ(latex: str, number: bool = True, size: float = 12.5):
    path, w, h = _render_eq(latex, size)
    C.eq += 1
    img = Image(str(path), width=w, height=h)
    label = Paragraph(f"({C.eq})", ParagraphStyle("eqn", fontName="Serif", fontSize=9.4, alignment=2)) if number else ""
    t = Table([[img, label]], colWidths=[TEXT_W - 34, 34])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [t, Spacer(1, 3)]


# --------------------------------------------------------------------------- document template
class ReportDoc(BaseDocTemplate):
    def __init__(self, path, title, short):
        super().__init__(path, pagesize=PAGE, leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM,
                         title=title, author="Inferno project", subject="Technical report")
        self.short = short
        self._bm = 0
        frame = Frame(LM, BM, TEXT_W, PAGE[1] - TM - BM, id="main", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=self._cover),
                               PageTemplate(id="body", frames=[frame], onPage=self._page)])

    def _startBuild(self, filename=None, canvasmaker=None):
        self._bm = 0  # bookmark keys must be identical on every multiBuild pass or the TOC never converges
        super()._startBuild(filename, canvasmaker)

    def _cover(self, canv, doc):
        canv.saveState()
        canv.setFillColor(INK)
        canv.rect(0, PAGE[1] - 14 * mm, PAGE[0], 14 * mm, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, PAGE[1] - 14 * mm, PAGE[0] / 3, 1.6 * mm, stroke=0, fill=1)
        canv.setFillColor(ACCENT2)
        canv.rect(PAGE[0] / 3, PAGE[1] - 14 * mm, PAGE[0] / 3, 1.6 * mm, stroke=0, fill=1)
        canv.setFillColor(WARN)
        canv.rect(2 * PAGE[0] / 3, PAGE[1] - 14 * mm, PAGE[0] / 3, 1.6 * mm, stroke=0, fill=1)
        canv.restoreState()

    def _page(self, canv, doc):
        canv.saveState()
        canv.setFont("Sans", 7.6)
        canv.setFillColor(MUTED)
        canv.drawString(LM, PAGE[1] - 14 * mm, self.short)
        canv.setStrokeColor(RULE)
        canv.line(LM, PAGE[1] - 16 * mm, PAGE[0] - RM, PAGE[1] - 16 * mm)
        canv.drawRightString(PAGE[0] - RM, 12 * mm, f"{doc.page}")
        canv.line(LM, 16 * mm, PAGE[0] - RM, 16 * mm)
        canv.restoreState()

    def afterFlowable(self, fl):
        # a KeepTogether that fits is drawn as one unit, so look inside it for headings
        if isinstance(fl, KeepTogether):
            for f in fl._content:
                self.afterFlowable(f)
            return
        if isinstance(fl, Heading) and fl.level <= 3:
            key = f"h{self._bm}"
            self._bm += 1
            self.canv.bookmarkPage(key)
            label = f"{fl.number} {fl.plain}".strip()
            self.canv.addOutlineEntry(label, key, level=fl.level - 1, closed=fl.level > 1)
            self.notify("TOCEntry", (fl.level - 1, (f"{fl.number}&nbsp;&nbsp;" if fl.number else "") + fl.plain, self.page, key))


def make_toc():
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"], S["toc3"]]
    toc.dotsMinLevel = 0
    return toc
