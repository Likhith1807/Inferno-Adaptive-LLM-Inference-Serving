"""
Small drawing kit on top of python-pptx for the mid-term deck: slide cloning that preserves the university template
chassis (logo, title rule, footer), text/cards/badges/arrows, native (editable) charts and tables.
All coordinates are in inches on the 13.333 x 7.5 in slide.
"""
from __future__ import annotations

import copy

from lxml import etree
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ---- palette: the template's own crimson / navy / green, plus neutrals
CRIMSON = RGBColor(0xA5, 0x00, 0x21)
DEEP = RGBColor(0x82, 0x00, 0x19)
NAVY = RGBColor(0x00, 0x20, 0x60)
GREEN = RGBColor(0x00, 0x9E, 0x5A)
AMBER = RGBColor(0xE0, 0x8A, 0x00)
INK = RGBColor(0x1F, 0x23, 0x2B)
GREY = RGBColor(0x5B, 0x63, 0x70)
LINE = RGBColor(0xD5, 0xD9, 0xE0)
PALE = RGBColor(0xF4, 0xF5, 0xF8)
TINT = RGBColor(0xFB, 0xEA, 0xED)
GREENTINT = RGBColor(0xE6, 0xF6, 0xEE)
AMBERTINT = RGBColor(0xFF, 0xF3, 0xDC)
BLUE = RGBColor(0x2A, 0x6F, 0xC9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"


# --------------------------------------------------------------------------- slide handling
def clone_slide(prs, src):
    """New slide with the same layout and a deep copy of every shape (title, rule, logo, footer) of `src`."""
    new = prs.slides.add_slide(src.slide_layout)
    for shp in list(new.shapes):
        shp._element.getparent().remove(shp._element)
    tree = new.shapes._spTree
    for el in src.shapes._spTree:
        if el.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
            continue
        tree.append(copy.deepcopy(el))
    rid_map = {rel.rId: new.part.relate_to(rel.target_part, RT.IMAGE) for rel in src.part.rels.values() if rel.reltype == RT.IMAGE}
    for blip in tree.iter(qn("a:blip")):
        old = blip.get(qn("r:embed"))
        if old in rid_map:
            blip.set(qn("r:embed"), rid_map[old])
    return new


def delete_slide(prs, slide):
    lst = prs.slides._sldIdLst
    for sld in list(lst):
        if prs.part.related_part(sld.rId) is slide.part:
            prs.part.drop_rel(sld.rId)
            lst.remove(sld)
            return


def reorder(prs, slides_in_order):
    lst = prs.slides._sldIdLst
    by_part = {prs.part.related_part(s.rId): s for s in list(lst)}
    for s in list(lst):
        lst.remove(s)
    for sl in slides_in_order:
        lst.append(by_part[sl.part])


def set_title(slide, text):
    """Keep the template's title text box (Times New Roman 32 bold) and replace its text; drops the '(Font size 24)' hint run."""
    for shp in slide.shapes:
        if shp.name == "TextBox 7":
            runs = shp.text_frame.paragraphs[0].runs
            runs[0].text = text
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
            for extra in shp.text_frame.paragraphs[1:]:
                extra._p.getparent().remove(extra._p)
            runs[0].font.bold = True
            return shp
    raise KeyError("title box not found")


def remove_shapes(slide, names):
    for shp in list(slide.shapes):
        if shp.name in names:
            shp._element.getparent().remove(shp._element)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


# --------------------------------------------------------------------------- text
def _apply_run(r, size, bold, color, italic=False, font=FONT):
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = font
    if color is not None:
        r.font.color.rgb = color


def _bullet(p, char="•", indent=0.28, color=CRIMSON):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buClr"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    clr = etree.SubElement(pPr, qn("a:buClr"))
    etree.SubElement(clr, qn("a:srgbClr")).set("val", str(color))
    etree.SubElement(pPr, qn("a:buChar")).set("char", char)


def text(slide, x, y, w, h, paras, size=18, color=INK, bold=False, align="l", anchor="t", bullets=False, space_after=6,
         line_spacing=1.0, margin=0.06, font=FONT, name=None):
    """paras: str | list[str | list[(text, dict)]]. A run dict may set bold / color / size / italic."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    fill_frame(tb.text_frame, paras, size, color, bold, align, anchor, bullets, space_after, line_spacing, margin, font)
    return tb


def fill_frame(tf, paras, size, color, bold, align, anchor, bullets, space_after, line_spacing, margin, font):
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin * 0.6)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for i, item in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        runs = [(item, {})] if isinstance(item, str) else item
        for t, o in runs:
            r = p.add_run()
            r.text = t
            _apply_run(r, o.get("size", size), o.get("bold", bold), o.get("color", color), o.get("italic", False), font)
        if bullets:
            _bullet(p)


# --------------------------------------------------------------------------- shapes
def _no_shadow(shp):
    shp.shadow.inherit = False


def box(slide, x, y, w, h, fill=WHITE, line=LINE, line_w=1.0, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE, gradient=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if gradient is not None:
        s.fill.gradient()
        s.fill.gradient_angle = 0
        s.fill.gradient_stops[0].color.rgb = gradient[0]
        s.fill.gradient_stops[1].color.rgb = gradient[1]
    elif fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    _no_shadow(s)
    return s


def card(slide, x, y, w, h, fill=WHITE, accent=None, line=LINE, radius=0.05):
    """Rounded card with an optional coloured accent bar on its left edge."""
    c = box(slide, x, y, w, h, fill, line, 1.0, radius)
    if accent is not None:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + 0.08), Inches(0.09), Inches(h - 0.16))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent
        bar.line.fill.background()
        _no_shadow(bar)
    return c


def labelled(slide, x, y, w, h, paras, fill=WHITE, line=LINE, size=16, color=INK, bold=False, align="l", anchor="m", accent=None,
             bullets=False, space_after=4, margin=0.14, radius=0.05):
    """A card whose text lives in the shape itself (so it moves and scales with it)."""
    c = card(slide, x, y, w, h, fill, accent, line, radius)
    left = margin + (0.1 if accent is not None else 0)
    fill_frame(c.text_frame, paras, size, color, bold, align, anchor, bullets, space_after, 1.0, left, FONT)
    c.text_frame.margin_right = Inches(margin)
    return c


def picture(slide, path, x, y, w=None, h=None, frame=LINE, radius=0.04, pad=0.05):
    """Embeds a real screenshot/plot with a thin frame. Give w or h (or both); aspect ratio fills in the other."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        iw, ih = im.size
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    if frame is not None:
        box(slide, x - pad, y - pad, w + 2 * pad, h + 2 * pad, fill=WHITE, line=frame, line_w=1.0, radius=radius)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def badge(slide, x, y, d, label, fill=CRIMSON, color=WHITE, size=14):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = WHITE
    s.line.width = Pt(1.5)
    _no_shadow(s)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _apply_run(r, size, True, color)
    return s


def chip(slide, x, y, w, h, label, fill=GREENTINT, color=GREEN, size=13, bold=True):
    s = box(slide, x, y, w, h, fill, None, radius=0.5)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    _apply_run(r, size, bold, color)
    return s


def arrow(slide, x1, y1, x2, y2, color=GREY, width=2.25, dash=False, head=True):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dash:
        etree.SubElement(ln, qn("a:prstDash")).set("val", "dash")
    if head:
        te = etree.SubElement(ln, qn("a:tailEnd"))
        te.set("type", "triangle")
        te.set("w", "med")
        te.set("len", "med")
    return c


def line(slide, x1, y1, x2, y2, color=LINE, width=1.0):
    return arrow(slide, x1, y1, x2, y2, color, width, head=False)


# --------------------------------------------------------------------------- charts
def _style_axis(ax, size, color=GREY, grid=False, gridcolor=LINE):
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.name = FONT
    ax.tick_labels.font.color.rgb = color
    ax.format.line.color.rgb = LINE
    ax.has_major_gridlines = grid
    if grid:
        ax.major_gridlines.format.line.color.rgb = gridcolor
        ax.major_gridlines.format.line.width = Pt(0.5)


def bar_chart(slide, x, y, w, h, categories, series, colors, horizontal=False, number_format='0.0', font=13, legend=False,
              gap=60, overlap=None, vmax=None, vmin=0, point_colors=None, labels=True, log=False, label_color=INK, axis_title=None):
    """series: {name: [values]}; colors: one RGB per series; point_colors: {series_idx: [rgb per point]}."""
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series.items():
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(font)
    ch.font.name = FONT
    ch.has_title = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(font)
    plot = ch.plots[0]
    plot.gap_width = gap
    if overlap is not None:
        plot.overlap = overlap
    va, ca = ch.value_axis, ch.category_axis
    _style_axis(va, font - 1, grid=True)
    _style_axis(ca, font)
    ca.format.line.color.rgb = GREY
    va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.tick_labels.number_format = number_format
    va.tick_labels.number_format_is_linked = False
    if axis_title:
        va.has_title = True
        va.axis_title.text_frame.text = axis_title
        r = va.axis_title.text_frame.paragraphs[0].runs[0]
        r.font.size = Pt(font - 1)
        r.font.bold = False
        r.font.color.rgb = GREY
    if log:
        sc = va._element.find(qn("c:scaling"))
        lb = etree.Element(qn("c:logBase"))
        lb.set("val", "10")
        sc.insert(0, lb)
    for i, s in enumerate(plot.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = colors[i]
        s.format.line.fill.background()
        s.invert_if_negative = False
        if point_colors and i in point_colors:
            for j, col in enumerate(point_colors[i]):
                pt = s.points[j]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = col
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = number_format
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(font)
        dl.font.bold = True
        dl.font.name = FONT
        dl.font.color.rgb = label_color
    return ch


# --------------------------------------------------------------------------- tables
def _cell_border(cell, color="D5D9E0", w=9525, sides=("b",)):
    tcPr = cell._tc.get_or_add_tcPr()
    for side in ("l", "r", "t", "b"):
        tag = qn(f"a:ln{side.upper()}")
        for e in tcPr.findall(tag):
            tcPr.remove(e)
        ln = etree.SubElement(tcPr, tag)
        if side in sides:
            ln.set("w", str(w))
            sf = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr")).set("val", color)
        else:
            ln.set("w", "0")
            etree.SubElement(ln, qn("a:noFill"))
    # schema order: borders must precede the fill
    fills = [e for e in tcPr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]
    for f in fills:
        tcPr.remove(f)
        tcPr.append(f)


def table(slide, x, y, w, col_w, rows, row_h, header_size=16, size=15, header_fill=CRIMSON, first_col_bold=True, zebra=True,
          first_col_color=NAVY, valign="m"):
    """rows[0] is the header. Cell text may be str or list of (text, opts) runs."""
    n_r, n_c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("bandRow", "firstRow"):
        tblPr.set(a, "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # No Style, No Grid
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i in range(n_r):
        tbl.rows[i].height = Inches(row_h if i else max(0.5, row_h * 0.8))
        for j in range(n_c):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = {"m": MSO_ANCHOR.MIDDLE, "t": MSO_ANCHOR.TOP}[valign]
            tf = cell.text_frame
            tf.word_wrap = True
            item = rows[i][j]
            runs = [(item, {})] if isinstance(item, str) else item
            p = tf.paragraphs[0]
            for t, o in runs:
                r = p.add_run()
                r.text = t
                if i == 0:
                    _apply_run(r, header_size, True, WHITE)
                else:
                    _apply_run(r, o.get("size", size), o.get("bold", first_col_bold and j == 0), o.get("color", first_col_color if j == 0 else INK),
                               o.get("italic", False))
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (PALE if (zebra and i % 2 == 0) else WHITE)
            _cell_border(cell, sides=() if i == 0 else ("b",))
    return gf
