"""Minimal Markdown -> PDF for REPORT.md using reportlab (no system deps).
Handles: #/##/### headings, **bold**, `code`, *italics*, - bullets, > quotes,
and pipe tables. Uses Arial (Windows) so →, ×, ≈, Δ render. Run:
  python scripts/md_to_pdf.py [in.md] [out.pdf]
"""
from __future__ import annotations

import html
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
)

FONTS = r"C:\Windows\Fonts"


def _register():
    reg = {}
    for name, fn in [("Body", "arial.ttf"), ("Body-Bold", "arialbd.ttf"),
                     ("Body-It", "ariali.ttf"), ("Body-BoldIt", "arialbi.ttf")]:
        p = os.path.join(FONTS, fn)
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont(name, p)); reg[name] = True
    if "Body" in reg:
        pdfmetrics.registerFontFamily(
            "Body", normal="Body",
            bold="Body-Bold" if "Body-Bold" in reg else "Body",
            italic="Body-It" if "Body-It" in reg else "Body",
            boldItalic="Body-BoldIt" if "Body-BoldIt" in reg else "Body")
        return "Body"
    return "Helvetica"  # fallback (Latin only)


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    return s


def build(md_path: str, pdf_path: str):
    base = _register()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName=base, fontSize=9.5, leading=13, alignment=TA_LEFT)
    h1 = ParagraphStyle("h1", parent=body, fontName=base, fontSize=18, leading=22, spaceBefore=6, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=body, fontSize=14, leading=18, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1a3c6e"))
    h3 = ParagraphStyle("h3", parent=body, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3)
    quote = ParagraphStyle("quote", parent=body, leftIndent=10, textColor=colors.HexColor("#555555"), fontSize=9)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11)

    lines = open(md_path, encoding="utf-8").read().split("\n")
    flow = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # tables: block of lines containing '|'
        if "|" in ln and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]):
            block = []
            while i < len(lines) and "|" in lines[i]:
                block.append(lines[i]); i += 1
            rows = []
            for r, raw in enumerate(block):
                if re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", raw):  # separator
                    continue
                cells = [c.strip() for c in raw.strip().strip("|").split("|")]
                rows.append([Paragraph(inline(c), cell) for c in cells])
            if rows:
                t = Table(rows, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                    ("FONTNAME", (0, 0), (-1, 0), base),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                flow.append(t); flow.append(Spacer(1, 6))
            continue
        s = ln.rstrip()
        if not s.strip():
            flow.append(Spacer(1, 4))
        elif s.startswith("### "):
            flow.append(Paragraph(inline(s[4:]), h3))
        elif s.startswith("## "):
            flow.append(Paragraph(inline(s[3:]), h2))
        elif s.startswith("# "):
            flow.append(Paragraph(inline(s[2:]), h1))
        elif s.startswith("> "):
            flow.append(Paragraph(inline(s[2:]), quote))
        elif re.match(r"^\s*[-*] ", s):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*] ", lines[i].rstrip()):
                items.append(ListItem(Paragraph(inline(re.sub(r"^\s*[-*] ", "", lines[i].rstrip())), body)))
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14))
            continue
        else:
            flow.append(Paragraph(inline(s), body))
        i += 1

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm, title="Sanskrit Retrieval Report")
    doc.build(flow)
    print("wrote", pdf_path, f"({os.path.getsize(pdf_path)/1024:.0f} KB)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "report", "REPORT.md")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "report", "REPORT.pdf")
    build(src, out)
