#!/usr/bin/env python3
"""Build the Songo SRN documentation PDF from the local HTML source."""

from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, "/private/tmp/songo_pdf_deps")

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Documentation_Modele_IA_Songo_SRN.source.html"
OUTPUT = ROOT / "Documentation_Modele_IA_Songo_SRN.pdf"


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def inline_text(node: ET.Element) -> str:
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        if el.text:
            parts.append(html.escape(el.text))
        for child in list(el):
            tag = child.tag.lower()
            before = len(parts)
            if tag in {"strong", "b"}:
                parts.append("<b>")
                walk(child)
                parts.append("</b>")
            elif tag in {"em", "i"}:
                parts.append("<i>")
                walk(child)
                parts.append("</i>")
            elif tag == "code":
                parts.append('<font name="Courier">')
                walk(child)
                parts.append("</font>")
            else:
                walk(child)
            if child.tail:
                parts.append(html.escape(child.tail))
            if len(parts) == before and child.tail:
                continue

    walk(node)
    return compact("".join(parts))


def node_plain_text(node: ET.Element) -> str:
    return compact("".join(node.itertext()))


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=colors.HexColor("#102033"),
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#52616f"),
            spaceAfter=18,
        ),
        "kicker": ParagraphStyle(
            "Kicker",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#52616f"),
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#102033"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#102033"),
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.2,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=5.5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#52616f"),
        ),
        "bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            leftIndent=10,
            firstLineIndent=0,
            textColor=colors.HexColor("#1f2933"),
        ),
        "pre": ParagraphStyle(
            "PreCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=9.0,
            textColor=colors.HexColor("#1f2933"),
            backColor=colors.HexColor("#f5f7fa"),
            borderColor=colors.HexColor("#d7dee8"),
            borderWidth=0.6,
            borderPadding=5,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.8,
            textColor=colors.HexColor("#1f2933"),
            backColor=colors.HexColor("#f7fafc"),
            borderColor=colors.HexColor("#2f6f9f"),
            borderWidth=0.8,
            borderPadding=6,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#52616f"),
            spaceAfter=3,
        ),
    }
    return styles


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#7b8794"))
    canvas.drawString(16 * mm, 10 * mm, "SRN - Songo Relational Network")
    canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def table_from_node(table_node: ET.Element, styles) -> Table:
    rows = []
    for tr in table_node.findall("./tr"):
        row = []
        for cell in list(tr):
            txt = inline_text(cell)
            style = styles["body"]
            row.append(Paragraph(txt, style))
        if row:
            rows.append(row)
    if not rows:
        rows = [[""]]
    col_count = max(len(row) for row in rows)
    for row in rows:
        while len(row) < col_count:
            row.append("")
    widths = [None] * col_count
    table = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d7dee8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#102033")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def list_from_node(list_node: ET.Element, styles) -> ListFlowable:
    items = []
    for li in list_node.findall("./li"):
        txt = inline_text(li)
        if txt:
            items.append(ListItem(Paragraph(txt, styles["bullet"]), leftIndent=12))
    return ListFlowable(items, bulletType="bullet", start="circle", leftIndent=12)


def story_from_html() -> list:
    raw = SOURCE.read_text(encoding="utf-8")
    match = re.search(r"<body[^>]*>(.*)</body>", raw, flags=re.I | re.S)
    if not match:
        raise RuntimeError("HTML source has no body")
    root = ET.fromstring(f"<root>{match.group(1)}</root>")
    styles = make_styles()
    story: list = []
    ended_with_pagebreak = False

    for section_index, section in enumerate(root.findall("./section")):
        classes = section.attrib.get("class", "")
        if section_index and "pagebreak" in classes and not ended_with_pagebreak:
            story.append(PageBreak())
            ended_with_pagebreak = True
        elif section_index:
            story.append(Spacer(1, 4))
            ended_with_pagebreak = False

        is_cover = "cover" in classes
        for child in list(section):
            tag = child.tag.lower()
            cls = child.attrib.get("class", "")
            if tag == "h1":
                story.append(Paragraph(inline_text(child), styles["title"]))
            elif tag == "h2":
                story.append(Paragraph(inline_text(child), styles["h2"]))
            elif tag == "h3":
                story.append(Paragraph(inline_text(child), styles["h3"]))
            elif tag == "p":
                style = styles["kicker"] if "kicker" in cls else styles["subtitle"] if "subtitle" in cls else styles["body"]
                story.append(Paragraph(inline_text(child), style))
            elif tag in {"ul", "ol"}:
                story.append(list_from_node(child, styles))
                story.append(Spacer(1, 3))
            elif tag == "pre":
                story.append(Preformatted(child.text or "", styles["pre"], maxLineLength=92))
            elif tag == "table":
                story.append(table_from_node(child, styles))
                story.append(Spacer(1, 4))
            elif tag == "div":
                if "toc" in cls:
                    items = [Paragraph(node_plain_text(p), styles["body"]) for p in child.findall("./p")]
                    story.append(KeepTogether(items[:10]))
                    story.append(KeepTogether(items[10:]))
                elif "callout" in cls:
                    for p in child.findall(".//p"):
                        story.append(Paragraph(inline_text(p), styles["callout"]))
                elif "meta" in cls:
                    story.append(Spacer(1, 330 if is_cover else 10))
                    for p in child.findall("./p"):
                        story.append(Paragraph(inline_text(p), styles["cover_meta"]))
        if is_cover:
            story.append(PageBreak())
            ended_with_pagebreak = True
        else:
            ended_with_pagebreak = False
    return story


def build() -> None:
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Documentation de conception du modele IA Songo SRN",
        author="Projet Songo",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    doc.build(story_from_html())


if __name__ == "__main__":
    build()
