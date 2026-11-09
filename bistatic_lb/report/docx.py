"""Write a report as a Word document, with figures embedded."""

from __future__ import annotations

import io
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .builder import Bullets, Figure, Heading, Paragraph, Report, Table

HEADER_FILL = "E8EEF7"
NOT_MET_COLOR = RGBColor(0xB0, 0x20, 0x20)


def _shade(cell, fill: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)


def _repeat_header(row) -> None:
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    row._tr.get_or_add_trPr().append(marker)


def _write_cell(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(text)
    run.font.size = Pt(8.5)
    run.bold = bold
    if text == "NOT MET":
        run.bold = True
        run.font.color.rgb = NOT_MET_COLOR


def _add_table(doc, block: Table, usable_cm: float) -> None:
    if not block.rows:
        return
    if block.caption:
        caption = doc.add_paragraph(block.caption, style="Caption")
        caption.paragraph_format.keep_with_next = True
    columns = list(block.rows[0])
    table = doc.add_table(rows=1, cols=len(columns))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Share the page width by each column's longest cell, within limits.
    weights = []
    for c in columns:
        longest = max(len(str(c)), *(len(str(row[c])) for row in block.rows))
        weights.append(min(max(longest, 8), 70))
    widths = [usable_cm * w / sum(weights) for w in weights]

    for i, c in enumerate(columns):
        cell = table.rows[0].cells[i]
        _write_cell(cell, str(c), bold=True)
        _shade(cell, HEADER_FILL)
        cell.width = Cm(widths[i])
    _repeat_header(table.rows[0])
    for row in block.rows:
        cells = table.add_row().cells
        for i, c in enumerate(columns):
            _write_cell(cells[i], str(row[c]))
            cells[i].width = Cm(widths[i])
    doc.add_paragraph()


def write_docx(report: Report, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    # A4 landscape leaves room for the wide applicability tables.
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(1.8))
    usable_cm = 29.7 - 3.6

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)

    doc.add_heading(report.title, 0)
    doc.add_paragraph(report.subtitle)
    for block in report.blocks:
        if isinstance(block, Heading):
            doc.add_heading(block.text, block.level)
        elif isinstance(block, Paragraph):
            doc.add_paragraph(block.text)
        elif isinstance(block, Bullets):
            for item in block.items:
                doc.add_paragraph(item, style="List Bullet")
        elif isinstance(block, Table):
            _add_table(doc, block, usable_cm)
        elif isinstance(block, Figure):
            buffer = io.BytesIO()
            block.figure.savefig(buffer, format="png", dpi=160)
            buffer.seek(0)
            picture = doc.add_paragraph()
            picture.paragraph_format.keep_with_next = True
            picture.add_run().add_picture(buffer, width=Cm(16.5))
            doc.add_paragraph(block.caption, style="Caption")
    doc.save(path)
    return path
