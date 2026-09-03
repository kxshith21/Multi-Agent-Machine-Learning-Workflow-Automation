"""
Report Markdown → PDF export.

Pure-Python conversion using reportlab's Platypus (no system dependencies,
works out of the box on Windows). Converts the subset of Markdown produced by
the AgentML report template (headings, bold, inline code, bullet-free prose,
and pipe tables) into a paginated PDF.

Kept in its own module (Rules.md §9 — file-size cap) and lazily imported so the
report pipeline never fails when reportlab is unavailable (Markdown is always
the primary output).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

# reportlab is imported lazily inside build_pdf() so the module can be imported
# even when reportlab is not installed.


def _inline_to_html(text: str) -> str:
    """Convert **bold**, `code`, and *italic* into reportlab mini-HTML."""
    # Escape HTML special characters first (they don't occur in our template, but
    # be safe) — then apply our own markup.
    out = text
    out = out.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Inline code -> monospace font
    out = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', out)
    # Bold
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    # Italic
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", out)
    # Strip emoji / other common non-renderable glyphs (reportlab default fonts
    # have no glyphs for these; they would render as black boxes).
    out = re.sub(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\u2714\u2716\u2764\u2B50\u274C\u2705\u26A0\uFE0F]",
        "",
        out,
    )
    return out


def _parse_table(rows: List[str]) -> List[List[str]]:
    """Convert consecutive pipe-table lines into a list of cell lists."""
    rows = [r.strip().strip("|") for r in rows]
    rows = [r for r in rows if r.strip() or True]
    data: List[List[str]] = []
    for row in rows:
        cells = [c.strip() for c in row.split("|")]
        # Drop an all-dashes separator row
        if all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells if c):
            continue
        data.append(cells)
    return data


def build_pdf(markdown_path: str, pdf_path: str) -> str:
    """
    Render a Markdown report to a paginated PDF via reportlab.

    Returns the PDF path on success, raises on failure (caller decides whether
    to treat it as fatal).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    text = Path(markdown_path).read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="AgentML Experiment Report",
    )

    styles = getSampleStyleSheet()
    h_styles = {
        1: ParagraphStyle("H1", parent=styles["Title"], fontSize=20, spaceAfter=10),
        2: ParagraphStyle("H2", parent=styles["Heading2"], fontSize=15, spaceBefore=12, spaceAfter=6),
        3: ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12.5, spaceBefore=8, spaceAfter=4),
        4: ParagraphStyle("H4", parent=styles["Heading4"], fontSize=11, spaceBefore=6, spaceAfter=3),
    }
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13,
        spaceAfter=6,
    )

    flow: List[object] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    def emit_table(rows: List[str]) -> None:
        data = _parse_table(rows)
        if not data:
            return
        ncols = max(len(r) for r in data)
        norm = [r + [""] * (ncols - len(r)) for r in data]
        table = Table(
            [[Paragraph(_inline_to_html(c or ""), body) for c in row] for row in norm],
            colWidths=None,
            hAlign="LEFT",
        )
        header_bg = colors.HexColor("#3B4C9B")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D4DC")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        flow.append(table)
        flow.append(Spacer(1, 8))

    while i < n:
        line = lines[i]

        # Horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}", line.strip()):
            flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#C9CEDA"), spaceBefore=6, spaceAfter=6))
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = min(len(m.group(1)), 4)
            flow.append(Paragraph(_inline_to_html(m.group(2)), h_styles[level]))
            i += 1
            continue

        # Table block: collect consecutive lines starting with '|'
        if line.startswith("|"):
            block = []
            while i < n and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            emit_table(block)
            continue

        # Blank line -> paragraph separator
        if not line.strip():
            i += 1
            continue

        # Otherwise gather a paragraph until a blank line / special marker
        para_lines = []
        while i < n and lines[i].strip() and not lines[i].startswith("|") and not re.match(r"^#{1,6}\s", lines[i]) and not re.fullmatch(r"-{3,}", lines[i].strip()):
            para_lines.append(lines[i])
            i += 1
        para_text = " ".join(p.strip() for p in para_lines if p.strip())
        if para_text:
            flow.append(Paragraph(_inline_to_html(para_text), body))

    doc.build(flow)
    return pdf_path
