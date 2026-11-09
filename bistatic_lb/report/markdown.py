"""Write a report as Markdown, with PNG figures in a sibling folder."""

from __future__ import annotations

from pathlib import Path

from .builder import Bullets, Figure, Heading, Paragraph, Report, Table


def _table(block: Table) -> list[str]:
    if not block.rows:
        return []
    columns = list(block.rows[0])
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(escape(row[c]) for c in columns) + " |" for row in block.rows]
    if block.caption:
        lines = [f"*{block.caption}*", ""] + lines
    return lines + [""]


def write_markdown(report: Report, path: str | Path) -> Path:
    path = Path(path)
    images = path.parent / f"{path.stem}_figures"
    images.mkdir(parents=True, exist_ok=True)
    out = [f"# {report.title}", "", f"*{report.subtitle}*", ""]
    for block in report.blocks:
        if isinstance(block, Heading):
            out += [f"{'#' * (block.level + 1)} {block.text}", ""]
        elif isinstance(block, Paragraph):
            out += [block.text, ""]
        elif isinstance(block, Bullets):
            out += [f"- {item}" for item in block.items] + [""]
        elif isinstance(block, Table):
            out += _table(block)
        elif isinstance(block, Figure):
            file = images / f"{block.name}.png"
            block.figure.savefig(file, dpi=160)
            out += [f"![{block.caption}]({images.name}/{file.name})", "", f"*{block.caption}*", ""]
    path.write_text("\n".join(out), encoding="utf-8")
    return path
