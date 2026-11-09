"""Report generation: one neutral structure, written as Markdown or Word."""

from .builder import Report, build_report, control_parameters
from .docx import write_docx
from .markdown import write_markdown

__all__ = ["Report", "build_report", "control_parameters", "write_docx", "write_markdown"]
