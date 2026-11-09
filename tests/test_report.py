"""The reports must contain what the budget computed, with figures in place."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from docx import Document

from bistatic_lb import Scenario, run_link_budget
from bistatic_lb.report import build_report, write_docx, write_markdown
from bistatic_lb.sweeps import sweep
from bistatic_lb.tracker import run_tracker_monte_carlo


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("report")
    s = Scenario()
    s.monte_carlo.runs, s.monte_carlo.scans = 20, 20
    r = run_link_budget(s)
    mc = run_tracker_monte_carlo(r.analyzer, r.detection.pfa)
    report = build_report(r, mc, status="Final", notes="Designer's test note.")
    return r, write_markdown(report, folder / "out.md"), write_docx(report, folder / "out.docx")


def test_markdown_images_exist_and_numbers_match(outputs):
    r, md, _ = outputs
    text = md.read_text(encoding="utf-8")
    images = re.findall(r"!\[.*?\]\((.*?)\)", text)
    assert len(images) >= 10
    for image in images:
        assert (md.parent / image).stat().st_size > 1000
    assert f"Pd {r.detection.pd:.3f}" in text
    assert f"{r.link.sinr_db:.1f} dB" in text
    assert "Designer's test note." in text


def test_markdown_shows_rejected_models_and_reasons(outputs):
    _, md, _ = outputs
    text = md.read_text(encoding="utf-8")
    assert "Lightning" in text and "Rejected" in text
    assert "P.526" in text


def test_docx_has_tables_figures_and_same_numbers(outputs):
    r, _, docx = outputs
    doc = Document(docx)
    assert len(doc.inline_shapes) >= 10
    assert len(doc.tables) >= 8
    cells = " ".join(c.text for t in doc.tables for row in t.rows for c in row.cells)
    assert f"Pd {r.detection.pd:.3f}" in cells
    # Figures must fit inside the page margins.
    usable = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    assert all(shape.width <= usable for shape in doc.inline_shapes)


def test_report_status_says_what_was_verified(outputs):
    _, md, _ = outputs
    text = md.read_text(encoding="utf-8")
    assert "P.372-18" in text and "Equations read from this edition" in text


def test_scenario_files_all_load_and_run():
    files = sorted((Path(__file__).resolve().parents[1] / "scenarios").glob("*.yaml"))
    assert len(files) == 6
    for f in files:
        r = run_link_budget(Scenario.from_yaml(f))
        assert r.frontend.gain_min_db <= r.frontend.configured_gain_db <= r.frontend.gain_max_db, f.name
        assert r.detection.margin_db > 0, f.name


def test_sweep_skips_invalid_values():
    x, y = sweep(Scenario(), "tracker.confirm_n", [2, 4], "Single-look Pd")
    assert y[0] != y[0]  # M = 3 cannot exceed N = 2
    assert y[1] == y[1]
