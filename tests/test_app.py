"""Every app page must load with no exception, and inputs must drive the results."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))
PAGES = ["Home.py"] + sorted(p.relative_to(APP).as_posix() for p in (APP / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_exception(page):
    at = AppTest.from_file(str(APP / page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_changing_an_input_changes_the_result():
    at = AppTest.from_file(str(APP / "pages" / "2_Antenna_Array.py"), default_timeout=180).run()
    before = next(m.value for m in at.metric if m.label == "Gain to target")
    field = next(n for n in at.number_input if n.label == "Ring elements")
    field.set_value(32).run()
    assert not at.exception
    after = next(m.value for m in at.metric if m.label == "Gain to target")
    assert before != after


def test_report_page_generates_documents():
    at = AppTest.from_file(str(APP / "pages" / "8_Report.py"), default_timeout=300).run()
    next(b for b in at.button if b.label == "Generate report").click().run()
    assert not at.exception
    assert at.session_state["report_docx"][:2] == b"PK"
    assert at.session_state["report_md"][:2] == b"PK"


def test_sweep_and_tracker_buttons_work():
    at = AppTest.from_file(str(APP / "pages" / "7_Trade_Offs.py"), default_timeout=300).run()
    next(b for b in at.button if b.label == "Run sweep").click().run()
    assert not at.exception, [e.value for e in at.exception]

    at = AppTest.from_file(str(APP / "pages" / "6_Tracker.py"), default_timeout=300).run()
    next(n for n in at.number_input if n.label == "Monte Carlo runs").set_value(5).run()
    next(b for b in at.button if b.label == "Run tracker Monte Carlo").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["tracker_mc"].runs == 5


def test_invalid_input_shows_an_error_and_does_not_crash():
    at = AppTest.from_file(str(APP / "pages" / "6_Tracker.py"), default_timeout=300).run()
    next(n for n in at.number_input if n.label == "Confirm: M detections").set_value(9).run()
    assert not at.exception
    assert at.error


def test_scenario_can_be_saved_and_loaded_again(tmp_path, monkeypatch):
    monkeypatch.setenv("BISTATIC_SAVED_DIR", str(tmp_path))
    sys.modules.pop("common", None)  # re-read the saved-folder setting
    try:
        at = AppTest.from_file(str(APP / "pages" / "1_Scenario.py"), default_timeout=180).run()
        next(t for t in at.text_input if t.label == "Scenario name").set_value("My test design").run()
        next(n for n in at.number_input if n.label == "Baseline (km)").set_value(42.0).run()
        next(b for b in at.button if b.label == "Save").click().run()
        assert not at.exception
        saved = tmp_path / "My_test_design.yaml"
        assert saved.exists()

        home = AppTest.from_file(str(APP / "Home.py"), default_timeout=180).run()
        box = next(s for s in home.selectbox if s.label == "Load a saved or example scenario")
        assert box.options[0] == "My test design (saved, My_test_design.yaml)"  # saved files are listed first
        box.select_index(0).run()
        next(b for b in home.button if b.label == "Load scenario").click().run()
        assert not home.exception
        from bistatic_lb import Scenario

        loaded = Scenario.from_yaml(home.session_state["scenario_yaml"])
        assert loaded.name == "My test design" and loaded.geometry.baseline_km == 42.0
    finally:
        sys.modules.pop("common", None)


def test_single_element_array_loads_every_stage_page():
    from bistatic_lb import Scenario

    yaml_text = Scenario.from_yaml(APP.parent / "scenarios" / "dvbt_uhf.yaml").with_value("array.n_circular", 1).to_yaml()
    for page in ("pages/2_Antenna_Array.py", "pages/5_Target_Detection.py", "pages/6_Tracker.py"):
        at = AppTest.from_file(str(APP / page), default_timeout=180)
        at.session_state["scenario_yaml"] = yaml_text
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])


def test_a_failed_calculation_shows_a_message_not_a_traceback(monkeypatch):
    import common

    def broken(yaml_text):
        raise ValueError("no solution")

    monkeypatch.setattr(common, "_run_cached", broken)
    at = AppTest.from_file(str(APP / "pages" / "1_Scenario.py"), default_timeout=180).run()
    assert not at.exception
    assert any("could not be calculated" in e.value and "no solution" in e.value for e in at.error)


def test_library_edits_are_picked_up_without_a_restart(monkeypatch):
    import common

    AppTest.from_file(str(APP / "pages" / "1_Scenario.py"), default_timeout=180).run()
    package = sys.modules["bistatic_lb"]
    assert hasattr(package, "_app_loaded_stamp")
    # Pretend a library file was saved after the package was loaded.
    monkeypatch.setattr(common, "_library_stamp", lambda: package._app_loaded_stamp + 10.0)
    at = AppTest.from_file(str(APP / "pages" / "1_Scenario.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert sys.modules["bistatic_lb"] is not package
    assert sys.modules["bistatic_lb"]._app_loaded_stamp == package._app_loaded_stamp + 10.0
