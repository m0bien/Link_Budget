"""Shared helpers for the app pages: scenario state, input widgets and the status sidebar.

The scenario lives in st.session_state as YAML text, and every input widget is bound to a
dotted field path. A page is therefore mostly a list of input calls followed by the
library's tables and figures.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bistatic_lb import LinkBudgetResult, Scenario, run_link_budget  # noqa: E402
from bistatic_lb.results import contributions_table  # noqa: E402

SCENARIO_DIR = ROOT / "scenarios"
# Scenarios saved from the app. The environment variable lets tests use a temporary folder.
SAVED_DIR = Path(os.environ.get("BISTATIC_SAVED_DIR", SCENARIO_DIR / "saved"))
STATE_KEY = "scenario_yaml"
WIDGET_PREFIX = "field::"


def scenario() -> Scenario:
    if STATE_KEY not in st.session_state:
        default = SCENARIO_DIR / "dvbt_uhf.yaml"
        st.session_state[STATE_KEY] = default.read_text(encoding="utf-8") if default.exists() else Scenario().to_yaml()
    return Scenario.from_yaml(st.session_state[STATE_KEY])


def replace_scenario(new: Scenario) -> None:
    """Swap the whole scenario, and let every widget pick up its new value."""
    st.session_state[STATE_KEY] = new.to_yaml()
    for key in [k for k in st.session_state if str(k).startswith(WIDGET_PREFIX)]:
        del st.session_state[key]
    st.session_state.pop("tracker_mc", None)


def scenario_files() -> dict[Path, str]:
    """Saved scenarios first, then the examples, each mapped to a label for a select box."""
    files = {}
    for folder, kind in ((SAVED_DIR, "saved"), (SCENARIO_DIR, "example")):
        for p in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
            try:
                files[p] = f"{Scenario.from_yaml(p).name} ({kind}, {p.name})"
            except Exception:  # a broken or outdated file should not stop the app
                files[p] = f"{p.name} ({kind}, cannot be read)"
    return files


def file_stem(name: str) -> str:
    """A safe file name from a scenario name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "scenario"


def save_controls() -> None:
    """Save the current scenario to the scenarios/saved folder, or download it."""
    s = scenario()
    with st.sidebar.expander("Save scenario"):
        # No key, so the suggested name follows a renamed scenario.
        stem = st.text_input("File name", value=file_stem(s.name), help=f"Saved in {SAVED_DIR}")
        target = SAVED_DIR / f"{file_stem(stem)}.yaml"
        exists = target.exists()
        overwrite = st.checkbox(f"Overwrite {target.name}", key="save::overwrite") if exists else True
        if st.button("Save", key="save::button", disabled=not overwrite, type="primary"):
            SAVED_DIR.mkdir(parents=True, exist_ok=True)
            target.write_text(s.to_yaml(), encoding="utf-8")
            st.success(f"Saved {target.name}. Load it again from the Home page.")
        st.download_button(
            "Download YAML", s.to_yaml(), file_name=f"{file_stem(stem)}.yaml", mime="text/yaml",
            key="save::download", on_click="ignore",
        )


def _read(s: Scenario, path: str):
    node = s
    for key in path.split("."):
        node = getattr(node, key)
    return node


def _commit(path: str, value) -> None:
    """Store a changed value. The scenario validators reject values that are out of range."""
    current = scenario()
    if _read(current, path) == value:
        return
    try:
        st.session_state[STATE_KEY] = current.with_value(path, value).to_yaml()
        st.session_state.pop("tracker_mc", None)
    except ValueError as error:
        st.error(f"{path}: {error}")


def number(path: str, label: str, scale: float = 1.0, step: float | None = None, min_value=None, max_value=None, fmt=None, help=None, integer: bool = False):
    """A number input bound to a scenario field. Shown as value / scale, e.g. Hz as MHz."""
    key = WIDGET_PREFIX + path
    stored = _read(scenario(), path)
    if key not in st.session_state:
        st.session_state[key] = int(stored) if integer else float(stored) / scale
    value = st.number_input(label, step=step, min_value=min_value, max_value=max_value, format=fmt, key=key, help=help)
    _commit(path, int(value) if integer else float(value) * scale)
    return value


def choice(path: str, label: str, options: dict, help=None):
    """A select box bound to a scenario field. options maps stored value to display text."""
    key = WIDGET_PREFIX + path
    stored = _read(scenario(), path)
    if key not in st.session_state:
        st.session_state[key] = stored
    value = st.selectbox(label, list(options), format_func=lambda v: options[v], key=key, help=help)
    _commit(path, value)
    return value


def toggle(path: str, label: str, help=None):
    key = WIDGET_PREFIX + path
    if key not in st.session_state:
        st.session_state[key] = bool(_read(scenario(), path))
    value = st.checkbox(label, key=key, help=help)
    _commit(path, bool(value))
    return value


@st.cache_data(show_spinner=False, max_entries=200)
def _run_cached(yaml_text: str):
    return run_link_budget(Scenario.from_yaml(yaml_text))


def result() -> LinkBudgetResult:
    return _run_cached(scenario().to_yaml())


def show(fig) -> None:
    """Show a Matplotlib figure and release its memory."""
    st.pyplot(fig, clear_figure=False, width="stretch")
    plt.close(fig)


def applicability_table(contributions, caption: str | None = None) -> None:
    if caption:
        st.caption(caption)
    st.dataframe(contributions_table(contributions), width="stretch", hide_index=True)


def requirements_table(requirements) -> None:
    status = {True: "Met", False: "NOT MET", None: "-"}
    st.dataframe(
        [{"Requirement": q.item, "Needed": q.required, "This design": q.configured, "Status": status[q.met], "Why": q.rationale} for q in requirements],
        width="stretch",
        hide_index=True,
    )


def status_sidebar(r: LinkBudgetResult) -> None:
    """Live status beside every page, so the effect of a change is always visible."""
    with st.sidebar:
        st.subheader("Design status")
        st.caption(r.scenario.name)
        st.metric("Single-look Pd", f"{r.detection.pd:.3f}", f"{r.detection.margin_db:+.1f} dB margin")
        st.metric("Pd at upper-decile noise", f"{r.detection.pd_upper_decile:.3f}", f"{r.detection.margin_upper_decile_db:+.1f} dB margin")
        st.metric("Gain window", f"{r.frontend.gain_window_db:.1f} dB", "Usable" if r.frontend.shortfall_db == 0 else "EMPTY", delta_color="normal" if r.frontend.shortfall_db == 0 else "inverse")
        st.metric("Track confirmation", f"{r.tracker.confirm_probability_at_relaxed_threshold:.3f}", f"{r.tracker.trade.gain_db:+.1f} dB from tracker")
        st.metric("SINR", f"{r.link.sinr_db:.1f} dB")
        if r.warnings:
            st.warning(f"{len(r.warnings)} model warning(s)")
            for w in r.warnings:
                st.caption(w)


def start_page(title: str, intro: str | None = None) -> LinkBudgetResult | None:
    st.set_page_config(page_title=f"{title} | Bi-static link budget", layout="wide")
    # Engineering values are long, so keep metric text compact enough not to be cut off.
    st.markdown(
        "<style>[data-testid='stMetricValue'] {font-size: 1.35rem;} "
        "[data-testid='stMetricValue'] > div {white-space: normal; overflow-wrap: anywhere;}</style>",
        unsafe_allow_html=True,
    )
    st.title(title)
    if intro:
        st.write(intro)
    return None


def finish_page() -> LinkBudgetResult:
    """Run the budget after a page's inputs have been read, and draw the status."""
    r = result()
    save_controls()
    status_sidebar(r)
    return r
