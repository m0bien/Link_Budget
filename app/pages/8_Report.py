"""Freeze the control parameters and generate the report in Word and Markdown."""

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from common import Scenario, finish_page, scenario, start_page
from bistatic_lb.pipeline import run_link_budget
from bistatic_lb.report import build_report, control_parameters, write_docx, write_markdown

start_page("Report", "Review the control parameters, mark the design final when you are happy with it, and export.")

r = finish_page()
current = scenario()
final_yaml = st.session_state.get("final_yaml")

left, right = st.columns([1, 1])
with left:
    st.subheader("Current control parameters")
    st.dataframe(control_parameters(r), width="stretch", hide_index=True)

with right:
    st.subheader("Finalize")
    if final_yaml is None:
        st.info("No design is finalized. A report now is labelled Draft.")
        if st.button("Mark this design as final", type="primary"):
            st.session_state["final_yaml"] = current.to_yaml()
            st.rerun()
    else:
        if final_yaml == current.to_yaml():
            st.success("This design is final. Reports use these parameters.")
        else:
            st.warning("Inputs have changed since this design was finalized. Reports still use the finalized values, until you update or unlock.")
            if st.button("Replace the final design with the current inputs"):
                st.session_state["final_yaml"] = current.to_yaml()
                st.rerun()
        if st.button("Unlock the final design"):
            del st.session_state["final_yaml"]
            st.rerun()
    notes = st.text_area("Designer's notes for the report", placeholder="Optional")

    if st.button("Generate report"):
        chosen = Scenario.from_yaml(final_yaml) if final_yaml else current
        status = "Final" if final_yaml else "Draft"
        mc = st.session_state.get("tracker_mc") if chosen.to_yaml() == current.to_yaml() else None
        with st.spinner("Building figures and documents"):
            report = build_report(run_link_budget(chosen), mc, status=status, notes=notes)
            with tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                write_docx(report, base / "link_budget.docx")
                write_markdown(report, base / "link_budget.md")
                st.session_state["report_docx"] = (base / "link_budget.docx").read_bytes()
                archive = io.BytesIO()
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(base / "link_budget.md", "link_budget.md")
                    for image in (base / "link_budget_figures").glob("*.png"):
                        z.write(image, f"link_budget_figures/{image.name}")
                st.session_state["report_md"] = archive.getvalue()
        if mc is None:
            st.caption("Tracker Monte Carlo isn't in this report. Run it on the Tracker page first if you want it included.")

    if "report_docx" in st.session_state:
        a, b = st.columns(2)
        a.download_button("Download Word report", st.session_state["report_docx"], "link_budget.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        b.download_button("Download Markdown and figures (.zip)", st.session_state["report_md"], "link_budget_markdown.zip", "application/zip")
