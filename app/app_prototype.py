"""
Prototype: upload runs -> enter metadata -> process -> label each run.

Run with:
    streamlit run app_prototype.py

No column schema is enforced yet. This is a single-flow prototype meant to
validate the workflow, not the final multi-page app structure.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core_functions as core

st.set_page_config(page_title="Sensor Labeling Prototype", layout="wide")

PLUG_OPTIONS = {"No plug": "no_plug", "Partial plug": "partial_plug", "Full plug": "plug"}
LABEL_COLORS = {1: "gold", 2: "orange", 3: "red"}


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

def init_state():
    st.session_state.setdefault("stage", "upload")
    st.session_state.setdefault("runs", [])  # list of dicts, see below
    st.session_state.setdefault("sampling_interval_s", 1)
    st.session_state.setdefault("current_run_idx", 0)


init_state()


def read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


# ---------------------------------------------------------------------------
# Plot helpers (UI-layer only, not reused outside this app)
# ---------------------------------------------------------------------------

def plot_rdd_figure(df: pd.DataFrame) -> go.Figure:
    x = df["run_specific_time_s"]
    y = df["Relative_Ratio_Dev"]
    smoothed = y.rolling(window=10, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Raw Deviation",
                              line=dict(color="skyblue"), opacity=0.5))
    fig.add_trace(go.Scatter(x=x, y=smoothed, mode="lines", name="Smoothed Trend",
                              line=dict(color="darkblue", width=2)))
    fig.update_layout(
        title="Relative Ratio Deviation Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Deviation Magnitude",
        height=450,
    )
    return fig


def plot_label_verification_figure(df: pd.DataFrame) -> go.Figure:
    fig = plot_rdd_figure(df)
    for lbl, color in LABEL_COLORS.items():
        subset = df[df["label"] == lbl]
        if subset.empty:
            continue
        fig.add_vrect(
            x0=subset["run_specific_time_s"].min(),
            x1=subset["run_specific_time_s"].max(),
            fillcolor=color,
            opacity=0.2,
            line_width=0,
            annotation_text=f"Label {lbl}",
            annotation_position="top left",
        )
    fig.update_layout(title="Label Verification")
    return fig


# ---------------------------------------------------------------------------
# Stage: upload
# ---------------------------------------------------------------------------

def stage_upload():
    st.title("1. Upload runs")
    uploaded = st.file_uploader(
        "Upload CSV/Excel files (no column requirements enforced yet)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded:
        # Rebuild the runs list each time so re-uploading reflects current selection
        st.session_state["runs"] = []
        for f in uploaded:
            try:
                raw_df = read_uploaded(f)
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                continue
            st.session_state["runs"].append({
                "filename": f.name,
                "raw_df": raw_df,
                "plug_type": "no_plug",
                "start_s": 0.0,
                "end_s": 0.0,
                "processed_df": None,
                "labeled_df": None,
                "one": 0.0,
                "two": 0.0,
                "three": 0.0,
                "confirmed": False,
            })
        st.success(f"{len(st.session_state['runs'])} file(s) loaded.")
        for run in st.session_state["runs"]:
            st.write(f"**{run['filename']}** — {run['raw_df'].shape[0]} rows, "
                     f"{run['raw_df'].shape[1]} columns")
            with st.expander(f"Columns in {run['filename']}"):
                st.write(list(run["raw_df"].columns))

    if st.session_state["runs"]:
        if st.button("Continue to run metadata ->"):
            st.session_state["stage"] = "metadata"
            st.rerun()


# ---------------------------------------------------------------------------
# Stage: metadata + sampling interval
# ---------------------------------------------------------------------------

def stage_metadata():
    st.title("2. Run metadata")
    st.caption(
        "Start/end are entered as seconds elapsed since the run began, not "
        "clock time or row index. End = 0 means 'to the end of the run'."
    )

    for i, run in enumerate(st.session_state["runs"]):
        with st.expander(run["filename"], expanded=True):
            cols = st.columns(3)
            plug_label = cols[0].selectbox(
                "Plug type", list(PLUG_OPTIONS.keys()),
                index=list(PLUG_OPTIONS.values()).index(run["plug_type"]),
                key=f"plug_{i}",
            )
            run["plug_type"] = PLUG_OPTIONS[plug_label]
            run["start_s"] = cols[1].number_input(
                "Start (s into run)", min_value=0.0, value=run["start_s"], step=1.0, key=f"start_{i}"
            )
            run["end_s"] = cols[2].number_input(
                "End (s into run, 0 = to end)", min_value=0.0, value=run["end_s"], step=1.0, key=f"end_{i}"
            )

    st.divider()
    st.subheader("Sampling settings")
    st.session_state["sampling_interval_s"] = st.number_input(
        "Standardized sampling interval (seconds)",
        min_value=1, value=int(st.session_state["sampling_interval_s"]), step=1,
    )

    col_a, col_b = st.columns(2)
    if col_a.button("<- Back to upload"):
        st.session_state["stage"] = "upload"
        st.rerun()
    if col_b.button("Process all runs ->"):
        run_processing()


def run_processing():
    target_freq = f"{int(st.session_state['sampling_interval_s'])}s"
    errors = []
    for run in st.session_state["runs"]:
        try:
            run["processed_df"] = core.run_pipeline(
                run["raw_df"], run["start_s"], run["end_s"], target_freq
            )
        except Exception as e:
            errors.append((run["filename"], str(e), list(run["raw_df"].columns)))

    if errors:
        st.error("Some runs failed to process:")
        for filename, msg, cols in errors:
            st.write(f"**{filename}**: {msg}")
            with st.expander(f"Columns in {filename}"):
                st.write(cols)
        return

    st.session_state["current_run_idx"] = 0
    st.session_state["stage"] = "label"
    st.rerun()


# ---------------------------------------------------------------------------
# Stage: label (one run at a time)
# ---------------------------------------------------------------------------

def stage_label():
    runs = st.session_state["runs"]
    idx = st.session_state["current_run_idx"]
    run = runs[idx]

    st.title(f"3. Label run {idx + 1} of {len(runs)}: {run['filename']}")
    st.caption(f"Plug type: {run['plug_type']}")

    checks = core.run_checks(run["processed_df"])
    if checks["missing_values"]:
        st.warning(f"Missing values present: {checks['missing_values']}")

    st.plotly_chart(plot_rdd_figure(run["processed_df"]), use_container_width=True)

    st.subheader("Label transitions (seconds into this run, 0 = not reached)")
    cols = st.columns(3)
    run["one"] = cols[0].number_input("0 -> 1", min_value=0.0, value=run["one"], step=1.0, key=f"lbl1_{idx}")
    run["two"] = cols[1].number_input("1 -> 2", min_value=0.0, value=run["two"], step=1.0, key=f"lbl2_{idx}")
    run["three"] = cols[2].number_input("2 -> 3", min_value=0.0, value=run["three"], step=1.0, key=f"lbl3_{idx}")

    if st.button("Preview labels"):
        run["labeled_df"] = core.label_run(run["processed_df"], run["one"], run["two"], run["three"])

    if run["labeled_df"] is not None:
        st.plotly_chart(plot_label_verification_figure(run["labeled_df"]), use_container_width=True)

        col_a, col_b = st.columns(2)
        if col_a.button("<- Back to metadata"):
            st.session_state["stage"] = "metadata"
            st.rerun()
        if col_b.button("Confirm labels & continue ->"):
            run["confirmed"] = True
            if idx + 1 < len(runs):
                st.session_state["current_run_idx"] += 1
            else:
                st.session_state["stage"] = "done"
            st.rerun()
    else:
        st.info("Set the transition points and click 'Preview labels' before confirming.")


# ---------------------------------------------------------------------------
# Stage: done
# ---------------------------------------------------------------------------

def stage_done():
    st.title("4. All runs labeled")
    st.success("Every uploaded run has been processed and labeled.")

    summary = []
    for run in st.session_state["runs"]:
        counts = run["labeled_df"]["label"].value_counts().to_dict() if run["labeled_df"] is not None else {}
        summary.append({
            "filename": run["filename"],
            "plug_type": run["plug_type"],
            **{f"label_{k}_rows": v for k, v in counts.items()},
        })
    st.dataframe(pd.DataFrame(summary))

    for run in st.session_state["runs"]:
        with st.expander(f"Preview: {run['filename']}"):
            st.dataframe(run["labeled_df"].head(20))

    st.caption(
        "Next steps (not built yet): persist annotations + processed data to "
        "disk, then move into dataset windowing and model training."
    )

    if st.button("Start over"):
        st.session_state["runs"] = []
        st.session_state["stage"] = "upload"
        st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

STAGES = {
    "upload": stage_upload,
    "metadata": stage_metadata,
    "label": stage_label,
    "done": stage_done,
}

st.sidebar.title("Progress")
for name in ["upload", "metadata", "label", "done"]:
    marker = "-> " if st.session_state["stage"] == name else "   "
    st.sidebar.write(f"{marker}{name}")

STAGES[st.session_state["stage"]]()
