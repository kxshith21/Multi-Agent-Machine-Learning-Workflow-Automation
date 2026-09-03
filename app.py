"""
AgentML — Streamlit Demo UI (Phase 7)
File: app.py

A technical, clear, and data-forward dashboard matching Design.md specifications.
Supports:
  - Off-white base background and clean white panel styling.
  - Custom horizontal 7-node pipeline stepper (Completed, Active, Interrupted, Pending).
  - Collapsible per-agent cards (expanded for currently active step).
  - Amber-accented human-in-the-loop checkpoint interfaces.
  - Factual representation of results and printable report pane with download buttons.
"""

from __future__ import annotations

import os
import uuid
import pandas as pd
import streamlit as st
from langgraph.types import Command

from src.orchestrator.graph import build_graph
from src.orchestrator.state import AgentMLState
from src.utils.safe_formula import FormulaValidationError, validate_formula

# Set page config first
st.set_page_config(
    page_title="AgentML — Workflow Automation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS Styling (Design.md §2, §3, §4)
# ---------------------------------------------------------------------------

def apply_custom_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main Background & Fonts */
    .stApp {
        background-color: #FAFAF8;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, li {
        font-family: 'Inter', sans-serif !important;
        color: #1A1A1A;
    }
    
    /* Secondary/Slate text class */
    .slate-text {
        color: #5C6470;
        font-size: 14px;
        line-height: 1.5;
    }
    
    /* Code / Data Metrics styling */
    code, pre, .mono-text, table, th, td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }
    
    /* Border/Divider style */
    hr {
        border-color: #E4E6EA !important;
    }
    
    /* Custom Card Containers */
    .card-panel {
        background-color: #FFFFFF;
        border: 1px solid #E4E6EA;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Amber Accent Left-Border for Checkpoint Cards */
    .checkpoint-card {
        background-color: #FFFFFF;
        border-left: 5px solid #C77D26;
        border-top: 1px solid #E4E6EA;
        border-right: 1px solid #E4E6EA;
        border-bottom: 1px solid #E4E6EA;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .checkpoint-title {
        color: #C77D26;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 0.5rem;
    }
    
    /* Muted Success green background highlight */
    .success-badge {
        background-color: #E6F4EA;
        color: #2E7D5B;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Expander card background overrides */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E4E6EA !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
        box-shadow: none !important;
    }
    
    /* Report Container style */
    .report-container {
        background-color: #FFFFFF;
        border: 1px solid #E4E6EA;
        border-radius: 8px;
        padding: 2.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stepper Visualisation (Design.md §4, §5)
# ---------------------------------------------------------------------------

# Node registry: (node_name, display_label, detail_key) in execution order
WORKFLOW_NODES = [
    ("dataset_profiling", "Profiling", "dataset_profile"),
    ("problem_detection", "Detection", "detection"),
    ("feature_engineering", "Features", "features"),
    ("data_preprocessing", "Preprocessing", "preprocessing"),
    ("experiment_orchestrator", "Training", "experiments"),
    ("model_evaluation", "Evaluation", "evaluation"),
    ("report_generation", "Report", "report"),
]


def _node_done(state: AgentMLState, node: str) -> bool:
    """Return True if the node has produced its primary output in the state."""
    g = state or {}
    if node == "dataset_profiling":
        return g.get("dataset_profile") is not None
    if node == "problem_detection":
        return g.get("task_type") is not None
    if node == "feature_engineering":
        # The checkpoint node is "done" once selected/custom features exist in
        # state (whether or not anything was chosen — even an empty selection
        # means the user resolved the checkpoint).
        return "selected_features" in g or "custom_features" in g
    if node == "data_preprocessing":
        return bool(g.get("preprocessing_log")) or bool(g.get("clean_dataset_path"))
    if node == "experiment_orchestrator":
        return len(g.get("experiment_results") or []) > 0
    if node == "model_evaluation":
        return len(g.get("ranking") or []) > 0
    if node == "report_generation":
        return g.get("report_path") is not None
    return False


def _node_status(state, node, current_phase, is_interrupted, hard_errors):
    """Return one of: pending | active | warn | error | complete."""
    if any(e.get("phase") == node and not e.get("recoverable", True)
           for e in (state.get("errors") or [])) or (
        node == "problem_detection" and any(
            e.get("error_type") == "instruction_parse_error"
            for e in (state.get("errors") or [])
        ) and not _node_done(state, node)
    ):
        return "error" if not _node_done(state, node) else "complete"

    if _node_done(state, node):
        return "complete"

    if current_phase == node:
        return "warn" if is_interrupted else "active"

    return "pending"


def _node_summary(state, node):
    """One-line human summary used in the status bar for a node."""
    g = state or {}
    if node == "dataset_profiling":
        p = g.get("dataset_profile") or {}
        if p:
            return f"{p.get('num_rows')} rows × {p.get('num_cols')} cols"
        return "shape / dtypes / missing%"
    if node == "problem_detection":
        if g.get("task_type"):
            return f"{str(g.get('task_type')).title()} → `{g.get('target_column') or '—'}`"
        return "target + task type"
    if node == "feature_engineering":
        n_sel = len(g.get("selected_features") or [])
        n_cust = len(g.get("custom_features") or [])
        if n_sel or n_cust:
            return f"{n_sel} suggested + {n_cust} custom"
        return "feature selection"
    if node == "data_preprocessing":
        n = len(g.get("preprocessing_log") or [])
        return f"{n} operations" if n else "clean dataset"
    if node == "experiment_orchestrator":
        n = len(g.get("experiment_results") or [])
        return f"{n} models" if n else "run model zoo"
    if node == "model_evaluation":
        bm = g.get("best_model_id")
        return f"best: {bm}" if bm else "leaderboard"
    if node == "report_generation":
        return "Markdown + PDF" if g.get("report_path") else "full report"
    return ""


STATUS_BADGE = {
    "pending": "○",
    "active": "●",
    "warn": "⚠",
    "error": "✗",
    "complete": "✓",
}


def render_status_bar(state, current_phase, is_interrupted):
    """Render a compact, clickable workflow status bar.

    Each workflow node is a clickable card showing its status badge + a one-line
    summary. Implementation details stay hidden until a node is clicked.
    Returns the currently selected node id.
    """
    hard_errors = [e for e in (state.get("errors") or []) if not e.get("recoverable", True)]

    # Default selection: the active running node while executing, else the
    # first pending node at startup, else the report once complete.
    if "selected_node" not in st.session_state:
        node_names = dict((n, l) for n, l, _ in WORKFLOW_NODES)
        if current_phase in node_names:
            default = current_phase
        else:
            # Not yet executing a specific node (e.g. orchestrator/setup):
            # pick the first node that is pending or currently active.
            default = "report_generation"
            for node, _, _ in WORKFLOW_NODES:
                if _node_status(state, node, current_phase, is_interrupted, []) in ("active", "pending"):
                    default = node
                    break
        st.session_state.selected_node = default

    cols = st.columns(len(WORKFLOW_NODES))
    selected = st.session_state.selected_node

    bar_css = """
    <style>
    .wf-node {
        border: 1px solid #E4E6EA;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 10px 8px;
        text-align: center;
        cursor: pointer;
        transition: all 0.1s ease;
    }
    .wf-node .wf-badge { font-size: 16px; }
    .wf-node .wf-name { font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 600; color: #1A1A1A; margin-top: 4px; }
    .wf-node .wf-summary { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #5C6470; margin-top: 2px; }
    .wf-selected { border: 2px solid #3B4C9B; box-shadow: 0 1px 4px rgba(59,76,155,0.25); }
    .wf-active { border-color: #3B4C9B; }
    .wf-complete { border-color: #2E7D5B; }
    .wf-complete .wf-badge { color: #2E7D5B; }
    .wf-warn { border-color: #C77D26; }
    .wf-warn .wf-badge { color: #C77D26; }
    .wf-error .wf-badge { color: #B42318; }
    </style>
    """
    st.markdown(bar_css, unsafe_allow_html=True)

    for col, (node, label, _) in zip(cols, WORKFLOW_NODES):
        status = _node_status(state, node, current_phase, is_interrupted, hard_errors)
        is_sel = node == selected
        css_cls = f"wf-node wf-{status}" + (" wf-selected" if is_sel else "")
        badge = STATUS_BADGE[status]
        summary = _node_summary(state, node)
        col.markdown(
            f'<div class="{css_cls}" data-node="{node}">'
            f'<div class="wf-badge">{badge}</div>'
            f'<div class="wf-name">{label}</div>'
            f'<div class="wf-summary">{summary}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if col.button("Open", key=f"node_open_{node}", type="secondary", use_container_width=True):
            st.session_state.selected_node = node

    st.caption("Click a node to inspect its output. Implementation details are hidden while agents run.")
    return st.session_state.selected_node


def _render_profile_detail(profile):
    st.markdown(
        f"**Shape:** `{profile.get('num_rows')} rows` × `{profile.get('num_cols')} columns` | "
        f"**Duplicates:** `{profile.get('duplicate_count')} rows`"
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Numeric Columns:**")
        st.write(", ".join(f"`{c}`" for c in profile.get("numeric_cols", [])))
    with col_t2:
        st.markdown("**Categorical Columns:**")
        st.write(", ".join(f"`{c}`" for c in profile.get("categorical_cols", [])))
    st.markdown("**Missing Data Metrics (% per column):**")
    st.json(profile.get("missing_pct", {}))


def render_node_detail(gstate, node, current_phase, is_interrupted):
    """Render the detail panel for a single selected workflow node.

    If the node is currently running (active) but not yet done, show a compact
    'running' status instead of implementation internals. If it has produced
    output, show that node's individual output.
    """
    done = _node_done(gstate, node)
    active = current_phase == node and not is_interrupted

    if not done and not active:
        st.info("Pending execution...")
        return
    if not done and active:
        label = dict((n, l) for n, l, _ in WORKFLOW_NODES).get(node, node)
        st.markdown(f"### {label}")
        st.markdown("**● Running…**")
        st.markdown('<p class="slate-text">This step is executing. Its output will appear here when complete.</p>', unsafe_allow_html=True)
        return

    if node == "dataset_profiling":
        st.markdown("### Dataset Profiling")
        _render_profile_detail(gstate.get("dataset_profile") or {})
    elif node == "problem_detection":
        st.markdown("### Problem Detection")
        user_instr = (gstate.get("user_instruction") or "").strip()
        if user_instr:
            st.markdown(f"> **Your instruction:** _{user_instr}_")
        st.markdown(f"**Task Type:** `{gstate.get('task_type', '').upper()}`")
        st.markdown(f"**Target Column:** `{gstate.get('target_column')}`")
        st.markdown(f"**Confidence:** `{gstate.get('detection_confidence', 0)*100:.1f}%`")
        with st.popover("Why? (Explainable Reasoning)"):
            st.markdown(gstate.get("detection_reasoning", "No explanation logged."))
    elif node == "data_preprocessing":
        st.markdown("### Data Preprocessing")
        st.markdown(f"**Clean Dataset Path:** `{gstate.get('clean_dataset_path')}`")
        log_entries = gstate.get("preprocessing_log", [])
        if log_entries:
            st.table(pd.DataFrame(log_entries))
        else:
            st.info("No preprocessing steps required.")
    elif node == "feature_engineering":
        st.markdown("### Feature Engineering Selection")
        selected = gstate.get("selected_features") or []
        custom = gstate.get("custom_features") or []
        log_entries = gstate.get("feature_engineering_log") or []
        if selected:
            st.markdown("**Selected suggested features:**")
            st.write(", ".join(f"`{s}`" for s in selected))
        else:
            st.markdown("**Selected suggested features:** _none_")
        if custom:
            st.markdown("**Custom features:**")
            for cf in custom:
                st.markdown(f"- `{cf.get('name')}` = `{cf.get('formula')}`")
        st.markdown("**Feature Engineering Log:**")
        if log_entries:
            st.table(pd.DataFrame(log_entries))
        else:
            st.info("No features were created.")
    elif node == "experiment_orchestrator":
        st.markdown("### Model Training (Experiment Orchestrator)")
        results = gstate.get("experiment_results", [])
        st.success(f"Concurrently ran model zoo. {len(results)} experiments logged.")
        st.dataframe(
            pd.DataFrame(results)[["model_name", "success", "runtime_seconds", "error_type", "error_message"]]
        )
    elif node == "model_evaluation":
        st.markdown("### Model Evaluation")
        ranking = gstate.get("ranking", [])
        st.markdown(f"**Top Ranked Model:** `{gstate.get('best_model_id', 'none')}`")
        st.dataframe(
            pd.DataFrame(ranking)[["rank", "model_name", "primary_metric", "primary_value", "runtime_seconds"]]
        )
        with st.popover("Why? (Explainable Evaluation Reasoning)"):
            st.markdown(gstate.get("evaluation_reasoning", "No evaluation reasoning logged."))
    elif node == "report_generation":
        st.markdown("### Report Viewer")
        report_path = gstate.get("report_path")
        if report_path and os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_md = f.read()
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(report_md)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button("Download Markdown Report", data=report_md,
                                   file_name=os.path.basename(report_path), mime="text/markdown",
                                   key=f"dl_md_node_{st.session_state.get('session_id', 'default')}")
            with col_d2:
                pdf_path = report_path.replace(".md", ".pdf")
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF Report", data=f.read(),
                                           file_name=os.path.basename(pdf_path), mime="application/pdf",
                                           key=f"dl_pdf_node_{st.session_state.get('session_id', 'default')}")
        else:
            st.warning("Report file was not found at the expected path.")


def render_workflow_summary(gstate):
    """Show the entire workflow at a glance once the pipeline has finished."""
    st.markdown("### Workflow Complete — Summary")
    rows = []
    for node, label, _ in WORKFLOW_NODES:
        status = _node_status(gstate, node, "report_generation", False, [])
        rows.append({
            "Stage": label,
            "Status": "✓ Done" if status == "complete" else ("Pending" if status == "pending" else "…"),
            "Result": _node_summary(gstate, node),
        })
    st.table(pd.DataFrame(rows))

    p = gstate.get("dataset_profile") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", p.get("num_rows", "—"))
    m2.metric("Columns", p.get("num_cols", "—"))
    m3.metric("Task Type", str(gstate.get("task_type", "—")).title())
    m4.metric("Best Model", gstate.get("best_model_id", "—"))

    st.markdown(f"**Target Column:** `{gstate.get('target_column')}`   "
                f"**Confidence:** `{gstate.get('detection_confidence', 0)*100:.1f}%`")
    report_path = gstate.get("report_path")
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_md = f.read()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("Download Markdown Report", data=report_md,
                               file_name=os.path.basename(report_path), mime="text/markdown",
                               key=f"dl_md_summary_{st.session_state.get('session_id', 'default')}")
        with col_d2:
            pdf_path = report_path.replace(".md", ".pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF Report", data=f.read(),
                                       file_name=os.path.basename(pdf_path), mime="application/pdf",
                                       key=f"dl_pdf_summary_{st.session_state.get('session_id', 'default')}")


# ---------------------------------------------------------------------------
# App Initialization & State Management
# ---------------------------------------------------------------------------

apply_custom_theme()

st.title("AgentML")
st.markdown(
    '<p class="slate-text">Multi-agent machine learning workflow automation. '
    "Transparent preprocessing, problem detection, concurrent training, and evaluation.</p>",
    unsafe_allow_html=True
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.graph = build_graph()
    st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}
    st.session_state.graph_state = None
    st.session_state.active_interrupt = None
    st.session_state.uploaded_file_path = None
    st.session_state.selected_node = None

# Sidebar reset button
with st.sidebar:
    st.subheader("Orchestration")
    if st.button("Reset Worksession", type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}
        st.session_state.graph_state = None
        st.session_state.active_interrupt = None
        st.session_state.uploaded_file_path = None
        st.session_state.selected_node = None
        st.rerun()

# ---------------------------------------------------------------------------
# 1. File Upload Phase
# ---------------------------------------------------------------------------

if st.session_state.graph_state is None:
    st.subheader("1. Select Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Create data directory if missing
        os.makedirs("data", exist_ok=True)
        temp_path = f"data/upload_{st.session_state.session_id}.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = temp_path
        
        # Preview data
        df_preview = pd.read_csv(temp_path)
        st.success(f"Loaded: `{uploaded_file.name}` ({len(df_preview)} rows, {len(df_preview.columns)} columns)")
        st.dataframe(df_preview.head(5))
        
        st.markdown("**Describe what to predict** *(optional)*")
        user_instruction = st.text_area(
            "e.g. 'This is the Titanic dataset; I need the model to predict who survived or not.'",
            height=70,
            label_visibility="collapsed",
        )
        st.caption(
            "The pipeline will auto-detect the prediction target if you leave this blank. "
            "You can still adjust or override it in the detection step before training."
        )

        if st.button("Initialize & Start Pipeline", type="primary"):
            initial_state = {
                "session_id": st.session_state.session_id,
                "raw_file_path": temp_path,
                "user_instruction": user_instruction if user_instruction.strip() else "",
                "target_column": None,
                "errors": [],
                "status": "running",
            }
            
            # Start pipeline run
            with st.spinner("Executing pipeline setup & dataset profiling..."):
                res = st.session_state.graph.invoke(initial_state, st.session_state.config)
                st.session_state.graph_state = res
                st.rerun()

# ---------------------------------------------------------------------------
# Active Pipeline Execution Display
# ---------------------------------------------------------------------------

if st.session_state.graph_state is not None:
    gstate: AgentMLState = st.session_state.graph_state
    
    # Check for interrupts
    is_interrupted = "__interrupt__" in gstate and len(gstate["__interrupt__"]) > 0
    current_phase = gstate.get("current_phase", "orchestrator")
    
    # Render the clickable workflow status bar (hides implementation details
    # until a node is clicked). Returns the currently selected node.
    selected_node = render_status_bar(gstate, current_phase, is_interrupted)
    
    # Extract details
    errors = gstate.get("errors", [])
    hard_errors = [e for e in errors if not e.get("recoverable", True)]
    
    # If hard error occurred, abort and show error panel
    if hard_errors:
        st.error("Pipeline aborted due to a non-recoverable error.")
        for err in hard_errors:
            st.markdown(f"**[{err['phase']}]** {err['message']} *(type: {err['error_type']})*")
        
        if st.button("Upload Another File"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}
            st.session_state.graph_state = None
            st.session_state.active_interrupt = None
            st.session_state.uploaded_file_path = None
            st.session_state.selected_node = None
            st.rerun()
            
        st.stop()

    # -----------------------------------------------------------------------
    # Interactive Checkpoint Cards (Design.md §4, §5)
    # -----------------------------------------------------------------------
    if is_interrupted:
        active_interrupt_payload = gstate["__interrupt__"][-1].value
        msg = active_interrupt_payload.get("message", "")
        
        st.markdown('<div class="checkpoint-card">', unsafe_allow_html=True)
        st.markdown('<div class="checkpoint-title">⚠ Human Action Required</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="slate-text"><strong>Pipeline Intercepted:</strong> {msg}</p>', unsafe_allow_html=True)

        # Checkpoint 0 (4th): Feature Engineering Selection (always offered)
        if "feature_suggestions" in active_interrupt_payload:
            suggestions = active_interrupt_payload["feature_suggestions"] or []
            available_cols = set()
            profile = gstate.get("dataset_profile") or {}
            available_cols.update(profile.get("numeric_cols") or [])
            available_cols.update(profile.get("categorical_cols") or [])

            st.markdown("**Suggested feature-engineering opportunities** (check the ones you want created):")
            checked = []
            if not suggestions:
                st.caption("No obvious feature-engineering opportunities were detected in this dataset.")
            for sug in suggestions:
                on = st.checkbox(
                    sug.get("name", "feature"),
                    value=False,
                    key=f"fe_sug_{sug.get('name')}",
                    help=sug.get("description", ""),
                )
                if on:
                    checked.append(sug.get("name"))
                else:
                    st.caption(sug.get("description", ""))

            st.markdown("**Add a custom feature (optional)**")
            st.caption("Formula may only use `+ - * /`, parentheses, and existing column names. "
                       "Unsafe formulas are rejected and never executed.")
            c_n1, c_n2 = st.columns(2)
            with c_n1:
                custom_name = st.text_input("New column name", placeholder="price_per_sqft", key="fe_custom_name")
            with c_n2:
                custom_formula = st.text_input("Formula", placeholder="price / sqft", key="fe_custom_formula")

            custom_error = None
            custom_features = []
            if custom_name.strip() and custom_formula.strip():
                try:
                    validate_formula(custom_formula, available_cols)
                    custom_features = [{"name": custom_name.strip(), "formula": custom_formula.strip()}]
                except (FormulaValidationError, ValueError) as e:
                    custom_error = str(e)

            if custom_error:
                st.error(f"Formula rejected — nothing will be executed: {custom_error}")

            if st.button("Confirm Feature Selection", type="primary"):
                resume_payload = {
                    "selected_features": checked,
                    "custom_features": custom_features,
                }
                with st.spinner("Applying selected features..."):
                    res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                    st.session_state.graph_state = res
                    st.rerun()

        # Checkpoint 1: Ambiguous Problem Detection Target/Task Type
        elif "detected_target_column" in active_interrupt_payload:
            detected_target = active_interrupt_payload["detected_target_column"]
            detected_task = active_interrupt_payload["detected_task_type"]
            confidence = active_interrupt_payload.get("confidence", 0.0)
            
            st.markdown(f"**Detected Target Column:** `{detected_target}`")
            st.markdown(f"**Detected Task Type:** `{detected_task.upper()}` (Confidence: {confidence*100:.1f}%)")
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                confirmed_target = st.text_input("Override Target Column:", value=detected_target)
            with c_col2:
                confirmed_task = st.selectbox(
                    "Override Task Type:",
                    options=["classification", "regression", "clustering"],
                    index=["classification", "regression", "clustering"].index(detected_task)
                )
                
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                if st.button("Confirm Detection", type="primary"):
                    resume_payload = {
                        "target_column": detected_target,
                        "task_type": detected_task
                    }
                    with st.spinner("Resuming execution..."):
                        res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                        st.session_state.graph_state = res
                        st.rerun()
            with btn_col2:
                if st.button("Apply Override"):
                    resume_payload = {
                        "target_column": confirmed_target,
                        "task_type": confirmed_task
                    }
                    with st.spinner("Applying overrides..."):
                        res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                        st.session_state.graph_state = res
                        st.rerun()

        # Checkpoint 2: Experiment Scope Configuration
        elif "default_scope" in active_interrupt_payload:
            default_scope = active_interrupt_payload["default_scope"]
            models_list = active_interrupt_payload.get("available_models", [])
            
            st.markdown(f"**Available Model Configurations:** `{len(models_list)} models` in the zoo.")
            
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            with col_sc1:
                max_exps = st.slider(
                    "Max Experiments to Run:",
                    min_value=1,
                    max_value=len(models_list),
                    value=default_scope.get("max_experiments", len(models_list))
                )
            with col_sc2:
                max_w = st.slider(
                    "Max Thread Workers (Concurrency):",
                    min_value=1,
                    max_value=8,
                    value=default_scope.get("max_workers", 4)
                )
            with col_sc3:
                time_cap = st.number_input(
                    "Time Cap per Run (seconds, 0 for unlimited):",
                    min_value=0,
                    value=int(default_scope.get("time_cap_seconds", 0))
                )
                
            btn_sc1, btn_sc2 = st.columns([1, 4])
            with btn_sc1:
                if st.button("Confirm Defaults", type="primary"):
                    resume_payload = {
                        "max_experiments": default_scope.get("max_experiments", len(models_list)),
                        "max_workers": default_scope.get("max_workers", 4),
                        "time_cap_seconds": default_scope.get("time_cap_seconds", 0)
                    }
                    with st.spinner("Executing experiments..."):
                        res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                        st.session_state.graph_state = res
                        st.rerun()
            with btn_sc2:
                if st.button("Run Overridden Scope"):
                    resume_payload = {
                        "max_experiments": max_exps,
                        "max_workers": max_w,
                        "time_cap_seconds": time_cap
                    }
                    with st.spinner("Executing experiments..."):
                        res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                        st.session_state.graph_state = res
                        st.rerun()

        # Checkpoint 3: Best Model Leaderboard Pick/Override
        elif "default_best_model_id" in active_interrupt_payload:
            default_best = active_interrupt_payload["default_best_model_id"]
            ranking_list = active_interrupt_payload.get("ranking", [])
            task_t = active_interrupt_payload.get("task_type", "classification")
            
            st.markdown(f"**Leaderboard Pick:** `{default_best}`")
            
            # Show the ranked leaderboard inside the checkpoint card itself!
            if ranking_list:
                df_rank = pd.DataFrame(ranking_list)
                st.dataframe(
                    df_rank[["rank", "model_name", "primary_metric", "primary_value", "runtime_seconds"]]
                )
            
            model_options = [r["model_id"] for r in ranking_list]
            if default_best not in model_options and default_best != "none":
                model_options.insert(0, default_best)
            
            override_model = st.selectbox(
                "Select Override Model:",
                options=model_options,
                index=model_options.index(default_best) if default_best in model_options else 0
            )
            
            btn_ev1, btn_ev2 = st.columns([1, 4])
            with btn_ev1:
                if st.button("Confirm Auto Selection", type="primary"):
                    resume_payload = {"best_model_id": default_best}
                    with st.spinner("Finalizing evaluation..."):
                        res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                        st.session_state.graph_state = res
                        st.rerun()
            with btn_ev2:
                if st.button("Apply Model Override"):
                    resume_payload = {"best_model_id": override_model}
                    with st.spinner("Finalizing evaluation..."):
                        res = st.session_state.graph.invoke(Command(resume=resume_payload), st.session_state.config)
                        st.session_state.graph_state = res
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Selected-node detail panel (implementation details stay hidden until a
    # node is clicked; the active running node is auto-selected).
    # -----------------------------------------------------------------------
    st.subheader("Step Output")

    render_node_detail(gstate, selected_node, current_phase, is_interrupted)

    # -----------------------------------------------------------------------
    # Whole-workflow summary once every step is finished
    # -----------------------------------------------------------------------
    if _node_done(gstate, "report_generation"):
        st.markdown("<hr>", unsafe_allow_html=True)
        render_workflow_summary(gstate)
