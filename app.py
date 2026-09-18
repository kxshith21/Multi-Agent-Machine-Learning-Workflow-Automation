"""
AgentML — Multi-Agent ML Experiment Orchestrator UI
File: app.py

A modern, high-contrast, structured Streamlit interface for the 7-agent LangGraph ML pipeline.
Features:
  - 3-Tab workflow layout: 1. Upload & Setup | 2. Pipeline & Agents | 3. Results & Final Report
  - 7-Stage visual progress stepper with status indicators (Pending, Running, Complete, Checkpoint, Failed)
  - Rich structured agent output cards using st.metric, st.dataframe, and formatted text (no raw JSON dumps)
  - Dedicated "Review & Approve" human-in-the-loop checkpoint cards for all 4 LangGraph interrupts
  - Collapsible developer inspect panes for raw state JSON payloads
  - Clean error banners with recovery actions
  - Full rendered Markdown & PDF report viewer with download actions
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

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgentML — Multi-Agent ML Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
def apply_custom_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12.5px !important;
    }

    /* Main Container Polish */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Stepper Styling */
    .stepper-container {
        display: flex;
        justify-content: space-between;
        align-items: stretch;
        gap: 8px;
        margin-bottom: 1.25rem;
        padding: 6px;
        background: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
    }

    .step-box {
        flex: 1;
        padding: 10px 8px;
        border-radius: 8px;
        text-align: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        transition: all 0.15s ease-in-out;
    }

    .step-box.step-pending {
        border-color: #E2E8F0;
        background: #FFFFFF;
        opacity: 0.75;
    }

    .step-box.step-running {
        border-color: #3182CE;
        background: #EBF8FF;
        box-shadow: 0 0 0 1px #3182CE;
    }

    .step-box.step-complete {
        border-color: #38A169;
        background: #F0FFF4;
    }

    .step-box.step-checkpoint {
        border-color: #D69E2E;
        background: #FFFFF0;
        box-shadow: 0 0 0 1px #D69E2E;
    }

    .step-box.step-failed {
        border-color: #E53E3E;
        background: #FFF5F5;
    }

    .step-icon {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .step-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #2D3748;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .step-subtext {
        font-size: 10px;
        color: #718096;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }

    /* Human Action Checkpoint Banner */
    .checkpoint-banner {
        background-color: #FFFDF5;
        border-left: 6px solid #D69E2E;
        border-top: 1px solid #F6E05E;
        border-right: 1px solid #F6E05E;
        border-bottom: 1px solid #F6E05E;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }

    .checkpoint-header {
        font-size: 16px;
        font-weight: 700;
        color: #975A16;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }

    .checkpoint-desc {
        color: #4A5568;
        font-size: 13.5px;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    /* Report Container */
    .report-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        line-height: 1.65;
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-pill-blue { background: #EBF8FF; color: #2B6CB0; border: 1px solid #BEE3F8; }
    .badge-pill-green { background: #F0FFF4; color: #276749; border: 1px solid #C6F6D5; }
    .badge-pill-amber { background: #FFFFF0; color: #975A16; border: 1px solid #FEFCBF; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 7-Agent Node Registry & Helpers
# ---------------------------------------------------------------------------
WORKFLOW_NODES = [
    ("dataset_profiling", "1. Profiling", "Data profiling & feature suggestions"),
    ("problem_detection", "2. Detection", "Task & target column detection"),
    ("feature_engineering", "3. Features", "Feature engineering selection"),
    ("data_preprocessing", "4. Preprocessing", "Cleaning, encoding & scaling"),
    ("experiment_orchestrator", "5. Training", "Concurrent model zoo execution"),
    ("model_evaluation", "6. Evaluation", "Leaderboard ranking & selection"),
    ("report_generation", "7. Report", "Markdown & PDF report generation"),
]


def _is_node_complete(state: AgentMLState, node_name: str) -> bool:
    """Check if the node has produced its required artifacts in state."""
    g = state or {}
    if node_name == "dataset_profiling":
        return g.get("dataset_profile") is not None
    if node_name == "problem_detection":
        return g.get("task_type") is not None
    if node_name == "feature_engineering":
        return "selected_features" in g or "custom_features" in g or g.get("clean_dataset_path") is not None
    if node_name == "data_preprocessing":
        return bool(g.get("preprocessing_log")) or bool(g.get("clean_dataset_path"))
    if node_name == "experiment_orchestrator":
        return len(g.get("experiment_results") or []) > 0
    if node_name == "model_evaluation":
        return len(g.get("ranking") or []) > 0
    if node_name == "report_generation":
        return g.get("report_path") is not None
    return False


def _get_node_status(state: AgentMLState, node_name: str, current_phase: str, is_interrupted: bool) -> str:
    """Returns one of: complete | running | checkpoint | failed | pending."""
    g = state or {}
    errors = g.get("errors") or []
    
    # Check for hard errors belonging to this node
    if any(e.get("phase") == node_name and not e.get("recoverable", True) for e in errors):
        if not _is_node_complete(g, node_name):
            return "failed"

    if _is_node_complete(g, node_name):
        return "complete"

    if current_phase == node_name or (node_name == "feature_engineering" and is_interrupted and ("feature_suggestions" in (g.get("__interrupt__", [{}])[-1].value if is_interrupted else {}))):
        return "checkpoint" if is_interrupted else "running"

    return "pending"


def _get_node_summary_text(state: AgentMLState, node_name: str) -> str:
    """One-line concise summary string for the stepper."""
    g = state or {}
    if node_name == "dataset_profiling":
        p = g.get("dataset_profile") or {}
        if p:
            return f"{p.get('num_rows', 0)}r × {p.get('num_cols', 0)}c"
        return "Schema & Missing"
    if node_name == "problem_detection":
        tt = g.get("task_type")
        if tt:
            return f"{str(tt).title()}"
        return "Target & Type"
    if node_name == "feature_engineering":
        sel = len(g.get("selected_features") or [])
        cust = len(g.get("custom_features") or [])
        if sel or cust:
            return f"{sel + cust} engineered"
        return "Feature Selection"
    if node_name == "data_preprocessing":
        log_len = len(g.get("preprocessing_log") or [])
        return f"{log_len} steps applied" if log_len else "Clean & Encode"
    if node_name == "experiment_orchestrator":
        exps = len(g.get("experiment_results") or [])
        return f"{exps} models trained" if exps else "Model Zoo"
    if node_name == "model_evaluation":
        bm = g.get("best_model_id")
        return f"Best: {bm}" if bm else "Leaderboard"
    if node_name == "report_generation":
        return "Report Ready" if g.get("report_path") else "Final Summary"
    return ""


# ---------------------------------------------------------------------------
# UI Components: Horizontal Stepper
# ---------------------------------------------------------------------------
def render_pipeline_stepper(state: AgentMLState, current_phase: str, is_interrupted: bool):
    """Renders the top 7-stage visual pipeline progress stepper."""
    status_icons = {
        "pending": ("○", "step-pending"),
        "running": ("●", "step-running"),
        "checkpoint": ("⚠", "step-checkpoint"),
        "complete": ("✓", "step-complete"),
        "failed": ("✗", "step-failed"),
    }

    cols = st.columns(len(WORKFLOW_NODES))
    completed_count = 0

    for col, (node_name, label, _) in zip(cols, WORKFLOW_NODES):
        status = _get_node_status(state, node_name, current_phase, is_interrupted)
        icon, css_class = status_icons[status]
        subtext = _get_node_summary_text(state, node_name)
        if status == "complete":
            completed_count += 1

        col.markdown(
            f"""
            <div class="step-box {css_class}">
                <div class="step-icon">{icon}</div>
                <div class="step-label">{label}</div>
                <div class="step-subtext">{subtext}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Progress bar indicator
    progress_val = completed_count / len(WORKFLOW_NODES)
    st.progress(progress_val, text=f"Pipeline Progress: {completed_count}/{len(WORKFLOW_NODES)} agents completed ({int(progress_val * 100)}%)")


# ---------------------------------------------------------------------------
# UI Components: Human-in-the-Loop Checkpoint Decision Cards
# ---------------------------------------------------------------------------
def render_checkpoint_card(gstate: AgentMLState, interrupt_payload: dict):
    """Renders a dedicated, clear review & approve card for human checkpoints."""
    msg = interrupt_payload.get("message", "Human approval required to proceed.")

    st.markdown(
        f"""
        <div class="checkpoint-banner">
            <div class="checkpoint-header">⚡ Human Checkpoint Required</div>
            <div class="checkpoint-desc">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Feature Engineering Checkpoint
    if "feature_suggestions" in interrupt_payload:
        st.subheader("💡 Select Feature Engineering Transformations")
        st.caption("Opt-in to suggested feature transformations or construct a custom formula. Default is to apply none.")
        
        suggestions = interrupt_payload.get("feature_suggestions") or []
        checked_features = []

        if suggestions:
            st.markdown("**Suggested Features:**")
            for sug in suggestions:
                name = sug.get("name", "feature")
                desc = sug.get("description", "")
                ftype = sug.get("type", "feature")
                if st.checkbox(f"**{name}** (`{ftype}`): {desc}", value=False, key=f"chk_{name}"):
                    checked_features.append(name)
        else:
            st.info("No obvious candidate features detected for this dataset structure.")

        # Custom formula
        st.markdown("---")
        st.markdown("**➕ Add Custom Numeric Formula (Optional)**")
        st.caption("Allowed operators: `+`, `-`, `*`, `/`, parentheses, and existing column names.")
        
        c1, c2 = st.columns(2)
        with c1:
            custom_name = st.text_input("New Feature Name", placeholder="e.g. price_per_sqft", key="fe_name")
        with c2:
            custom_formula = st.text_input("Formula Expression", placeholder="e.g. price / sqft", key="fe_formula")

        custom_features = []
        if custom_name.strip() and custom_formula.strip():
            profile = gstate.get("dataset_profile") or {}
            avail_cols = set(profile.get("numeric_cols", []) + profile.get("categorical_cols", []))
            try:
                validate_formula(custom_formula, avail_cols)
                custom_features = [{"name": custom_name.strip(), "formula": custom_formula.strip()}]
                st.success(f"✓ Valid formula: `{custom_name.strip()} = {custom_formula.strip()}`")
            except (FormulaValidationError, ValueError) as err:
                st.error(f"Invalid Formula: {err}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([2, 5])
        with col_btn1:
            if st.button("✓ Confirm & Apply Features", type="primary", use_container_width=True):
                payload = {
                    "selected_features": checked_features,
                    "custom_features": custom_features,
                }
                with st.spinner("Applying selected features & executing preprocessing..."):
                    res = st.session_state.graph.invoke(Command(resume=payload), st.session_state.config)
                    st.session_state.graph_state = res
                    st.rerun()
        with col_btn2:
            if st.button("Skip Feature Engineering (Apply None)", type="secondary"):
                payload = {"selected_features": [], "custom_features": []}
                with st.spinner("Continuing preprocessing..."):
                    res = st.session_state.graph.invoke(Command(resume=payload), st.session_state.config)
                    st.session_state.graph_state = res
                    st.rerun()

    # 2. Problem Detection Ambiguity Checkpoint
    elif "detected_target_column" in interrupt_payload:
        st.subheader("🎯 Confirm Target Column & Task Type")
        det_target = interrupt_payload.get("detected_target_column")
        det_task = interrupt_payload.get("detected_task_type", "classification")
        conf = interrupt_payload.get("confidence", 0.0)

        st.markdown(f"Detected: Target = `{det_target}`, Task = `{str(det_task).upper()}` (Confidence: `{conf * 100:.1f}%`)")
        
        c1, c2 = st.columns(2)
        with c1:
            confirmed_target = st.text_input("Target Column Name", value=det_target or "")
        with c2:
            task_options = ["classification", "regression", "clustering"]
            idx = task_options.index(det_task) if det_task in task_options else 0
            confirmed_task = st.selectbox("Task Type", options=task_options, index=idx)

        col_b1, col_b2 = st.columns([2, 5])
        with col_b1:
            if st.button("✓ Confirm & Proceed", type="primary", use_container_width=True):
                payload = {
                    "target_column": confirmed_target.strip() if confirmed_target.strip() else None,
                    "task_type": confirmed_task,
                }
                with st.spinner("Resuming pipeline..."):
                    res = st.session_state.graph.invoke(Command(resume=payload), st.session_state.config)
                    st.session_state.graph_state = res
                    st.rerun()

    # 3. Experiment Scope Checkpoint
    elif "default_scope" in interrupt_payload:
        st.subheader("⚙️ Configure Model Zoo Execution Scope")
        default_scope = interrupt_payload.get("default_scope", {})
        available_models = interrupt_payload.get("available_models", [])

        st.markdown(f"**Available Models in Zoo:** `{len(available_models)} candidate models` ({', '.join(available_models)})")

        c1, c2, c3 = st.columns(3)
        with c1:
            max_exps = st.slider("Max Models to Evaluate", min_value=1, max_value=max(len(available_models), 1), value=default_scope.get("max_experiments", len(available_models)))
        with c2:
            max_workers = st.slider("Concurrent Thread Workers", min_value=1, max_value=8, value=default_scope.get("max_workers", 4))
        with c3:
            time_cap = st.number_input("Timeout per Model (seconds, 0=unlimited)", min_value=0, value=int(default_scope.get("time_cap_seconds", 0)))

        col_b1, col_b2 = st.columns([2, 5])
        with col_b1:
            if st.button("✓ Start Model Training", type="primary", use_container_width=True):
                payload = {
                    "max_experiments": max_exps,
                    "max_workers": max_workers,
                    "time_cap_seconds": time_cap,
                }
                with st.spinner("Executing model zoo concurrently..."):
                    res = st.session_state.graph.invoke(Command(resume=payload), st.session_state.config)
                    st.session_state.graph_state = res
                    st.rerun()

    # 4. Best Model Leaderboard Override Checkpoint
    elif "default_best_model_id" in interrupt_payload:
        st.subheader("🏆 Confirm Winning Model Selection")
        default_best = interrupt_payload.get("default_best_model_id")
        ranking = interrupt_payload.get("ranking", [])
        
        st.markdown(f"The evaluation agent ranked **`{default_best}`** in 1st place based on primary validation performance.")
        
        if ranking:
            st.markdown("**Evaluated Leaderboard:**")
            df_rank = pd.DataFrame(ranking)
            display_cols = [c for c in ["rank", "model_name", "primary_metric", "primary_value", "runtime_seconds"] if c in df_rank.columns]
            st.dataframe(df_rank[display_cols], use_container_width=True)

        model_options = [r["model_id"] for r in ranking]
        if default_best not in model_options and default_best != "none":
            model_options.insert(0, default_best)

        selected_best = st.selectbox(
            "Select Approved Winning Model:",
            options=model_options,
            index=model_options.index(default_best) if default_best in model_options else 0,
        )

        col_b1, col_b2 = st.columns([2, 5])
        with col_b1:
            if st.button("✓ Confirm Winning Model & Generate Report", type="primary", use_container_width=True):
                payload = {"best_model_id": selected_best}
                with st.spinner("Finalizing evaluation & generating report..."):
                    res = st.session_state.graph.invoke(Command(resume=payload), st.session_state.config)
                    st.session_state.graph_state = res
                    st.rerun()


# ---------------------------------------------------------------------------
# UI Components: Structured Agent Cards
# ---------------------------------------------------------------------------
def render_profiling_card(state: AgentMLState):
    """Render Dataset Profiling Agent outputs in clean structured cards."""
    profile = state.get("dataset_profile") or {}
    if not profile:
        st.info("Pending execution...")
        return

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{profile.get('num_rows', 0):,}")
    m2.metric("Columns", f"{profile.get('num_cols', 0):,}")
    m3.metric("Duplicate Rows", f"{profile.get('duplicate_count', 0):,}")
    m4.metric("Numeric Features", len(profile.get("numeric_cols", [])))
    m5.metric("Categorical Features", len(profile.get("categorical_cols", [])))

    # Column Summary Table
    st.markdown("##### 📋 Column Schema & Missing Values")
    col_rows = []
    dtypes = profile.get("dtypes", {})
    missing_pct = profile.get("missing_pct", {})
    col_stats = profile.get("column_stats", {})

    for col_name, dtype in dtypes.items():
        stats = col_stats.get(col_name, {})
        missing = missing_pct.get(col_name, 0.0)
        col_type = "Numeric" if col_name in profile.get("numeric_cols", []) else "Categorical"
        
        sample_stat = ""
        if col_type == "Numeric":
            mean_val = stats.get("mean")
            sample_stat = f"Mean: {mean_val:.2f}, Min: {stats.get('min')}, Max: {stats.get('max')}" if mean_val is not None else "—"
        else:
            top_v = stats.get("top_value")
            sample_stat = f"Top: '{top_v}' (freq: {stats.get('top_freq')})" if top_v is not None else "—"

        col_rows.append({
            "Column": col_name,
            "Type": col_type,
            "Dtype": dtype,
            "Missing %": f"{missing:.1f}%",
            "Unique Values": stats.get("unique_count", "—"),
            "Summary Stats": sample_stat,
        })

    if col_rows:
        st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    # Feature suggestions detected
    suggestions = state.get("feature_suggestions") or []
    if suggestions:
        st.markdown("##### 💡 Detected Feature Engineering Opportunities")
        for s in suggestions:
            st.markdown(f"- **`{s.get('name')}`** ({s.get('type')}): {s.get('description')}")

    # Collapsed raw json
    with st.expander("🔍 View Raw Profiling JSON Payload", expanded=False):
        st.json(profile)


def render_problem_detection_card(state: AgentMLState):
    """Render Problem Detection Agent outputs."""
    task_type = state.get("task_type")
    if not task_type:
        st.info("Pending execution...")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Detected Task Type", str(task_type).upper())
    m2.metric("Target Column", state.get("target_column") or "None (Unsupervised)")
    conf = state.get("detection_confidence", 0.0)
    m3.metric("Detection Confidence", f"{conf * 100:.1f}%")

    user_instr = state.get("user_instruction")
    if user_instr and user_instr.strip():
        st.info(f"**User Prediction Goal:** *\"{user_instr.strip()}\"*")

    reasoning = state.get("detection_reasoning")
    if reasoning:
        st.markdown("##### 🧠 Explainable Detection Reasoning")
        st.markdown(f"> {reasoning}")

    with st.expander("🔍 View Raw Detection State", expanded=False):
        st.json({
            "task_type": task_type,
            "target_column": state.get("target_column"),
            "detection_confidence": conf,
            "detection_reasoning": reasoning,
        })


def render_feature_engineering_card(state: AgentMLState):
    """Render Feature Engineering Selection outputs."""
    selected = state.get("selected_features") or []
    custom = state.get("custom_features") or []
    fe_log = state.get("feature_engineering_log") or []

    if "selected_features" not in state and not fe_log:
        st.info("Pending execution...")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Suggested Features", len(selected))
    m2.metric("Custom Formulas Created", len(custom))
    m3.metric("Total Transformations Applied", len(fe_log))

    if custom:
        st.markdown("##### ➕ Custom Formulas")
        for cf in custom:
            st.markdown(f"- `{cf.get('name')}` = `{cf.get('formula')}`")

    if fe_log:
        st.markdown("##### 📜 Feature Engineering Operation Log")
        st.dataframe(pd.DataFrame(fe_log), use_container_width=True, hide_index=True)
    else:
        st.caption("No custom or suggested features were selected.")

    with st.expander("🔍 View Raw Feature Engineering Log", expanded=False):
        st.json({"selected_features": selected, "custom_features": custom, "feature_engineering_log": fe_log})


def render_preprocessing_card(state: AgentMLState):
    """Render Data Preprocessing Agent outputs."""
    clean_path = state.get("clean_dataset_path")
    prep_log = state.get("preprocessing_log") or []

    if not clean_path and not prep_log:
        st.info("Pending execution...")
        return

    m1, m2 = st.columns(2)
    m1.metric("Clean Dataset Path", clean_path or "—")
    m2.metric("Total Preprocessing Operations", len(prep_log))

    if prep_log:
        st.markdown("##### 🧹 Preprocessing Operations Log")
        st.dataframe(pd.DataFrame(prep_log), use_container_width=True, hide_index=True)
    else:
        st.info("No preprocessing steps required.")

    with st.expander("🔍 View Raw Preprocessing Log", expanded=False):
        st.json({"clean_dataset_path": clean_path, "preprocessing_log": prep_log})


def render_experiment_card(state: AgentMLState):
    """Render Model Training / Experiment Orchestrator outputs."""
    results = state.get("experiment_results") or []
    if not results:
        st.info("Pending execution...")
        return

    successful = [r for r in results if r.get("success") is True]
    failed = [r for r in results if r.get("success") is not True]
    total_time = sum(r.get("runtime_seconds", 0) for r in results)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Models Evaluated", len(results))
    m2.metric("Successful Runs", len(successful))
    m3.metric("Failed Runs", len(failed))
    m4.metric("Total Train Time", f"{total_time:.2f}s")

    st.markdown("##### 🧪 Evaluated Models Summary")
    table_rows = []
    for r in results:
        table_rows.append({
            "Model Name": r.get("model_name"),





        
            "Family": r.get("family", "—"),
            "Status": "✓ Success" if r.get("success") else "✗ Failed",
            "Runtime (s)": f"{r.get('runtime_seconds', 0):.3f}",
            "Error / Details": r.get("error_message") or "Completed without errors",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with st.expander("🔍 View Raw Experiment Results Payload", expanded=False):
        st.json(results)


def render_evaluation_card(state: AgentMLState):
    """Render Model Evaluation Agent outputs."""
    ranking = state.get("ranking") or []
    best_model = state.get("best_model_id")

    if not ranking and not best_model:
        st.info("Pending execution...")
        return

    m1, m2 = st.columns(2)
    m1.metric("🏆 Top-Ranked Model", best_model or "None")
    
    top_score = "—"
    if ranking:
        top_entry = ranking[0]
        top_score = f"{top_entry.get('primary_metric', 'Metric')}: {top_entry.get('primary_value', 0):.4f}"
    m2.metric("Primary Metric Score", top_score)

    if ranking:
        st.markdown("##### 🥇 Model Evaluation Leaderboard")
        df_rank = pd.DataFrame(ranking)
        display_cols = [c for c in ["rank", "model_name", "primary_metric", "primary_value", "runtime_seconds", "family"] if c in df_rank.columns]
        st.dataframe(df_rank[display_cols], use_container_width=True, hide_index=True)

    reasoning = state.get("evaluation_reasoning")
    if reasoning:
        st.markdown("##### 📊 Evaluation Narrative")
        st.markdown(f"> {reasoning}")

    with st.expander("🔍 View Raw Leaderboard Payload", expanded=False):
        st.json({"best_model_id": best_model, "ranking": ranking, "evaluation_reasoning": reasoning})


def render_report_agent_card(state: AgentMLState):
    """Render Report Generation Agent outputs."""
    report_path = state.get("report_path")
    if not report_path:
        st.info("Pending execution...")
        return

    st.success(f"✓ Report successfully generated at `{report_path}`")
    pdf_path = report_path.replace(".md", ".pdf")
    has_pdf = os.path.exists(pdf_path)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Markdown Output:** `{os.path.basename(report_path)}`")
    with c2:
        st.markdown(f"**PDF Output:** `{os.path.basename(pdf_path) if has_pdf else 'N/A'}`")


# ---------------------------------------------------------------------------
# Tab 3: Final Results & Report View
# ---------------------------------------------------------------------------
def render_final_results_tab(gstate: AgentMLState):
    """Renders the comprehensive Results & Final Report tab."""
    report_path = gstate.get("report_path")
    ranking = gstate.get("ranking") or []
    best_model = gstate.get("best_model_id")

    if not _is_node_complete(gstate, "report_generation") or not report_path:
        st.info("⏳ The complete experiment report will be displayed here once all 7 pipeline agents finish.")
        return

    # Header Metrics
    st.subheader("🏆 Experiment Results & Leaderboard")
    top_score = "—"
    metric_name = "Metric"
    if ranking:
        metric_name = ranking[0].get("primary_metric", "Score")
        top_score = f"{ranking[0].get('primary_value', 0):.4f}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", best_model or "—")
    m2.metric(f"Top {metric_name}", top_score)
    m3.metric("Task Type", str(gstate.get("task_type", "—")).title())
    m4.metric("Target Column", gstate.get("target_column") or "None")

    # Ranked Leaderboard
    if ranking:
        st.markdown("#### Model Leaderboard")
        df_rank = pd.DataFrame(ranking)
        display_cols = [c for c in ["rank", "model_name", "primary_metric", "primary_value", "runtime_seconds", "family"] if c in df_rank.columns]
        st.dataframe(df_rank[display_cols], use_container_width=True, hide_index=True)

    # Rendered Markdown Report
    st.markdown("---")
    st.subheader("📄 Generated Experiment Report")

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()

        # Download Buttons
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "⬇️ Download Markdown Report (.md)",
                data=report_content,
                file_name=os.path.basename(report_path),
                mime="text/markdown",
                use_container_width=True,
                type="primary",
            )
        with col_d2:
            pdf_path = report_path.replace(".md", ".pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pf:
                    st.download_button(
                        "⬇️ Download PDF Report (.pdf)",
                        data=pf.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )
            else:
                st.caption("PDF version not available (reportlab / weasyprint).")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(report_content)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Report file not found at path: {report_path}")


# ---------------------------------------------------------------------------
# App Initialization & State Setup
# ---------------------------------------------------------------------------
apply_custom_theme()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.graph = build_graph()
    st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}
    st.session_state.graph_state = None
    st.session_state.uploaded_file_path = None

# Sidebar Controls
with st.sidebar:
    st.title("🤖 AgentML")
    st.markdown("**Autonomous Multi-Agent Machine Learning Orchestration**")
    st.caption("Built with LangGraph, Scikit-Learn & XGBoost.")
    
    st.markdown("---")
    st.markdown("##### 📍 Active Session")
    st.code(st.session_state.session_id[:8] + "...", language="text")
    
    status_label = "Ready"
    if st.session_state.graph_state is not None:
        g = st.session_state.graph_state
        if "__interrupt__" in g and len(g["__interrupt__"]) > 0:
            status_label = "Paused at Checkpoint"
        elif _is_node_complete(g, "report_generation"):
            status_label = "Complete"
        else:
            status_label = "Running"
    
    st.markdown(f"**Pipeline Status:** `{status_label}`")

    st.markdown("---")
    if st.button("🔄 Reset Worksession", use_container_width=True, type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.graph = build_graph()
        st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}
        st.session_state.graph_state = None
        st.session_state.uploaded_file_path = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main App Header & Tab Layout
# ---------------------------------------------------------------------------
st.title("AgentML — Multi-Agent ML Orchestrator")
st.markdown(
    '<p style="color: #4A5568; font-size: 14.5px; margin-top: -10px;">'
    "Autonomous end-to-end pipeline: profiling, problem detection, interactive feature engineering, "
    "concurrent model zoo training, ranking, and report generation."
    "</p>",
    unsafe_allow_html=True,
)

tab_upload, tab_pipeline, tab_results = st.tabs([
    "📁 1. Upload & Setup",
    "⚡ 2. Pipeline & Agents",
    "📊 3. Results & Final Report",
])

# ---------------------------------------------------------------------------
# Tab 1: Upload & Setup
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("1. Select & Configure Dataset")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], help="Upload a structured CSV file with numerical and/or categorical features.")

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        temp_path = f"data/upload_{st.session_state.session_id}.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_file_path = temp_path

        df_preview = pd.read_csv(temp_path)
        
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Filename", uploaded_file.name)
        m2.metric("Total Rows", f"{len(df_preview):,}")
        m3.metric("Total Columns", len(df_preview.columns))

        st.markdown("##### 🔎 First 5 Rows Preview")
        st.dataframe(df_preview.head(5), use_container_width=True)

        st.markdown("---")
        st.markdown("##### 💬 Natural Language Prediction Goal *(Optional)*")
        user_instruction = st.text_area(
            "Prediction Instruction",
            placeholder="e.g. 'Predict who survived or not based on passenger demographics' or 'Cluster customers into segments'",
            height=70,
            help="AgentML will ground your request in the actual CSV columns and auto-select the right target and task type.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Initialize & Start Pipeline", type="primary", use_container_width=True):
            initial_state = {
                "session_id": st.session_state.session_id,
                "raw_file_path": temp_path,
                "user_instruction": user_instruction.strip() if user_instruction else "",
                "target_column": None,
                "errors": [],
                "status": "running",
            }
            with st.spinner("Starting AgentML pipeline & profiling dataset..."):
                res = st.session_state.graph.invoke(initial_state, st.session_state.config)
                st.session_state.graph_state = res
                st.rerun()

    elif st.session_state.graph_state is None:
        st.info("👋 Upload a CSV file above to begin the multi-agent experiment workflow.")


# ---------------------------------------------------------------------------
# Tab 2: Pipeline & Agents Execution
# ---------------------------------------------------------------------------
with tab_pipeline:
    if st.session_state.graph_state is None:
        st.info("No active pipeline execution. Please upload a dataset in **📁 1. Upload & Setup** to start.")
    else:
        gstate: AgentMLState = st.session_state.graph_state
        is_interrupted = "__interrupt__" in gstate and len(gstate["__interrupt__"]) > 0
        current_phase = gstate.get("current_phase", "orchestrator")

        # 1. Render Stepper & Progress
        render_pipeline_stepper(gstate, current_phase, is_interrupted)
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Check for Hard Errors
        hard_errors = [e for e in (gstate.get("errors") or []) if not e.get("recoverable", True)]
        if hard_errors:
            st.error("🚨 Pipeline execution encountered a non-recoverable error.")
            for err in hard_errors:
                st.markdown(f"**Agent [{err.get('phase', 'pipeline')}]:** {err.get('message')} *(Type: `{err.get('error_type')}`)*")
            if st.button("Restart with New Dataset"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.graph = build_graph()
                st.session_state.config = {"configurable": {"thread_id": st.session_state.session_id}}
                st.session_state.graph_state = None
                st.rerun()
            st.stop()

        # 3. Check for Human Checkpoint Interrupts
        if is_interrupted:
            active_interrupt = gstate["__interrupt__"][-1].value
            render_checkpoint_card(gstate, active_interrupt)

        # 4. Structured Expandable Cards for all 7 Agents
        st.markdown("### 🤖 Agent Outputs & Logs")
        
        # Agent 1: Profiling
        prof_done = _is_node_complete(gstate, "dataset_profiling")
        with st.expander(f"{'✓' if prof_done else '○'} 1. Dataset Profiling Agent", expanded=prof_done and not _is_node_complete(gstate, "problem_detection")):
            render_profiling_card(gstate)

        # Agent 2: Detection
        det_done = _is_node_complete(gstate, "problem_detection")
        with st.expander(f"{'✓' if det_done else '○'} 2. Problem Detection Agent", expanded=det_done and not _is_node_complete(gstate, "data_preprocessing")):
            render_problem_detection_card(gstate)

        # Agent 3: Feature Engineering
        fe_done = _is_node_complete(gstate, "feature_engineering")
        with st.expander(f"{'✓' if fe_done else '○'} 3. Feature Engineering Selection", expanded=fe_done and not _is_node_complete(gstate, "data_preprocessing")):
            render_feature_engineering_card(gstate)

        # Agent 4: Preprocessing
        prep_done = _is_node_complete(gstate, "data_preprocessing")
        with st.expander(f"{'✓' if prep_done else '○'} 4. Data Preprocessing Agent", expanded=prep_done and not _is_node_complete(gstate, "experiment_orchestrator")):
            render_preprocessing_card(gstate)

        # Agent 5: Experiment Orchestrator
        exp_done = _is_node_complete(gstate, "experiment_orchestrator")
        with st.expander(f"{'✓' if exp_done else '○'} 5. Experiment Orchestration (Model Training)", expanded=exp_done and not _is_node_complete(gstate, "model_evaluation")):
            render_experiment_card(gstate)

        # Agent 6: Model Evaluation
        eval_done = _is_node_complete(gstate, "model_evaluation")
        with st.expander(f"{'✓' if eval_done else '○'} 6. Model Evaluation & Leaderboard", expanded=eval_done and not _is_node_complete(gstate, "report_generation")):
            render_evaluation_card(gstate)

        # Agent 7: Report Generation
        rep_done = _is_node_complete(gstate, "report_generation")
        with st.expander(f"{'✓' if rep_done else '○'} 7. Report Generation Agent", expanded=rep_done):
            render_report_agent_card(gstate)


# ---------------------------------------------------------------------------
# Tab 3: Results & Final Report
# ---------------------------------------------------------------------------
with tab_results:
    if st.session_state.graph_state is None:
        st.info("No experiment has been run yet. Upload a dataset in **📁 1. Upload & Setup** to generate results.")
    else:
        render_final_results_tab(st.session_state.graph_state)
