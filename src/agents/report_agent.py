"""
Agent 7: Report Generation Agent (Phase 6)

Fills templates/report_template.md.j2 from state (structured, rule-based),
then layers a Groq LLM narration on top — the LLM explains decisions in prose
but does NOT alter any underlying fact, metric, or ranking (Rules.md §3).

Error handling (Rules.md §4):
  - Groq call failure → retry once with backoff (groq_client handles retries),
    then fall back to template-only report (llm_available=False).
    The pipeline is never blocked by a narration failure.

Output:
  - Markdown report written to outputs/reports/report_<session_id>.md
  - Optional PDF via weasyprint (if installed and PDF conversion requested).
  - state["report_path"] set to the written file path.

Per Architecture.md §5: output goes to outputs/reports/ (gitignored).
"""

from __future__ import annotations

import logging
import math
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.llm import groq_client
from src.metrics.task_metrics import PRIMARY_METRIC, TIEBREAKER_METRIC
from src.orchestrator.state import AgentMLState, ErrorEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"
_TEMPLATE_FILE = "report_template.md.j2"
_OUTPUT_DIR = Path("outputs") / "reports"

# Extra metrics to show per task type (besides the primary)
_EXTRA_COLS: Dict[str, List[str]] = {
    "classification": ["f1", "precision", "recall"],
    "regression":     ["rmse", "mae"],
    "clustering":     ["n_clusters", "n_noise"],
}

# LLM system prompt — emphasises narration-only role
_LLM_SYSTEM = (
    "You are a concise, data-driven ML report narrator. "
    "You are given structured facts (numbers, decisions, rankings) that were determined "
    "by rule-based logic. Your ONLY job is to explain those facts in clear, plain English — "
    "2–5 sentences per section. "
    "Do NOT introduce new conclusions, do NOT suggest new models or actions, "
    "do NOT use hedging language ('may', 'might', 'could'). "
    "Speak in present tense. No bullet points. No markdown formatting in your reply."
)


# ---------------------------------------------------------------------------
# Jinja2 helpers (custom filters)
# ---------------------------------------------------------------------------

def _round_metric(value: Any) -> str:
    """Format a metric value for display in the report table."""
    if value is None or value == "—":
        return "—"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(fval) or math.isinf(fval):
        return "N/A"
    # integers stay as integers
    if fval == int(fval) and abs(fval) < 1_000_000:
        return str(int(fval))
    return f"{fval:.4f}"


def _build_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["round_metric"] = _round_metric
    return env


# ---------------------------------------------------------------------------
# Context builder — rule-based, no LLM
# ---------------------------------------------------------------------------

def _build_context(state: AgentMLState, llm_available: bool) -> Dict[str, Any]:
    """
    Assemble the full Jinja2 template context from state.
    All values here are rule-based / factual — LLM narration is added
    separately and injected into the context.
    """
    profile: Dict[str, Any] = state.get("dataset_profile") or {}
    experiment_results: List[Dict] = list(state.get("experiment_results") or [])
    ranking: List[Dict] = list(state.get("ranking") or [])
    task_type: str = state.get("task_type") or "unknown"
    errors: List[ErrorEntry] = list(state.get("errors") or [])

    # ---- dataset profile fields ----
    # profiling_agent uses: num_rows, num_cols, duplicate_count, missing_pct (dict per col)
    n_rows = profile.get("num_rows") or profile.get("n_rows", 0)
    n_cols = profile.get("num_cols") or profile.get("n_cols", 0)
    # missing_pct is a dict {col: pct_float} — sum counts to get total missing cells
    missing_pct_per_col: Dict[str, float] = profile.get("missing_pct") or {}
    total_missing = int(sum(
        (pct / 100.0) * n_rows
        for pct in missing_pct_per_col.values()
    )) if missing_pct_per_col and n_rows else 0
    total_cells = (n_rows * n_cols) if (n_rows and n_cols) else 1
    missing_pct = (total_missing / total_cells * 100) if total_cells else 0.0
    n_duplicates = profile.get("duplicate_count") or profile.get("n_duplicates", 0) or 0
    column_types: Dict[str, str] = profile.get("dtypes") or {}

    profile_summary = (
        f"The dataset has {n_rows} rows and {n_cols} columns. "
        f"Missing values account for {missing_pct:.1f}% of all cells. "
        f"{n_duplicates} duplicate rows were found."
    )

    # ---- preprocessing ----
    # preprocessing_agent log keys: column, operation, details, reasoning
    preprocessing_log: List[Dict] = list(state.get("preprocessing_log") or [])
    preprocessing_steps = []
    for entry in preprocessing_log:
        operation = entry.get("operation") or entry.get("step") or "transformation"
        col = entry.get("column") or entry.get("columns") or "(all)"
        columns = [col] if isinstance(col, str) else list(col)
        reason = entry.get("reasoning") or entry.get("reason") or entry.get("details") or ""
        preprocessing_steps.append({
            "operation": operation,
            "columns": columns,
            "reason": reason,
        })

    # ---- experiments ----
    n_ok = sum(1 for r in experiment_results if r.get("success") is True)
    n_fail = len(experiment_results) - n_ok
    primary_metric_name = PRIMARY_METRIC.get(task_type, "metric")
    extra_metric_cols = _EXTRA_COLS.get(task_type, [])

    # ---- best model ----
    best_model_id = state.get("best_model_id") or "none"
    best_record = next(
        (r for r in experiment_results
         if r.get("model_id") == best_model_id or r.get("model_name") == best_model_id),
        None,
    )
    best_model_name = (best_record or {}).get("model_name") or best_model_id
    best_model_params = (best_record or {}).get("params") or {}
    best_model_metrics = (best_record or {}).get("metrics") or {}

    return {
        "session_id":           state.get("session_id") or "unknown",
        "generated_at":         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "llm_available":        llm_available,
        # dataset
        "raw_file_path":        state.get("raw_file_path") or "unknown",
        "n_rows":               n_rows,
        "n_cols":               n_cols,
        "n_missing":            total_missing,
        "missing_pct":          missing_pct,
        "n_duplicates":         n_duplicates,
        "column_types":         column_types,
        "profile_summary":      profile_summary,
        "dataset_narration":    "",   # filled by _add_narrations if LLM available
        # preprocessing
        "preprocessing_steps":  preprocessing_steps,
        "clean_dataset_path":   state.get("clean_dataset_path") or "N/A",
        "preprocessing_narration": "",
        # detection
        "task_type":            task_type,
        "target_column":        state.get("target_column"),
        "detection_confidence": state.get("detection_confidence") or 0.0,
        "detection_reasoning":  state.get("detection_reasoning") or "",
        "detection_narration":  "",
        # experiments
        "experiment_results":   experiment_results,
        "n_experiments_run":    len(experiment_results),
        "n_experiments_ok":     n_ok,
        "n_experiments_failed": n_fail,
        # leaderboard
        "ranking":              ranking,
        "primary_metric_name":  primary_metric_name,
        "extra_metric_cols":    extra_metric_cols,
        "leaderboard_narration": "",
        # best model
        "best_model_id":        best_model_id,
        "best_model_name":      best_model_name,
        "evaluation_reasoning": state.get("evaluation_reasoning") or "",
        "best_model_params":    best_model_params,
        "best_model_metrics":   best_model_metrics,
        "best_narration":       "",
        # errors
        "errors":               errors,
    }


# ---------------------------------------------------------------------------
# LLM narration layer (Rules.md §3 — narrates facts, never alters them)
# ---------------------------------------------------------------------------

def _add_narrations(ctx: Dict[str, Any]) -> tuple[Dict[str, Any], List[ErrorEntry]]:
    """
    Call Groq for each report section. Returns the enriched context and any
    errors collected. Each call is independent — one failure doesn't stop the
    rest (though we bail out of ALL narration if the first call signals a
    non-recoverable auth error).
    """
    errors: List[ErrorEntry] = []

    def _narrate(prompt: str) -> str:
        text, call_errors = groq_client.chat(prompt=prompt, system=_LLM_SYSTEM)
        for e in call_errors:
            errors.append(e)
        return text

    # 1. Dataset narration
    ctx["dataset_narration"] = _narrate(
        f"Narrate this dataset summary for a report:\n"
        f"- {ctx['n_rows']} rows, {ctx['n_cols']} columns\n"
        f"- {ctx['missing_pct']:.1f}% missing values\n"
        f"- {ctx['n_duplicates']} duplicate rows\n"
        f"- Column types: {', '.join(f'{k}({v})' for k, v in list(ctx['column_types'].items())[:8])}"
    )

    # 2. Preprocessing narration
    if ctx["preprocessing_steps"]:
        steps_desc = "; ".join(
            f"{s['operation']} on {s['columns']} ({s['reason']})"
            for s in ctx["preprocessing_steps"][:6]
        )
        ctx["preprocessing_narration"] = _narrate(
            f"Narrate these preprocessing steps for a report:\n{steps_desc}"
        )

    # 3. Detection narration
    ctx["detection_narration"] = _narrate(
        f"Narrate this problem detection result:\n"
        f"- Task type: {ctx['task_type']}\n"
        f"- Target column: {ctx['target_column']}\n"
        f"- Confidence: {ctx['detection_confidence']*100:.1f}%\n"
        f"- Reasoning: {ctx['detection_reasoning']}"
    )

    # 4. Leaderboard narration
    if ctx["ranking"]:
        top3 = ctx["ranking"][:3]
        top3_desc = "\n".join(
            f"  {r['rank']}. {r['model_name']}: {ctx['primary_metric_name']}={r['primary_value']:.4f}"
            for r in top3
        )
        ctx["leaderboard_narration"] = _narrate(
            f"Narrate the top experiment results for a {ctx['task_type']} task:\n"
            f"{top3_desc}\n"
            f"Total: {ctx['n_experiments_ok']} successful, {ctx['n_experiments_failed']} failed."
        )

    # 5. Best model / executive summary narration
    if ctx["best_model_id"] and ctx["best_model_id"] != "none":
        metrics_desc = ", ".join(
            f"{k}={v}" for k, v in list(ctx["best_model_metrics"].items())[:5]
        )
        ctx["best_narration"] = _narrate(
            f"Write an executive summary for an ML report. "
            f"Task: {ctx['task_type']}. Dataset: {ctx['n_rows']} rows, {ctx['n_cols']} cols. "
            f"Best model: {ctx['best_model_name']}. Metrics: {metrics_desc}. "
            f"The model was chosen because: {ctx['evaluation_reasoning']}"
        )

    return ctx, errors


# ---------------------------------------------------------------------------
# Rendering + writing
# ---------------------------------------------------------------------------

def _render(ctx: Dict[str, Any]) -> str:
    env = _build_jinja_env()
    template = env.get_template(_TEMPLATE_FILE)
    return template.render(**ctx)


def _write_report(content: str, session_id: str) -> str:
    """Write the report to outputs/reports/ and return the path."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"report_{session_id}.md"
    out_path = _OUTPUT_DIR / filename
    out_path.write_text(content, encoding="utf-8")
    logger.info("[report_agent] wrote report → %s", out_path)
    return str(out_path)


def _maybe_pdf(md_path: str) -> Optional[str]:
    """
    Attempt PDF conversion via reportlab (pure Python, no system deps).

    reportlab is already a lightweight, Windows-friendly dependency. If it is
    unavailable, the report is still complete in Markdown form (non-fatal).
    """
    try:
        from src.utils.pdf_export import build_pdf
        pdf_path = md_path.replace(".md", ".pdf")
        build_pdf(md_path, pdf_path)
        logger.info("[report_agent] PDF written → %s", pdf_path)
        return pdf_path
    except ImportError:
        logger.debug("[report_agent] reportlab not installed — skipping PDF conversion")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[report_agent] PDF conversion failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public agent function
# ---------------------------------------------------------------------------

def report_agent(state: AgentMLState) -> Dict[str, Any]:
    """
    Generate the final Phase 6 report from state.

    Steps:
      1. Build rule-based Jinja2 context from state.
      2. Attempt Groq LLM narration for each section (Rules.md §4 fallback).
      3. Render the Jinja2 template.
      4. Write Markdown to outputs/reports/.
      5. Attempt optional PDF conversion.

    Returns:
        dict with keys: report_path, errors, status
    """
    session_id: str = state.get("session_id") or "unknown"
    accumulated_errors: List[ErrorEntry] = list(state.get("errors") or [])

    logger.info("[report_agent] starting | session=%s", session_id)

    # ---- 1. Build structured context (rule-based, no LLM) ----
    ctx = _build_context(state, llm_available=True)

    # ---- 2. LLM narration — fallback per Rules.md §4 ----
    llm_available = True
    try:
        ctx, narration_errors = _add_narrations(ctx)
        # Check for hard (auth) failures — if the key is wrong, skip all narration
        hard_narration = [e for e in narration_errors if not e.get("recoverable", True)]
        if hard_narration or any(e.get("error_type") == "groq_api_error" for e in narration_errors):
            llm_available = False
            logger.warning("[report_agent] Groq narration failed — falling back to template-only report")
        for e in narration_errors:
            accumulated_errors.append(e)
    except Exception as exc:  # noqa: BLE001
        llm_available = False
        logger.warning("[report_agent] Narration raised unexpectedly: %s — falling back", exc)
        accumulated_errors.append(ErrorEntry(
            phase="report_agent",
            error_type="narration_exception",
            message=f"LLM narration raised an exception: {exc}",
            recoverable=True,
        ))

    # Update context with final llm_available flag (template uses it for conditional blocks)
    ctx["llm_available"] = llm_available

    # ---- 3. Render template ----
    try:
        markdown_content = _render(ctx)
    except Exception as exc:  # noqa: BLE001
        # Template rendering should never fail — if it does, emit a minimal fallback
        logger.error("[report_agent] Jinja2 render failed: %s", exc)
        accumulated_errors.append(ErrorEntry(
            phase="report_agent",
            error_type="template_render_error",
            message=f"Jinja2 render failed: {exc}",
            recoverable=False,
        ))
        markdown_content = (
            f"# AgentML Report — RENDER FAILED\n\n"
            f"Session: {session_id}\n\n"
            f"Template rendering raised an exception. "
            f"Please check the errors list for details.\n\n"
            f"Error: {exc}\n"
        )

    # ---- 4. Write Markdown ----
    report_path = _write_report(markdown_content, session_id)

    # ---- 5. Optional PDF ----
    pdf_path = _maybe_pdf(report_path)
    if pdf_path:
        logger.info("[report_agent] PDF available at %s", pdf_path)

    logger.info(
        "[report_agent] complete | llm=%s | path=%s", llm_available, report_path
    )

    return {
        "report_path": report_path,
        "errors":      accumulated_errors,
        "status":      "completed",
    }
