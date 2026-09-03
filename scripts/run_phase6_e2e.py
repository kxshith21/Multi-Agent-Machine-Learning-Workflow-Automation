"""
Phase 6 end-to-end test + Groq fallback demonstration.

Run as:
    python scripts/run_phase6_e2e.py
    python scripts/run_phase6_e2e.py --simulate-groq-failure

This script:
  1. Generates a synthetic binary-classification CSV (300 rows, 6 features).
  2. Runs the full Phase 0→6 pipeline, handling LangGraph interrupt()
     checkpoints automatically (accepts defaults at each one).
  3. Prints the generated report path and its first 60 lines.

--simulate-groq-failure mode:
  Monkeypatches groq_client.chat() to always raise, then re-runs the pipeline.
  The report_agent must fall back to template-only mode and still produce a
  complete report (exit code 0).  If it crashes instead, the fallback is broken.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.types import Command
from sklearn.datasets import make_classification

# Make sure project root is on the path when run from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e2e")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_csv(tmp_dir: Path) -> str:
    """Generate a clean, realistic-looking binary-classification CSV."""
    rng = np.random.default_rng(42)
    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[
        "age", "income", "credit_score", "debt_ratio", "years_employed", "num_accounts"
    ])
    df["age"] = (df["age"] * 10 + 40).clip(18, 80).astype(int)
    df["income"] = (df["income"] * 15000 + 55000).clip(20000, 150000).astype(int)
    df["credit_score"] = (df["credit_score"] * 80 + 650).clip(300, 850).astype(int)
    df["approved"] = y  # binary target
    csv_path = tmp_dir / "loan_applications.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Generated CSV: %s (%d rows × %d cols)", csv_path, len(df), len(df.columns))
    return str(csv_path)


def _run_pipeline_with_auto_checkpoints(
    csv_path: str,
    target: str,
    session_id: str,
) -> dict:
    """
    Run the AgentML graph, automatically accepting all interrupt() checkpoints
    with their default values (this mirrors what a non-interactive runner would do).
    Returns the final state.
    """
    from src.orchestrator.graph import build_graph

    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "session_id":    session_id,
        "raw_file_path": csv_path,
        "target_column": target,
        "errors":        [],
        "status":        "running",
    }

    logger.info("=== INVOKE 1: Starting pipeline ===")
    result = graph.invoke(initial_state, config)

    # Keep resuming until no more interrupts
    max_resumes = 15
    for resume_num in range(1, max_resumes + 1):
        if "__interrupt__" not in result:
            break
        payload = result["__interrupt__"][-1].value
        msg = payload.get("message", "")
        logger.info("=== INTERRUPT %d: %.80s ===", resume_num, msg)

        # Determine the right resume payload based on interrupt content.
        if "default_scope" in payload:
            # Phase 4 scope checkpoint — pass explicit scope values
            default_scope = payload["default_scope"]
            resume_value = {
                "max_experiments":  default_scope.get("max_experiments", 6),
                "max_workers":      default_scope.get("max_workers", 4),
                "time_cap_seconds": 0,
            }
        elif "default_best_model_id" in payload:
            # Phase 5 evaluation checkpoint — confirm the automatic winner
            # (must echo the default_best_model_id; empty dict loops)
            resume_value = {"best_model_id": payload["default_best_model_id"]}
        elif "detected_task_type" in payload:
            # Phase 3 detection checkpoint — accept the detected target/task.
            # MUST return a non-empty payload: resuming with {} re-triggers the
            # interrupt indefinitely (empty resume is treated as "no value").
            resume_value = {
                "target_column": payload.get("detected_target_column"),
                "task_type": payload.get("detected_task_type"),
            }
        else:
            resume_value = {}


        logger.info("=== RESUME %d: %s ===", resume_num, resume_value)
        result = graph.invoke(Command(resume=resume_value), config)

    return result


def _print_report(report_path: str, lines: int = 80) -> None:
    p = Path(report_path)
    if not p.exists():
        logger.error("Report file not found: %s", report_path)
        return
    content = p.read_text(encoding="utf-8")
    all_lines = content.splitlines()
    print("\n" + "=" * 70)
    print(f"REPORT: {report_path}  ({len(all_lines)} lines total)")
    print("=" * 70)
    for line in all_lines[:lines]:
        sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))
    if len(all_lines) > lines:
        print(f"\n... [{len(all_lines) - lines} more lines — open {report_path} to see the full report]")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="AgentML Phase 6 end-to-end demo")
    parser.add_argument(
        "--simulate-groq-failure",
        action="store_true",
        help="Monkeypatch groq_client.chat() to always fail, proving the fallback works.",
    )
    args = parser.parse_args()

    tmp_dir = Path("data")
    tmp_dir.mkdir(exist_ok=True)

    csv_path = _build_csv(tmp_dir)
    session_id = f"phase6_e2e_{uuid.uuid4().hex[:8]}"

    # ---- Optionally simulate Groq failure ----
    if args.simulate_groq_failure:
        print("\n" + "!" * 70)
        print("!  SIMULATING GROQ FAILURE — narration should fall back cleanly  !")
        print("!" * 70 + "\n")
        import src.llm.groq_client as _gc

        def _always_fail(prompt: str, **kwargs):
            from src.orchestrator.state import ErrorEntry
            return "", [
                ErrorEntry(
                    phase="groq_client",
                    error_type="groq_api_error",
                    message="[SIMULATED] Groq API call failed — testing fallback path",
                    recoverable=True,
                )
            ]

        _gc.chat = _always_fail
        session_id += "_groq_fail"

    # ---- Run pipeline ----
    logger.info("Running full Phase 0→6 pipeline | session=%s", session_id)
    try:
        final_state = _run_pipeline_with_auto_checkpoints(
            csv_path=csv_path,
            target="approved",
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("Pipeline raised unexpectedly: %s", exc)
        return 1

    # ---- Summary ----
    report_path = final_state.get("report_path")
    errors = final_state.get("errors") or []
    recoverable = [e for e in errors if e.get("recoverable", True)]
    hard = [e for e in errors if not e.get("recoverable", True)]

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Session ID   : {final_state.get('session_id')}")
    print(f"  Status       : {final_state.get('status')}")
    print(f"  Task type    : {final_state.get('task_type')}")
    print(f"  Best model   : {final_state.get('best_model_id')}")
    print(f"  Report path  : {report_path}")
    print(f"  Errors       : {len(recoverable)} recoverable, {len(hard)} hard-fail")
    if args.simulate_groq_failure:
        report_content = Path(report_path).read_text(encoding="utf-8") if report_path else ""
        if "template-only" in report_content.lower() or "llm unavailable" in report_content.lower() or "template mode" in report_content.lower():
            print("\n  [PASS] FALLBACK CONFIRMED: report mentions template-only mode")
        else:
            print("\n  [WARN] Could not confirm fallback mode in report content")
    print("=" * 70)

    if report_path:
        _print_report(report_path, lines=100)

    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
