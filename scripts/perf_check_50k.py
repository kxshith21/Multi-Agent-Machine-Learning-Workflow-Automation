"""
Phase 8 — Performance check on a large (50k+) CSV.

Generates a synthetic 55k-row binary-classification dataset, runs the full
AgentML pipeline (auto-accepting checkpoint defaults), and reports wall-clock
timing for each stage plus the total.

Usage:
    python scripts/perf_check_50k.py
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.types import Command
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("perf")


def build_csv(path: Path, n: int = 55_000) -> str:
    rng = np.random.default_rng(7)
    n_features = 12
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=7,
    )
    cols = [
        "age", "income", "credit_score", "debt_ratio", "years_employed",
        "num_accounts", "balance", "loan_amount", "interest_rate", "term_months",
        "dependents", "employment_years",
    ]
    df = pd.DataFrame(X, columns=cols)
    df["age"] = (df["age"] * 10 + 40).clip(18, 80).astype(int)
    df["income"] = (df["income"] * 20000 + 60000).clip(15000, 250000).astype(int)
    df["credit_score"] = (df["credit_score"] * 90 + 650).clip(300, 850).astype(int)
    df["loan_amount"] = (df["loan_amount"] * 30000 + 50000).clip(5000, 500000).astype(int)
    df["approved"] = y
    df.to_csv(path, index=False)
    logger.info("Wrote %d-row CSV → %s", len(df), path)
    return str(path)


def run(csv_path: str, target: str, session_id: str):
    from src.orchestrator.graph import build_graph

    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    initial = {
        "session_id": session_id,
        "raw_file_path": csv_path,
        "target_column": target,
        "errors": [],
        "status": "running",
    }

    phases: list[tuple[str, float, float]] = []
    phase_start = time.perf_counter()

    result = graph.invoke(initial, config)
    phases.append(("setup_through_first_interrupt", phase_start, time.perf_counter()))

    for _ in range(15):
        if "__interrupt__" not in result:
            break
        payload = result["__interrupt__"][-1].value
        if "default_scope" in payload:
            ds = payload["default_scope"]
            resume = {
                "max_experiments": ds.get("max_experiments", 6),
                "max_workers": ds.get("max_workers", 4),
                "time_cap_seconds": 0,
            }
        elif "default_best_model_id" in payload:
            resume = {"best_model_id": payload["default_best_model_id"]}
        elif "detected_task_type" in payload:
            # Detection checkpoint — must resume with non-empty payload.
            resume = {
                "target_column": payload.get("detected_target_column"),
                "task_type": payload.get("detected_task_type"),
            }
        else:
            resume = {}
        t0 = time.perf_counter()
        result = graph.invoke(Command(resume=resume), config)
        phases.append((f"resume({payload.get('message','…')[:40]})", t0, time.perf_counter()))

    return result, phases


def main() -> int:
    csv_path = build_csv(Path("data") / "perf_50k.csv")
    session_id = f"perf_{uuid.uuid4().hex[:8]}"

    wall_start = time.perf_counter()
    result, phases = run(csv_path, "approved", session_id)
    total = time.perf_counter() - wall_start

    print("\n=== PERFORMANCE: 50k+ ROW CSV ===")
    print(f"  CSV            : {csv_path}")
    profile = result.get("dataset_profile", {})
    print(f"  Rows × Cols    : {profile.get('num_rows')} × {profile.get('num_cols')}")
    print(f"  Task           : {result.get('task_type')}")
    print(f"  Experiments    : {len(result.get('experiment_results', []))}")
    print(f"  Best model     : {result.get('best_model_id')}")
    print(f"  Report path    : {result.get('report_path')}")
    hard = [e for e in result.get("errors", []) if not e.get("recoverable", True)]
    print(f"  Hard errors    : {len(hard)}")
    print(f"  TOTAL WALL TIME: {total:.2f} s")
    print(f"  Phase breakdown:")
    for label, t0, t1 in phases:
        print(f"    {t1 - t0:8.3f} s  {label}")
    print("=" * 40)
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
