"""
Phase 0 tests — Foundation (updated for Architecture.md State Schema)
"""

from __future__ import annotations

import os
import uuid
import pytest

from src.orchestrator.graph import build_graph, run_pipeline
from src.orchestrator.state import AgentMLState

SAMPLE_CSV = "data/sample.csv"


class TestStateSchema:
    def test_state_schema_fields(self):
        """AgentMLState TypedDict contains all expected keys from Architecture.md §3."""
        expected_fields = {
            "raw_file_path", "dataset_profile", "clean_dataset_path",
            "preprocessing_log", "task_type", "target_column",
            "detection_confidence", "detection_reasoning", "experiment_results",
            "best_model_id", "ranking", "report_path", "errors", "status",
            "session_id", "current_phase"
        }
        actual_fields = set(AgentMLState.__annotations__.keys())
        missing = expected_fields - actual_fields
        assert not missing, f"AgentMLState is missing fields: {missing}"


class TestPhase0Pipeline:
    def test_stub_pipeline_runs_without_error(self):
        """A CSV path flows through all stubs/implemented agents end-to-end without raising."""
        result = run_pipeline(raw_file_path=SAMPLE_CSV, target_column="label")
        assert result is not None

    def test_placeholder_report_path_is_populated(self):
        """report_path is populated after the run."""
        result = run_pipeline(raw_file_path=SAMPLE_CSV, target_column="label")
        path = result.get("report_path")
        assert isinstance(path, str) and len(path) > 0

    def test_session_id_is_valid_uuid4(self):
        """orchestrator_node generates a syntactically valid UUID4."""
        result = run_pipeline(raw_file_path=SAMPLE_CSV, target_column="label")
        sid = result.get("session_id")
        assert sid is not None
        parsed = uuid.UUID(sid, version=4)
        assert str(parsed) == sid

    def test_final_phase_is_report_generation(self):
        """current_phase in the final state is 'report_generation'."""
        result = run_pipeline(raw_file_path=SAMPLE_CSV, target_column="label")
        assert result.get("current_phase") == "report_generation"

    def test_errors_list_is_empty_on_clean_run(self):
        """No errors are added during an all-stub run."""
        result = run_pipeline(raw_file_path=SAMPLE_CSV, target_column="label")
        assert result.get("errors") == []

    def test_pipeline_without_target_column(self):
        """Pipeline accepts None target_column without error."""
        result = run_pipeline(raw_file_path=SAMPLE_CSV, target_column=None)
        assert result is not None
        assert result.get("report_path")

    def test_graph_is_compilable(self):
        """build_graph() produces a compiled LangGraph graph without error."""
        graph = build_graph()
        assert graph is not None


class TestPhase0EdgeCases:
    def test_hard_error_in_initial_state_raises_valueerror(self, monkeypatch):
        """
        If a non-recoverable error is present in state, run_pipeline must raise ValueError.
        """
        from src.orchestrator.state import ErrorEntry
        import src.orchestrator.graph as graph_module

        bad_error = ErrorEntry(
            phase="test_injected",
            error_type="test_hard_fail",
            message="Intentional hard fail for testing.",
            recoverable=False,
        )

        def run_pipeline_with_injected_error(raw_file_path, target_column=None, extra_state=None):
            initial: dict = {
                "raw_file_path": raw_file_path,
                "target_column": target_column,
                "errors": [bad_error],
                "status": "running",
            }
            if extra_state:
                initial.update(extra_state)
            g = graph_module.build_graph()
            final = g.invoke(initial)
            hard = [e for e in final.get("errors", []) if not e.get("recoverable", True)]
            if hard or final.get("status") == "failed":
                raise ValueError("Pipeline terminated with non-recoverable error(s)")
            return final

        monkeypatch.setattr(graph_module, "run_pipeline", run_pipeline_with_injected_error)

        with pytest.raises(ValueError, match="terminated"):
            graph_module.run_pipeline(raw_file_path=SAMPLE_CSV, target_column="label")

    def test_session_id_preserved_if_provided(self):
        """If a session_id is injected into the initial state, it must be preserved."""
        fixed_id = str(uuid.uuid4())
        result = run_pipeline(
            raw_file_path=SAMPLE_CSV,
            target_column="label",
            extra_state={"session_id": fixed_id},
        )
        assert result.get("session_id") == fixed_id

    def test_groq_client_raises_on_missing_key(self, monkeypatch):
        """hello_world() raises EnvironmentError when GROQ_API_KEY is unset/default."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")

        from src.llm import groq_client
        with pytest.raises((EnvironmentError, RuntimeError)):
            groq_client.hello_world()
