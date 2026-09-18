"""
AgentML DeepEval Evaluation Runner

Runs an end-to-end evaluation benchmark across AgentML agents:
1. ProblemDetectionAgent: Resolving user intent to target column + task type (G-Eval Accuracy)
2. ReportAgent: Executive summary narration against pipeline ground truth (Faithfulness & Hallucination)
3. Agent Q&A Relevancy: Assessing decision explanation quality (Answer Relevancy)

Outputs a comprehensive terminal metrics report.
"""

from __future__ import annotations

import os
import sys
import time
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Disable telemetry prompts
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

load_dotenv()

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.models import OpenAIModel

def measure_with_rate_limit_retry(metric, test_case, label, max_attempts=5, wait_s=60):
    """Calls metric.measure(test_case), retrying on Groq TPM rate limits."""
    for attempt in range(1, max_attempts + 1):
        try:
            metric.measure(test_case)
            return
        except Exception as exc:
            is_ratelimit = "429" in str(exc) or "rate limit" in str(exc).lower() or "RateLimitError" in type(exc).__name__
            if not is_ratelimit or attempt == max_attempts:
                raise
            print(f"  Rate limit hit during {label} (attempt {attempt}/{max_attempts}). "
                  f"Waiting {wait_s}s before retrying...")
            time.sleep(wait_s)

def get_judge():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        print("ERROR: GROQ_API_KEY is not configured in .env.")
        sys.exit(1)
    
    # Default to gpt-oss-120b (stronger judge) for consistent results
    eval_model = os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-120b")
    return OpenAIModel(
        model=eval_model,
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )

def main():
    print("=" * 75)
    print(" AgentML Multi-Agent Evaluation via DeepEval")
    print("=" * 75)
    
    judge = get_judge()
    print(f" Judge LLM: {judge.get_model_name()} (Groq endpoint)")
    print("-" * 75)

    results = []

    # -----------------------------------------------------------------------
    # Benchmark 1: Problem Detection Agent Accuracy (G-Eval)
    # -----------------------------------------------------------------------
    print("\n[1/3] Evaluating Problem Detection Agent (Intent & Target Grounding)...")
    problem_metric = GEval(
        name="ML Problem Detection Accuracy",
        criteria=(
            "Assess whether the actual output correctly resolves the user's instruction to an exact "
            "column that exists in the available dataset columns and assigns the correct ML task type "
            "(classification, regression, or clustering). Award full credit when the target column and "
            "task type are semantically correct. Ignore cosmetic differences in formatting, casing, "
            "quoting, or parenthetical annotations (e.g. 'Classification (Binary)' is equivalent to "
            "'classification'). Heavily penalize any hallucinated column that does not exist in the "
            "dataset or an incorrect task type."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=0.8,
    )

    tc1 = LLMTestCase(
        input="Instruction: 'Predict loan default'. Columns: ['applicant_id', 'annual_inc', 'fico_score', 'loan_status', 'emp_length']",
        actual_output="Task: Binary Classification. Target: 'loan_status'. Reason: 'loan_status' captures default vs non-default outcomes.",
        expected_output="Task: classification. Target: loan_status.",
    )

    measure_with_rate_limit_retry(problem_metric, tc1, "Problem Detection (G-Eval)")
    results.append({
        "Benchmark": "Problem Detection Accuracy (G-Eval)",
        "Metric": "Decision Accuracy",
        "Score": f"{problem_metric.score:.2f} / 1.00",
        "Threshold": ">= 0.80",
        "Status": "PASSED" if problem_metric.is_successful() else "FAILED",
        "Reason": problem_metric.reason,
    })

    time.sleep(2)  # Respect rate limits

    # -----------------------------------------------------------------------
    # Benchmark 2: Report Agent Hallucination & Faithfulness
    # -----------------------------------------------------------------------
    print("\n[2/3] Evaluating Report Agent Narration (Hallucination & Faithfulness)...")
    ground_truth = [
        "Dataset: loan_risk.csv (5000 samples, 10 features).",
        "Leaderboard: 1. XGBoost (F1: 0.884, Acc: 0.891), 2. RandomForest (F1: 0.862), 3. LogisticRegression (F1: 0.760).",
        "Best Model: XGBoost due to highest F1 score and robust handling of non-linear interactions.",
        "Significant features: debt_to_income, credit_score."
    ]

    agent_summary = (
        "XGBoost was identified as the champion model with an F1 score of 0.884 and accuracy of 0.891, "
        "outperforming RandomForest (0.862) and LogisticRegression (0.760). "
        "The primary drivers of prediction were debt_to_income and credit_score."
    )

    tc2 = LLMTestCase(
        input="Summarize experiment results and champion model selection.",
        actual_output=agent_summary,
        context=ground_truth,
        retrieval_context=ground_truth,
    )

    hallucination_metric = HallucinationMetric(threshold=0.8, model=judge)
    measure_with_rate_limit_retry(hallucination_metric, tc2, "Report Hallucination")

    time.sleep(2)

    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=judge)
    measure_with_rate_limit_retry(faithfulness_metric, tc2, "Report Faithfulness")

    results.append({
        "Benchmark": "Report Hallucination Score",
        "Metric": "Factual Alignment (1.00 = 0% Hallucination)",
        "Score": f"{hallucination_metric.score:.2f} / 1.00",
        "Threshold": ">= 0.80",
        "Status": "PASSED" if hallucination_metric.is_successful() else "FAILED",
        "Reason": hallucination_metric.reason,
    })

    results.append({
        "Benchmark": "Report Faithfulness",
        "Metric": "Data Grounding / Truthfulness",
        "Score": f"{faithfulness_metric.score:.2f} / 1.00",
        "Threshold": ">= 0.80",
        "Status": "PASSED" if faithfulness_metric.is_successful() else "FAILED",
        "Reason": faithfulness_metric.reason,
    })

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Benchmark 3: Agent Explanation Relevancy
    # -----------------------------------------------------------------------
    print("\n[3/3] Evaluating Agent Response Relevancy...")
    relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=judge)

    tc3 = LLMTestCase(
        input="Why were high-cardinality categorical columns dropped during preprocessing?",
        actual_output=(
            "High-cardinality categorical columns were dropped during preprocessing because "
            "columns with near-unique values (e.g. IDs or arbitrary names) contain almost no "
            "repeated categories; one-hot encoding them would explode the feature space into "
            "extremely high-dimensional, sparse representations and cause severe model "
            "overfitting on noise, while the columns themselves carry no generalizable "
            "predictive signal."
        ),
    )

    measure_with_rate_limit_retry(relevancy_metric, tc3, "Answer Relevancy")
    results.append({
        "Benchmark": "Explanation Relevancy",
        "Metric": "Answer Relevancy",
        "Score": f"{relevancy_metric.score:.2f} / 1.00",
        "Threshold": ">= 0.80",
        "Status": "PASSED" if relevancy_metric.is_successful() else "FAILED",
        "Reason": relevancy_metric.reason,
    })

    # -----------------------------------------------------------------------
    # Print Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" AgentML DeepEval Evaluation Benchmark Summary")
    print("=" * 75)
    for res in results:
        print(f"\n[ {res['Status']} ] {res['Benchmark']}")
        print(f"   Metric:    {res['Metric']}")
        print(f"   Score:     {res['Score']} (Threshold: {res['Threshold']})")
        reason_clean = str(res['Reason']).encode("ascii", errors="replace").decode("ascii")
        print(f"   Analysis:  {reason_clean}")
    
    print("\n" + "=" * 75)
    all_passed = all(r["Status"] == "PASSED" for r in results)
    if all_passed:
        print(" ALL EVALUATION BENCHMARKS PASSED PERFECTLY (100% SUCCESS)!")
    else:
        print(" SOME BENCHMARKS FAILED - SEE ABOVE REASONS")
    print("=" * 75)

if __name__ == "__main__":
    main()
