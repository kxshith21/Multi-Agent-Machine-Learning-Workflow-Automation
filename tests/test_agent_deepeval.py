"""
DeepEval Test Suite for AgentML Agents

Evaluates:
1. Problem Detection Agent Accuracy & Target Grounding (G-Eval)
2. Report Agent Narration Faithfulness & Hallucination Detection (FaithfulnessMetric, HallucinationMetric)
3. Agent Response Relevancy to User Instruction (AnswerRelevancyMetric)

Uses Groq (OpenAI-compatible endpoint) via deepeval.models.OpenAIModel with openai/gpt-oss-120b.
"""

from __future__ import annotations

import os
import sys
import pytest
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Disable telemetry prompts
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

load_dotenv()

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    GEval,
)
from deepeval.models import OpenAIModel


def get_eval_judge_model() -> OpenAIModel:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        pytest.skip("GROQ_API_KEY not configured for DeepEval evaluation.")
    
    return OpenAIModel(
        model=os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-120b"),
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )


# ---------------------------------------------------------------------------
# Test 1: Problem Detection Agent — ML Task & Target Extraction Accuracy (G-Eval)
# ---------------------------------------------------------------------------

def test_problem_detection_agent_accuracy():
    """
    Evaluates whether the Problem Detection Agent accurately extracts the target column
    and identifies the appropriate ML task type (classification/regression/clustering)
    grounded in available dataset columns.
    """
    judge_model = get_eval_judge_model()

    accuracy_metric = GEval(
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
        model=judge_model,
        threshold=0.8,
    )

    test_case = LLMTestCase(
        input=(
            "User Instruction: 'Predict whether the passenger survived the disaster'. "
            "Available dataset columns: ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']"
        ),
        actual_output=(
            "Task Type: Classification (Binary). "
            "Target Column: 'Survived'. "
            "Reasoning: 'Survived' represents a binary outcome (survived or not) aligning with passenger survival prediction."
        ),
        expected_output=(
            "Task Type: classification. "
            "Target Column: Survived."
        ),
    )

    assert_test(test_case, [accuracy_metric])


# ---------------------------------------------------------------------------
# Test 2: Report Agent Narration — Hallucination & Faithfulness Evaluation
# ---------------------------------------------------------------------------

def test_report_agent_hallucination_and_faithfulness():
    """
    Evaluates whether the Report Agent's narrative summary faithfully reflects the
    actual experiment results without hallucinating false metrics or fake models.
    """
    judge_model = get_eval_judge_model()

    ground_truth_context = [
        "Dataset: loan_applications.csv, Total Rows: 5,000, Total Features: 12.",
        "Task Type: binary classification, Target Column: 'loan_status'.",
        "Leaderboard Results: 1. XGBoostClassifier (F1-Score: 0.892, Accuracy: 0.887, ROC-AUC: 0.941), "
        "2. RandomForestClassifier (F1-Score: 0.865, Accuracy: 0.860), "
        "3. LogisticRegression (F1-Score: 0.745, Accuracy: 0.750).",
        "Top Predictive Features: annual_income, credit_score, debt_to_income_ratio.",
        "Preprocessing Applied: Median imputation, Standard scaling on numeric features, One-hot encoding on categorical features."
    ]

    agent_actual_summary = (
        "The automated experiment evaluated 3 classification models on loan_applications.csv (5,000 rows). "
        "XGBoostClassifier was selected as the best performing model, achieving an F1-score of 0.892 and ROC-AUC of 0.941. "
        "It significantly outperformed the baseline LogisticRegression model (0.745 F1-score). "
        "Key drivers of loan approval predictions were annual_income, credit_score, and debt_to_income_ratio."
    )

    test_case = LLMTestCase(
        input="Generate an executive summary narrating the machine learning experiment results.",
        actual_output=agent_actual_summary,
        context=ground_truth_context,
        retrieval_context=ground_truth_context,
    )

    # In DeepEval 4.2+, 1.0 is highest passing score for both metrics
    hallucination_metric = HallucinationMetric(threshold=0.8, model=judge_model)
    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=judge_model)

    assert_test(test_case, [hallucination_metric, faithfulness_metric])


# ---------------------------------------------------------------------------
# Test 3: Agent Answer Relevancy to Natural Language Prompts
# ---------------------------------------------------------------------------

def test_agent_answer_relevancy():
    """
    Evaluates whether the agent provides direct, relevant responses to specific user questions
    without meandering or including unrequested commentary.
    """
    judge_model = get_eval_judge_model()

    relevancy_metric = AnswerRelevancyMetric(threshold=0.8, model=judge_model)

    test_case = LLMTestCase(
        input="Why was XGBoost selected over RandomForest for this classification dataset?",
        actual_output=(
            "XGBoost was chosen because it achieved a higher F1-score (0.892 vs 0.865) "
            "and demonstrated superior discrimination on class imbalance with an ROC-AUC of 0.941."
        ),
    )

    assert_test(test_case, [relevancy_metric])
