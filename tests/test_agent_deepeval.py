"""
DeepEval Test Suite for AgentML Agents

Evaluates every pipeline stage that produces a reportable decision:
1. Problem Detection Agent Accuracy & Target Grounding (G-Eval)
2. Report Agent Narration Faithfulness & Hallucination Detection (FaithfulnessMetric, HallucinationMetric)
3. Agent Response Relevancy to User Instruction (AnswerRelevancyMetric)
4. Dataset Profiling Agent Profile Correctness (G-Eval)
5. Feature-Engineering Suggestion Accuracy (G-Eval)
6. Data Preprocessing Agent Transformation Correctness (G-Eval)
7. Experiment Orchestrator Integrity (G-Eval)
8. Model Evaluation Ranking Decision Correctness (G-Eval)

Uses Groq (OpenAI-compatible endpoint) via deepeval.models.OpenAIModel with openai/gpt-oss-120b.
"""

from __future__ import annotations

import os
import sys
import time
import pytest
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
from openai import RateLimitError as OpenAIRateLimitError


def get_eval_judge_model() -> OpenAIModel:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        pytest.skip("GROQ_API_KEY not configured for DeepEval evaluation.")
    
    return OpenAIModel(
        model=os.getenv("GROQ_EVAL_MODEL", "openai/gpt-oss-120b"),
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
    )


def _is_rate_limit_error(exc) -> bool:
    """True if exc (or its raise-from cause chain) is an API rate-limit error."""
    seen: set[int] = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, OpenAIRateLimitError):
            return True
        if (
            "RateLimitError" in type(cur).__name__
            or "429" in str(cur)
            or "rate limit" in str(cur).lower()
        ):
            return True
        cur = cur.__cause__
    return False


def assert_test_with_rate_limit_retry(test_case, metrics, label, max_attempts=5, wait_s=45):
    """
    Measures every metric and asserts success, retrying on the free Groq tier's
    TPM limit (429). Running all eight benchmarks back-to-back easily exceeds the
    8000-token/minute window, so this mirrors the runner script's policy:
    `metric.measure()` + rate-limit backoff, with a short quiet period between
    judge calls so the whole file stays green in a single pytest invocation.
    """
    _PACING_GAP = 6.0
    if hasattr(assert_test_with_rate_limit_retry, "_last_measure"):
        elapsed = time.monotonic() - assert_test_with_rate_limit_retry._last_measure
        if elapsed < _PACING_GAP:
            time.sleep(_PACING_GAP - elapsed)
    for attempt in range(1, max_attempts + 1):
        try:
            for metric in metrics:
                metric.measure(test_case)
            for metric in metrics:
                assert metric.is_successful(), (
                    f"{label} failed on '{metric.__class__.__name__}': "
                    f"score={metric.score:.2f} threshold={metric.threshold} "
                    f"reason={metric.reason}"
                )
            return
        except AssertionError:
            raise
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == max_attempts:
                raise
            print(
                f"  Rate limit hit during {label} (attempt {attempt}/{max_attempts}). "
                f"Waiting {wait_s}s before retrying..."
            )
            time.sleep(wait_s)
    raise RuntimeError(f"Exhausted rate-limit retries for {label}.")


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

    assert_test_with_rate_limit_retry(test_case, [accuracy_metric], "Problem Detection (G-Eval)")


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

    assert_test_with_rate_limit_retry(
        test_case, [hallucination_metric, faithfulness_metric], "Report Narration"
    )


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

    assert_test_with_rate_limit_retry(test_case, [relevancy_metric], "Answer Relevancy")


# ---------------------------------------------------------------------------
# Test 4: Dataset Profiling Agent — Profile Correctness (G-Eval)
# ---------------------------------------------------------------------------

def test_profiling_agent_profile_correctness():
    """
    Evaluates whether the Profiling Agent's dataset profile (shape, dtypes,
    missing %, duplicate count, numeric/categorical split) is accurate and
    grounded in the actual dataset — no invented rows, columns, or stats.
    """
    judge_model = get_eval_judge_model()

    profile_metric = GEval(
        name="Dataset Profile Correctness",
        criteria=(
            "Assess whether the actual output correctly reports the dataset profile: total rows, "
            "total columns, duplicate-row count, percentage of missing values, and which columns are "
            "numeric vs categorical. Award full credit when every stated number matches the expected "
            "output for the given dataset. Ignore cosmetic differences in formatting or column ordering. "
            "Heavily penalize any hallucinated number (wrong row/column/duplicate/missing counts), any "
            "column that does not exist, or any numeric column misclassified as categorical (or vice versa)."
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
            "Dataset: loan_applications.csv — a loan-approval dataset with columns "
            "['age', 'income', 'credit_score', 'debt_ratio', 'years_employed', 'num_accounts', 'approved']. "
            "Generate a full dataset profile."
        ),
        actual_output=(
            "Dataset Profile for loan_applications.csv: 300 rows, 7 columns, 0 duplicate rows, "
            "0% missing values across all columns. Numeric columns: age, income, credit_score, "
            "debt_ratio, years_employed, num_accounts, approved. Categorical columns: none. "
            "All seven columns are numeric (int64 or float64)."
        ),
        expected_output=(
            "Dataset Profile for loan_applications.csv: 300 rows, 7 columns, 0 duplicate rows, "
            "0% missing values. Numeric columns: age, income, credit_score, debt_ratio, "
            "years_employed, num_accounts, approved. Categorical columns: none."
        ),
    )

    assert_test_with_rate_limit_retry(test_case, [profile_metric], "Profiling (G-Eval)")


# ---------------------------------------------------------------------------
# Test 5: Feature-Engineering Suggestions — Suggestion Accuracy (G-Eval)
# ---------------------------------------------------------------------------

def test_feature_engineering_suggestion_accuracy():
    """
    Evaluates whether the Profiling Agent's feature-engineering suggestions
    (datetime decompose, correlated-pair ratio/product, high-cardinality binning)
    are appropriate for the described dataset and never hallucinated.
    """
    judge_model = get_eval_judge_model()

    suggestion_metric = GEval(
        name="Feature-Engineering Suggestion Accuracy",
        criteria=(
            "Assess whether the actual output lists feature-engineering suggestions that match the "
            "dataset's true characteristics. A datetime column should yield a datetime_decompose "
            "suggestion; a strongly correlated numeric pair (|r| >= 0.7) should yield a ratio or "
            "product suggestion; a high-cardinality categorical column should yield a binning "
            "suggestion. Award full credit when every suggestion type targets an appropriate column. "
            "Heavily penalize hallucinated suggestions (targeting columns that don't exist or proposing "
            "a transformation type that does not fit the data, e.g. binning a low-cardinality column or "
            "ratio on uncorrelated columns). Ignore cosmetic phrasing differences."
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
            "Dataset columns: application_date (datetime, unique per row), income (numeric), "
            "credit_score (numeric, correlated with income at r=0.82), job_title (categorical with "
            "312 unique values), approved (target). Suggest feature-engineering opportunities."
        ),
        actual_output=(
            "Suggested features: "
            "(1) application_date_decomposed — datetime_decompose on 'application_date' (split into "
            "year/month/day/weekday); "
            "(2) income_per_credit_score — ratio on the strongly correlated pair income/credit_score "
            "(|r| = 0.82); "
            "(3) job_title_binned — binning on the high-cardinality categorical column 'job_title' "
            "(312 unique values). "
            "No suggestions target the target column 'approved'."
        ),
        expected_output=(
            "Suggestions: application_date → datetime_decompose; income & credit_score → ratio; "
            "job_title → binning."
        ),
    )

    assert_test_with_rate_limit_retry(test_case, [suggestion_metric], "FE Suggestions (G-Eval)")


# ---------------------------------------------------------------------------
# Test 6: Data Preprocessing Agent — Transformation Correctness (G-Eval)
# ---------------------------------------------------------------------------

def test_preprocessing_agent_transformation_correctness():
    """
    Evaluates whether the Preprocessing Agent's transformation log applies the
    correct strategy per column (dedup, impute, one-hot encode, scale, drop)
    while always preserving the resolved target column unscaled.
    """
    judge_model = get_eval_judge_model()

    transformation_metric = GEval(
        name="Preprocessing Transformation Correctness",
        criteria=(
            "Assess whether the actual output describes preprocessing transformations that are "
            "correct for the dataset: duplicate rows removed; missing numeric values imputed with the "
            "median; missing categorical values imputed with the mode; categorical features one-hot "
            "encoded; numeric features standard-scaled (zero mean, unit variance); unusable free-text / "
            "unique-ID columns dropped; and the resolved target column preserved as-is without scaling "
            "or encoding. Award full credit when every stated operation is appropriate. Heavily penalize "
            "incorrect strategies (e.g. scaling or encoding the target column), dropping a useful "
            "numeric/categorical feature without justification, or claiming operations not applied. "
            "Ignore cosmetic phrasing differences."
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
            "Preprocess loan_applications.csv. Target column: 'approved'. The dataset has 2,000 rows "
            "with 40 duplicate rows, 'age' has 12 missing values, 'education' is categorical, "
            "numeric features are 'income', 'credit_score', 'debt_ratio'; 'notes' is free-text and "
            "largely unique per row."
        ),
        actual_output=(
            "Preprocessing log: removed 40 duplicate rows; imputed 12 missing values in 'age' with the "
            "median; one-hot encoded the categorical column 'education'; standard-scaled the numeric "
            "features 'income', 'credit_score', 'debt_ratio'; dropped the uninformative free-text column "
            "'notes'; preserved the target column 'approved' as-is (never scaled or encoded)."
        ),
        expected_output=(
            "Preprocessing: deduplicate (40 rows); impute 'age' with median; one-hot encode 'education'; "
            "standard-scale numeric features; drop free-text 'notes'; keep target 'approved' untouched."
        ),
    )

    assert_test_with_rate_limit_retry(test_case, [transformation_metric], "Preprocessing (G-Eval)")


# ---------------------------------------------------------------------------
# Test 7: Experiment Orchestrator — Integrity (G-Eval)
# ---------------------------------------------------------------------------

def test_experiment_orchestrator_integrity():
    """
    Evaluates whether the Experiment Orchestrator logs all model-zoo runs with
    their parameters and metrics, isolates failures (a single crash does not
    abort the run), and applies imbalanced-data parameters correctly.
    """
    judge_model = get_eval_judge_model()

    integrity_metric = GEval(
        name="Experiment Orchestrator Integrity",
        criteria=(
            "Assess whether the actual output truthfully describes the experiment run: every model in the "
            "fixed zoo was executed and logged with its parameters and metrics; imbalanced-classification "
            "models carried imbalance-aware parameters — class_weight='balanced' on Logistic Regression / "
            "Random Forest / SVC and a runtime-computed scale_pos_weight (> 1.0) on XGBoost; a model that "
            "failed to run was recorded as a failure and skipped WITHOUT aborting the remaining runs. Award "
            "full credit when the parameter claims and the failure-isolation claim are correct. Heavily "
            "penalize false claims (e.g. claiming all runs succeeded when one failed, missing parameters, "
            "or a scale_pos_weight of 1.0 on imbalanced data). Ignore cosmetic phrasing differences."
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
            "Run the classification model zoo (DummyClassifier baseline, LogisticRegression, "
            "KNeighbors, RandomForest, GradientBoosting, SVC, XGBoost) on an imbalanced 95/5 dataset "
            "with 600 rows."
        ),
        actual_output=(
            "All 7 classification zoo models were executed and logged with results. Imbalance-aware "
            "parameters were recorded per model: LogisticRegression, RandomForest and SVC used "
            "class_weight='balanced'; XGBoost used scale_pos_weight = 18.05 computed from the training "
            "fold (majority/minority ratio > 1.0). One run (KNeighborsClassifier) failed during fit and "
            "was logged as a failure and excluded from ranking — the other 6 runs completed normally; "
            "the orchestrator did not abort."
        ),
        expected_output=(
            "7 models run and logged; class_weight='balanced' on LogisticRegression / RandomForest / SVC; "
            "XGBoost scale_pos_weight > 1.0; 1 failed run isolated without aborting the rest."
        ),
    )

    assert_test_with_rate_limit_retry(test_case, [integrity_metric], "Orchestrator (G-Eval)")


# ---------------------------------------------------------------------------
# Test 8: Model Evaluation — Ranking Decision Correctness (G-Eval)
# ---------------------------------------------------------------------------

def test_model_evaluation_ranking_correctness():
    """
    Evaluates whether the Model Evaluation agent ranks classification by F1
    (macro) rather than accuracy, uses precision as the tiebreaker, applies the
    imbalance-aware explanation when a minority class is <10%, and selects the
    rank-1 model as best_model_id.
    """
    judge_model = get_eval_judge_model()

    ranking_metric = GEval(
        name="Model Evaluation Ranking Correctness",
        criteria=(
            "Assess whether the actual output describes a correct model-ranking decision for a "
            "binary classification task: the ranking primary metric must be f1 (NOT accuracy); the "
            "selected best model must be the one ranked first on f1; the tiebreaker metric must be "
            "precision; and when the minority class represents less than 10% of samples, the reasoning "
            "must explicitly state that F1/PR-AUC were prioritized over accuracy (a majority-class-only "
            "model would otherwise look good on accuracy while scoring near-zero F1). Award full credit "
            "when the stated primary metric, winner, and imbalance explanation are all correct. Heavily "
            "penalize ranking by accuracy, picking a non-first model, or missing the imbalance "
            "explanation on an imbalanced dataset. Ignore cosmetic phrasing differences."
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
            "Evaluate the classification leaderboard for an imbalanced 95/5 dataset. Class 0 = 94.17%, "
            "class 1 = 5.83%. F1 scores: XGBClassifier 0.526, GradientBoosting 0.462, SVC 0.320, "
            "LogisticRegression 0.233. Accuracy: DummyClassifier baseline 0.940, XGBClassifier 0.940, "
            "GradientBoosting 0.953. Select the best model and explain the ranking."
        ),
        actual_output=(
            "Selected 'XGBClassifier_n100' as the best model: it ranked first out of 7 successful "
            "experiments on the primary metric f1 = 0.5263 with a precision tiebreaker of 0.5000. "
            "Because the minority class represents only 5.8% of samples (imbalanced target), F1 and "
            "PR-AUC were prioritized over accuracy for ranking — the DummyClassifier baseline scored "
            "0.940 accuracy but f1 = 0.000 with pr_auc = 0.060, so a majority-class-only model would "
            "otherwise look artificially good."
        ),
        expected_output=(
            "Best model: XGBClassifier_n100. Classification ranked by f1 (not accuracy); tiebreaker "
            "precision; minority 5.8% < 10% so F1/PR-AUC prioritized over accuracy."
        ),
    )

    assert_test_with_rate_limit_retry(test_case, [ranking_metric], "Evaluation (G-Eval)")
