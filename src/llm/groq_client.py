"""
Groq LLM client abstraction.

ALL calls to the Groq API must go through this module.
No agent file may import `groq` directly — this is a hard rule (Rules.md §3).

Responsibilities:
  - Load GROQ_API_KEY from environment (via python-dotenv).
  - Provide a single `chat()` function that wraps groq.Groq.chat.completions.create().
  - Enforce the LLM-use boundary: the client accepts only plain text prompts
    and returns plain text — it does not make any ML decisions.
  - Implement retry logic with exponential back-off for transient failures.
  - Log every call (prompt hash, model, tokens used) to state["errors"] via
    the returned error list — callers decide what to do with errors.

Public API:
    chat(prompt: str, system: str | None, model: str | None,
         temperature: float | None) -> tuple[str, list[ErrorEntry]]

    hello_world() -> str   # smoke-test; returns model's response to a greeting
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
import groq as groq_sdk

from src.orchestrator.state import ErrorEntry

# Load .env if present (no-op in environments where vars are already set)
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read once at module load; override via env vars)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
_DEFAULT_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0  # seconds; doubles on each retry


# ---------------------------------------------------------------------------
# Internal: build a Groq client instance
# ---------------------------------------------------------------------------

def _get_client() -> groq_sdk.Groq:
    """
    Create and return a Groq client.

    Raises:
        EnvironmentError: if GROQ_API_KEY is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key from https://console.groq.com"
        )
    return groq_sdk.Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> tuple[str, list[ErrorEntry]]:
    """
    Send a chat completion request to Groq.

    The LLM is used ONLY for narration / explanation text.
    All ML decisions (task type, preprocessing, model selection, ranking)
    are made rule-based upstream; this function just asks the model to
    describe those decisions in plain English.

    Args:
        prompt:      The user-turn message (the thing to narrate/explain).
        system:      Optional system prompt. Defaults to a neutral narrator role.
        model:       Groq model identifier. Defaults to GROQ_MODEL env var.
        temperature: Sampling temperature. Defaults to GROQ_TEMPERATURE env var.

    Returns:
        (response_text, errors)
        response_text — the model's reply as a plain string.
        errors        — list of ErrorEntry dicts; empty on full success.
                        Callers should append these to state["errors"].
    """
    errors: list[ErrorEntry] = []
    resolved_model = model or _DEFAULT_MODEL
    resolved_temp = temperature if temperature is not None else _DEFAULT_TEMPERATURE
    resolved_system = system or (
        "You are a concise, plain-spoken ML experiment narrator. "
        "Explain decisions in 2-4 sentences. No hedging language. No bullet points."
    )

    messages = [
        {"role": "system", "content": resolved_system},
        {"role": "user", "content": prompt},
    ]

    # Log call (hash the prompt so we don't store PII-adjacent data in logs)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    logger.debug("Groq call | model=%s | prompt_hash=%s", resolved_model, prompt_hash)

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=resolved_model,
                messages=messages,
                temperature=resolved_temp,
                max_tokens=512,
            )
            text = response.choices[0].message.content or ""
            logger.debug(
                "Groq response | attempt=%d | tokens=%s",
                attempt,
                response.usage.total_tokens if response.usage else "?",
            )
            return text.strip(), errors

        except groq_sdk.AuthenticationError as exc:
            # Hard fail — no retry; wrong key won't fix itself.
            errors.append(
                ErrorEntry(
                    phase="groq_client",
                    error_type="groq_auth_error",
                    message=f"Groq authentication failed: {exc}",
                    recoverable=False,
                )
            )
            return "", errors

        except (groq_sdk.RateLimitError, groq_sdk.InternalServerError) as exc:
            last_exc = exc
            wait = _BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "Groq transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.error("Groq unexpected error: %s", exc)
            break

    # All retries exhausted or unexpected exception
    errors.append(
        ErrorEntry(
            phase="groq_client",
            error_type="groq_api_error",
            message=f"Groq API call failed after {_MAX_RETRIES} attempts: {last_exc}",
            recoverable=True,   # narration failure → log-and-continue
        )
    )
    return "", errors


def hello_world() -> str:
    """
    Smoke-test: send a simple greeting and return the model's plain-text reply.

    Raises:
        EnvironmentError: if GROQ_API_KEY is not configured.
        RuntimeError:     if the API call fails after all retries.
    """
    text, errors = chat(
        prompt="Say hello in exactly one sentence.",
        system="You are a friendly assistant.",
        temperature=0.0,
    )
    hard_errors = [e for e in errors if not e["recoverable"]]
    if hard_errors:
        raise RuntimeError(hard_errors[0]["message"])
    if errors:
        raise RuntimeError(errors[0]["message"])
    return text
