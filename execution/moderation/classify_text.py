"""
execution/moderation/classify_text.py
--------------------------------------
Layer 3 — Deterministic text toxicity classifier.

Reads directive: directives/moderation_classify_text.md

Usage:
    python -m execution.moderation.classify_text \
        --text "some user text" \
        --user-id u_123 \
        [--language hi] \
        [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a module from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from execution.utils.logger import get_logger
from execution.utils.validator import validate_text_input
from execution.utils.rate_limiter import RateLimiter, RateLimitExceeded

log = get_logger(__name__)

# Shared rate limiter instance (in-process; swap for Redis in prod)
_rate_limiter = RateLimiter(
    max_calls=int(os.getenv("RATE_LIMIT_PER_MINUTE", 60)),
    period_seconds=60,
)

# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_text(text: str, user_id: str, language: str = "en", dry_run: bool = False) -> dict:
    """
    Classify a text snippet for toxicity using the configured LLM.

    Args:
        text:      Raw user text to classify.
        user_id:   Caller identity for rate-limiting and audit.
        language:  ISO 639-1 language code.
        dry_run:   If True, skip the LLM call and return a mock result.

    Returns:
        dict with keys: is_toxic, confidence, categories, reasoning,
                        user_id, processed_at, language_mismatch (optional).

    Raises:
        RateLimitExceeded: If the user has exceeded their request quota.
        ValueError:        If validation fails.
        RuntimeError:      If the LLM call fails after all retries.
    """
    # --- Step 1: Validate ---
    payload = validate_text_input({"text": text, "user_id": user_id, "language": language})
    log.info("Input validated for user=%s len=%d", payload.user_id, len(payload.text))

    # --- Step 2: Rate limit ---
    _rate_limiter.check_or_raise(payload.user_id)
    log.info("Rate limit OK for user=%s", payload.user_id)

    # --- Step 3: Classify (LLM call with retry) ---
    if dry_run:
        log.info("DRY RUN — skipping LLM call")
        result = {
            "is_toxic": False,
            "confidence": 0.0,
            "categories": [],
            "reasoning": "Dry run — no LLM call made.",
        }
    else:
        result = _call_llm_with_retry(payload.text, payload.language)

    # --- Step 4: Build output ---
    output = {
        **result,
        "user_id": payload.user_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    log.info("Classification complete user=%s is_toxic=%s", payload.user_id, output["is_toxic"])
    return output


def _call_llm_with_retry(text: str, language: str, max_retries: int = 3) -> dict:
    """
    Call the LLM classification API with exponential backoff on failure.

    Args:
        text:        Validated, stripped text to classify.
        language:    Language code for prompt context.
        max_retries: Number of retry attempts before raising.

    Returns:
        dict with is_toxic, confidence, categories, reasoning.

    Raises:
        RuntimeError: After all retries are exhausted.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No LLM API key configured. Set OPENAI_API_KEY or GEMINI_API_KEY in .env")

    # TODO: Replace stub with real LLM client call.
    # Example using openai-python:
    #   from openai import OpenAI
    #   client = OpenAI(api_key=api_key)
    #   response = client.chat.completions.create(...)

    for attempt in range(1, max_retries + 1):
        try:
            log.debug("LLM call attempt %d/%d", attempt, max_retries)
            # --- STUB — replace with real API call ---
            return {
                "is_toxic": False,
                "confidence": 0.0,
                "categories": [],
                "reasoning": "LLM call stub — implement _call_llm_with_retry body.",
            }
            # --- END STUB ---
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** (attempt - 1)
            log.warning("LLM attempt %d failed: %s. Retrying in %ds...", attempt, exc, wait)
            if attempt == max_retries:
                raise RuntimeError(f"LLM classification failed after {max_retries} attempts.") from exc
            time.sleep(wait)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ToxiClean AI — text toxicity classifier")
    parser.add_argument("--text", required=True, help="Text to classify (max 10,000 chars)")
    parser.add_argument("--user-id", required=True, help="Caller user ID")
    parser.add_argument("--language", default="en", help="ISO 639-1 language code")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM call; return mock result")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        result = classify_text(
            text=args.text,
            user_id=args.user_id,
            language=args.language,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
    except RateLimitExceeded as e:
        log.error("Rate limit exceeded: %s", e)
        sys.exit(429)
    except ValueError as e:
        log.error("Validation error: %s", e)
        sys.exit(400)
    except RuntimeError as e:
        log.error("Execution error: %s", e)
        sys.exit(500)
