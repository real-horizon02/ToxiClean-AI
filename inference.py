"""
inference.py

ToxiClean AI — Baseline inference script for the OpenEnv hackathon evaluation.

MANDATORY ENVIRONMENT VARIABLES
---------------------------------
    API_BASE_URL   The API endpoint for the LLM.
                   Default: https://router.huggingface.co/v1
    MODEL_NAME     The model identifier to use for inference.
                   Default: Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (STRICTLY ENFORCED)
----------------------------------
One [START] line at episode begin:
    [START] task=<task_name> env=<benchmark> model=<model_name>

One [STEP] line per step:
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>

One [END] line after episode closes (always emitted):
    [END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

    Rules:
      - reward/rewards formatted to 2 decimal places
      - score formatted to 3 decimal places
      - done and success are lowercase booleans: true or false
      - error is the raw error string, or null if none
      - ALL fields on a SINGLE line with no newlines within a line

Usage
-----
    python inference.py
    python inference.py --task spam_detection
    python inference.py --task toxicity_classification
    python inference.py --task contextual_moderation
    python inference.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from core.environment import ToxiCleanEnv
from core.models import ModerationAction, Observation

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()  # Load .env if present

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("toxiclean.inference")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_TASKS = ["spam_detection", "toxicity_classification", "contextual_moderation"]
BENCHMARK = "toxiclean"

# ---------------------------------------------------------------------------
# Mandatory environment variables (per submission spec)
# Defaults set ONLY for API_BASE_URL and MODEL_NAME — NOT for HF_TOKEN
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be set in environment

# Optional — only needed if using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

# System prompt that shapes the agent's moderation persona
_SYSTEM_PROMPT = """You are a professional content moderation AI agent.

Your job is to analyse a piece of content and decide the correct moderation action.

=== ACTION SPACE ===
ALLOW    – The content is completely safe. No action needed.
FLAG     – The content is suspicious or mildly problematic. Queue for human review.
DELETE   – The content is clearly harmful, abusive, or violates policies. Remove it.
ESCALATE – The content poses a critical threat (violence, self-harm, doxxing). Escalate to the safety team immediately.

=== DECISION GUIDELINES ===
- Read the content carefully, including any metadata provided.
- Consider idiomatic language, sarcasm, and cultural context.
- Do NOT over-moderate safe content.
- Do NOT under-moderate harmful content.
- Repeat offenders should be treated more strictly.

=== OUTPUT FORMAT (STRICT) ===
Respond with a single JSON object — NO markdown, NO extra text:
{
  "action": "<ALLOW|FLAG|DELETE|ESCALATE>",
  "reasoning": "<one-sentence explanation>"
}
"""


# ---------------------------------------------------------------------------
# Mandatory stdout log helpers (single-line format per spec)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] line. Must be a single line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit a [STEP] line. Must be a single line."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the [END] line. Must be a single line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    """Construct an OpenAI-compatible client from environment variables.
    
    Uses HF_TOKEN as the API key (mandatory, no default).
    API_BASE_URL and MODEL_NAME have defaults per spec.
    """
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. "
            "Add it to your .env file or export it as an environment variable.\n"
            "  export HF_TOKEN=hf_your_token_here"
        )
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL.rstrip("/"))


def _observation_to_user_message(obs: Observation) -> str:
    """Convert an Observation into a user-facing prompt string for the LLM."""
    meta = obs.metadata
    return (
        f"=== CONTENT TO MODERATE ===\n"
        f"{obs.content}\n\n"
        f"=== CONTEXT ===\n"
        f"Platform      : {meta.platform}\n"
        f"Language      : {meta.language}\n"
        f"User History  : {meta.user_history}\n"
        f"Task          : {obs.task_name}\n"
        f"Step          : {obs.step_index + 1}\n\n"
        f"Respond with the JSON action object only."
    )


def _call_llm(
    client: OpenAI,
    model: str,
    user_message: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Call the LLM and parse the JSON response.

    Retries up to `max_retries` times on transient errors.
    Falls back to ALLOW on persistent failure (conservative fail-safe).
    """
    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,   # Deterministic for reproducibility
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)

            # Validate action field
            action_str = parsed.get("action", "ALLOW").upper().strip()
            ModerationAction(action_str)  # raises ValueError if invalid
            return parsed

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            last_error = str(exc)
            logger.warning("Attempt %d: parse error — %s", attempt, exc)
            if attempt == max_retries:
                logger.error("All retries exhausted. Defaulting to ALLOW.")
                return {"action": "ALLOW", "reasoning": "Parse failure — default safe.", "_error": last_error}
            time.sleep(0.5 * attempt)

        except Exception as exc:  # network/API errors
            last_error = str(exc)
            logger.warning("Attempt %d: API error — %s", attempt, exc)
            if attempt == max_retries:
                logger.error("All retries exhausted. Defaulting to ALLOW.")
                return {"action": "ALLOW", "reasoning": "API failure — default safe.", "_error": last_error}
            time.sleep(1.0 * attempt)

    return {"action": "ALLOW", "reasoning": "Unexpected exit — default safe."}


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    client: OpenAI,
    model: str,
    verbose: bool = False,
) -> float:
    """
    Run a single task episode and print logs in the required evaluator format.

    Returns
    -------
    float
        Final normalised score for the episode (clamped to [0, 1]).
    """
    env = ToxiCleanEnv(task_name=task_name)
    obs = env.reset()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # ---- [START] line ----
    log_start(task=task_name, env=BENCHMARK, model=model)

    try:
        step = 0
        last_error: Optional[str] = None

        while True:
            state = env.state()
            if state.done:
                break

            step += 1

            # Build the LLM prompt from the current observation
            user_message = _observation_to_user_message(obs)

            # Get action from agent
            llm_response = _call_llm(client, model, user_message)
            action_str = llm_response.get("action", "ALLOW").upper().strip()
            reasoning = llm_response.get("reasoning", "")
            last_error = llm_response.get("_error", None)

            if verbose:
                print(f"[DEBUG] step={step} content={obs.content[:60]!r}...", flush=True)
                print(f"[DEBUG] agent action={action_str} | reason={reasoning}", flush=True)

            # Step environment
            next_obs, reward, done, info = env.step(ModerationAction(action_str))

            rewards.append(reward)
            steps_taken = step

            # ---- [STEP] line (single line, per spec) ----
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=last_error,
            )

            if verbose:
                print(
                    f"[DEBUG] correct={info.get('correct_action')} "
                    f"grader_score={info.get('grader_score')} "
                    f"reason={info.get('reason')}",
                    flush=True,
                )

            obs = next_obs

            if done:
                break

        # Compute final normalised score: (mean_reward + 1.2) / 2.4 clamped to [0, 1]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, (mean_reward + 1.2) / 2.4))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        logger.error("Unexpected error in run_task: %s", exc)
        # Ensure [END] is always emitted even on crash
        score = 0.0
        success = False

    finally:
        # ---- [END] line — always emitted ----
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ToxiClean AI — OpenEnv Baseline Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=VALID_TASKS + ["all"],
        default="all",
        help="Task to run. Use 'all' to run all three tasks sequentially (default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step observation and reasoning details.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the MODEL_NAME environment variable.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        dest="api_base",
        help="Override the API_BASE_URL environment variable.",
    )
    args = parser.parse_args()

    # Allow CLI overrides (update module-level constants)
    global API_BASE_URL, MODEL_NAME, HF_TOKEN
    if args.api_base:
        os.environ["API_BASE_URL"] = args.api_base
        API_BASE_URL = args.api_base
    if args.model:
        os.environ["MODEL_NAME"] = args.model
        MODEL_NAME = args.model

    print(f"=== ToxiClean AI — OpenEnv Baseline ===", flush=True)
    print(f"Model     : {MODEL_NAME}", flush=True)
    print(f"API Base  : {API_BASE_URL}", flush=True)
    print(f"Benchmark : {BENCHMARK}", flush=True)

    try:
        client = _build_client()
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    tasks_to_run = VALID_TASKS if args.task == "all" else [args.task]
    all_scores: dict[str, float] = {}

    for task in tasks_to_run:
        score = run_task(task, client, MODEL_NAME, verbose=args.verbose)
        all_scores[task] = score

    if len(tasks_to_run) > 1:
        overall = sum(all_scores.values()) / len(all_scores)
        print(f"\n=== OVERALL SCORE ===", flush=True)
        for task, score in all_scores.items():
            print(f"  {task}: {score:.4f}", flush=True)
        print(f"  AVERAGE: {overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
