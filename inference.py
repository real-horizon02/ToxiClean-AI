"""
inference.py

ToxiClean AI — Baseline inference script for the OpenEnv hackathon evaluation.

This script runs an LLM agent through all three tasks using an OpenAI-compatible
API.  It strictly follows the log format required by the evaluator:

    [START]
    task: <task_name>

    [STEP]
    action: <ACTION>
    reward: <float>

    [END]
    final_score: <float>

Environment variables (required)
---------------------------------
    OPENAI_API_KEY   : API key for the LLM provider
    API_BASE_URL     : Base URL  (default: https://api.openai.com/v1)
    MODEL_NAME       : Model to use (default: gpt-4o-mini)

Usage
-----
    python inference.py
    python inference.py --task spam_detection
    python inference.py --task toxicity_classification --verbose
    python inference.py --task contextual_moderation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

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
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

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
# LLM client helpers
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    """Construct an OpenAI-compatible client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE).rstrip("/")
    return OpenAI(api_key=api_key, base_url=base_url)


def _observation_to_user_message(obs: Observation) -> str:
    """
    Convert an Observation into a user-facing prompt string for the LLM.
    """
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
            logger.warning("Attempt %d: parse error — %s", attempt, exc)
            if attempt == max_retries:
                logger.error("All retries exhausted. Defaulting to ALLOW.")
                return {"action": "ALLOW", "reasoning": "Parse failure — default safe."}
            time.sleep(0.5 * attempt)  # brief back-off

        except Exception as exc:  # network/API errors
            logger.warning("Attempt %d: API error — %s", attempt, exc)
            if attempt == max_retries:
                logger.error("All retries exhausted. Defaulting to ALLOW.")
                return {"action": "ALLOW", "reasoning": "API failure — default safe."}
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
        Final normalised score for the episode.
    """
    env = ToxiCleanEnv(task_name=task_name)
    obs = env.reset()

    # ---- [START] log ----
    print(f"\n[START]")
    print(f"task: {task_name}")

    episode_rewards: list[float] = []

    while True:
        # Build the LLM prompt from the current observation
        user_message = _observation_to_user_message(obs)

        # Get action from agent
        llm_response = _call_llm(client, model, user_message)
        action_str = llm_response.get("action", "ALLOW").upper().strip()
        reasoning = llm_response.get("reasoning", "")

        if verbose:
            print(f"\n  [obs] {obs.content[:80]}...")
            print(f"  [agent] action={action_str} | reason={reasoning}")

        # Step environment
        next_obs, reward, done, info = env.step(ModerationAction(action_str))

        # ---- [STEP] log ----
        print(f"\n[STEP]")
        print(f"action: {action_str}")
        print(f"reward: {reward}")

        if verbose:
            print(f"  correct: {info.get('correct_action')} | score: {info.get('grader_score')}")
            print(f"  reason: {info.get('reason')}")

        episode_rewards.append(reward)

        if done:
            break

        obs = next_obs

    # Compute final score: mean reward, normalised to [0, 1]
    # Raw rewards range from -1.0 to +1.0 (plus reputation modifier up to ±0.2)
    # We normalise: final_score = (mean_reward + 1.2) / 2.4 clamped to [0, 1]
    mean_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    final_score = max(0.0, min(1.0, (mean_reward + 1.2) / 2.4))

    # ---- [END] log ----
    print(f"\n[END]")
    print(f"final_score: {round(final_score, 4)}")

    return final_score


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

    # Allow CLI overrides
    if args.api_base:
        os.environ["API_BASE_URL"] = args.api_base
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    model = os.getenv("MODEL_NAME", DEFAULT_MODEL)

    print(f"=== ToxiClean AI — OpenEnv Baseline ===")
    print(f"Model     : {model}")
    print(f"API Base  : {os.getenv('API_BASE_URL', DEFAULT_API_BASE)}")

    try:
        client = _build_client()
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    tasks_to_run = VALID_TASKS if args.task == "all" else [args.task]
    all_scores: dict[str, float] = {}

    for task in tasks_to_run:
        score = run_task(task, client, model, verbose=args.verbose)
        all_scores[task] = score

    if len(tasks_to_run) > 1:
        overall = sum(all_scores.values()) / len(all_scores)
        print(f"\n=== OVERALL SCORE ===")
        for task, score in all_scores.items():
            print(f"  {task}: {score:.4f}")
        print(f"  AVERAGE: {overall:.4f}")


if __name__ == "__main__":
    main()
