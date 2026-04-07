"""
env/tasks/toxicity.py

Task 2 — Toxicity Classification (Medium)
==========================================

Dataset of 12 samples spanning clean content, mild insults, and severe abuse,
in English and Hinglish.  The grader distinguishes mild from severe toxicity and
awards partial credit for near-misses.

Severity levels (internal only):
    0 — clean
    1 — mild toxicity  (FLAG is correct)
    2 — severe toxicity (DELETE is correct)
    3 — critical / threat (ESCALATE is correct)
"""

from __future__ import annotations

from typing import Any

from env.models import ContentMetadata, ModerationAction, Observation

# ---------------------------------------------------------------------------
# Internal dataset
# ---------------------------------------------------------------------------

_SAMPLES: list[dict[str, Any]] = [
    # ---- Severe toxicity → DELETE ----
    {
        "content": "You're a worthless piece of garbage. Go die in a fire.",
        "label": ModerationAction.DELETE,
        "severity": 2,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="comments", language="en"),
        "reason": "Severe abusive language with death wish.",
    },
    {
        "content": "All [ethnic group] people should be banned from this country!",
        "label": ModerationAction.DELETE,
        "severity": 2,
        "metadata": ContentMetadata(user_history="repeat offender", platform="public_feed", language="en"),
        "reason": "Hate speech targeting an ethnic group.",
    },
    {
        "content": "Teri maa ki aankh, tu aur teri poori nasal trash hai.",
        "label": ModerationAction.DELETE,
        "severity": 2,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="comments", language="hinglish"),
        "reason": "Severe Hinglish abuse targeting family.",
    },
    # ---- Critical threats → ESCALATE ----
    {
        "content": "I know your address. Keep talking and you'll regret it.",
        "label": ModerationAction.ESCALATE,
        "severity": 3,
        "metadata": ContentMetadata(user_history="repeat offender", platform="direct_messages", language="en"),
        "reason": "Implicit doxxing threat — requires safety-team escalation.",
    },
    {
        "content": "Someone should hunt down the CEO and make him pay.",
        "label": ModerationAction.ESCALATE,
        "severity": 3,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "Incitement to violence against a named target.",
    },
    # ---- Mild toxicity → FLAG ----
    {
        "content": "This product is absolute crap. Total waste of money.",
        "label": ModerationAction.FLAG,
        "severity": 1,
        "metadata": ContentMetadata(user_history="clean", platform="reviews", language="en"),
        "reason": "Strong negative language — may be legitimate frustration.",
    },
    {
        "content": "You're so dumb for believing that lol",
        "label": ModerationAction.FLAG,
        "severity": 1,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Mild insult; context-dependent.",
    },
    {
        "content": "Yaar tu bahut bekar insaan hai, kuch nahi aata tujhe.",
        "label": ModerationAction.FLAG,
        "severity": 1,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="hinglish"),
        "reason": "Mild Hinglish insult calling someone useless.",
    },
    # ---- Clean / Allow ----
    {
        "content": "I respectfully disagree with your analysis of the budget proposal.",
        "label": ModerationAction.ALLOW,
        "severity": 0,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Polite civil disagreement.",
    },
    {
        "content": "This movie was terrible. Two stars out of five.",
        "label": ModerationAction.ALLOW,
        "severity": 0,
        "metadata": ContentMetadata(user_history="clean", platform="reviews", language="en"),
        "reason": "Negative opinion expressed without abuse.",
    },
    {
        "content": "Mujhe yeh game bilkul pasand nahi aayi, boring thi.",
        "label": ModerationAction.ALLOW,
        "severity": 0,
        "metadata": ContentMetadata(user_history="clean", platform="reviews", language="hinglish"),
        "reason": "Hinglish negative review — no abuse.",
    },
    {
        "content": "Constructive criticism: the UX could use a lot of improvement.",
        "label": ModerationAction.ALLOW,
        "severity": 0,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Helpful, non-abusive feedback.",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_NAME = "toxicity_classification"
TOTAL_STEPS = len(_SAMPLES)

# Adjacency map: actions one step away from correct get partial credit.
_ADJACENT: dict[ModerationAction, set[ModerationAction]] = {
    ModerationAction.ALLOW: {ModerationAction.FLAG},
    ModerationAction.FLAG: {ModerationAction.ALLOW, ModerationAction.DELETE},
    ModerationAction.DELETE: {ModerationAction.FLAG, ModerationAction.ESCALATE},
    ModerationAction.ESCALATE: {ModerationAction.DELETE},
}


def get_observation(index: int) -> Observation:
    """Return the observation at `index`."""
    sample = _SAMPLES[index]
    return Observation(
        content=sample["content"],
        metadata=sample["metadata"],
        step_index=index,
        task_name=TASK_NAME,
    )


def grade(index: int, action: ModerationAction) -> tuple[float, dict[str, Any]]:
    """
    Deterministic grader for toxicity classification.

    Partial credit is awarded when the agent picks an adjacent severity level.
    """
    sample = _SAMPLES[index]
    correct: ModerationAction = sample["label"]

    if action == correct:
        score = 1.0
    elif action in _ADJACENT.get(correct, set()):
        score = 0.5
    else:
        score = 0.0

    info = {
        "task": TASK_NAME,
        "reason": sample["reason"],
        "severity": sample["severity"],
        "correct_action": correct.value,
        "agent_action": action.value,
        "grader_score": score,
    }
    return score, info
