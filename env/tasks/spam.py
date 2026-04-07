"""
env/tasks/spam.py

Task 1 — Spam Detection (Easy)
==============================

Dataset of 12 samples (6 spam, 6 clean) covering English and Hinglish spam
patterns.  The grader is fully deterministic: the correct action is looked up
from the ground-truth label and compared against the agent's choice.

Scoring
-------
    correct      → 1.0
    partially    → 0.5  (FLAG spam, or FLAG something minor-but-suspicious)
    incorrect    → 0.0
"""

from __future__ import annotations

from typing import Any

from env.models import ContentMetadata, ModerationAction, Observation

# ---------------------------------------------------------------------------
# Internal dataset
# ---------------------------------------------------------------------------

_SAMPLES: list[dict[str, Any]] = [
    # ---- English spam ----
    {
        "content": "Buy now!!! Limited offer!!! Click here → http://bit.ly/XrT9",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Classic spam with aggressive promotion and suspicious link.",
    },
    {
        "content": "Congratulations! You've WON a $1,000 gift card. Claim NOW: scam.xyz",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="public_feed", language="en"),
        "reason": "Prize-bait phishing attempt.",
    },
    {
        "content": "EARN $500/DAY working from home!!! No experience needed!!!",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="repeat offender", platform="comments", language="en"),
        "reason": "Work-from-home income scam.",
    },
    {
        "content": "FREE iPhone 15 giveaway! Just follow and retweet! Ends TONIGHT!",
        "label": ModerationAction.FLAG,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "Suspicious giveaway — warrants human review.",
    },
    {
        "content": "Visit my OnlyFans for exclusive content 🔥 Link in bio!",
        "label": ModerationAction.FLAG,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="comments", language="en"),
        "reason": "Promotional self-spam; platform-policy dependent.",
    },
    {
        "content": "Diet pills that REALLY WORK!! Doctors HATE this one trick!",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="repeat offender", platform="comments", language="en"),
        "reason": "Health misinformation / clickbait spam.",
    },
    # ---- Hinglish spam ----
    {
        "content": "Bhai FREE mein paisa kamao! Sirf ek click karo → earn.biz/ref",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="hinglish"),
        "reason": "Hinglish earn-money spam with referral link.",
    },
    {
        "content": "Isko share karo aur Rs 500 jeetne ka mauka pao! Abhi register karo!",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="public_feed", language="hinglish"),
        "reason": "Pyramid-style share-to-win spam.",
    },
    # ---- Safe / Clean content ----
    {
        "content": "Just finished reading The Pragmatic Programmer — highly recommend it!",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "Genuine book recommendation.",
    },
    {
        "content": "Anyone else excited about the new season of Succession?",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Normal conversational content.",
    },
    {
        "content": "Kal ka match bahut achha tha! India ne zabardast khela 🏏",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="hinglish"),
        "reason": "Positive sports commentary in Hinglish.",
    },
    {
        "content": "Great tutorial on Python decorators — thanks for sharing!",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Constructive comment on educational content.",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_NAME = "spam_detection"
TOTAL_STEPS = len(_SAMPLES)


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
    Deterministic grader for spam detection.

    Returns
    -------
    score : float
        0.0, 0.5, or 1.0
    info : dict
        Explainability info with keys: reason, correct_action, agent_action.
    """
    sample = _SAMPLES[index]
    correct: ModerationAction = sample["label"]

    # Exact match → full credit
    if action == correct:
        score = 1.0
    # FLAG instead of DELETE for spam → partial credit (caught it, didn't delete)
    elif correct in (ModerationAction.DELETE, ModerationAction.FLAG) and action == ModerationAction.FLAG:
        score = 0.5
    elif correct == ModerationAction.ALLOW and action == ModerationAction.FLAG:
        # Over-cautious but not catastrophic
        score = 0.5
    else:
        score = 0.0

    info = {
        "task": TASK_NAME,
        "reason": sample["reason"],
        "correct_action": correct.value,
        "agent_action": action.value,
        "grader_score": score,
    }
    return score, info
