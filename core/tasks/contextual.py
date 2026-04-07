"""
env/tasks/contextual.py

Task 3 — Contextual Moderation (Hard)
======================================

The hardest task.  Surface-level keywords are deceptive — the agent must
understand *intent* and *context* to take the right action.

Examples of deceptive pairs:
    "Kill yourself"            → ESCALATE  (direct self-harm instruction)
    "This bug is killing me 😂" → ALLOW    (idiomatic frustration)
    "I want to murder this song" → ALLOW   (slang enthusiasm)
    "I'll murder you slowly"    → DELETE   (literal threat)

Dataset: 12 samples across English and Hinglish, including:
    - Idiomatic / slang false positives
    - Genuine threats disguised in polite language
    - Sarcasm that reads as positive
    - Context switches mid-thread excerpts
"""

from __future__ import annotations

from typing import Any

from core.models import ContentMetadata, ModerationAction, Observation

# ---------------------------------------------------------------------------
# Internal dataset
# ---------------------------------------------------------------------------

_SAMPLES: list[dict[str, Any]] = [
    # ---- Idiomatic / safe ----
    {
        "content": "This bug is killing me 😂 I've been debugging for 6 hours straight.",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="en"),
        "reason": "Idiomatic use of 'killing' — expresses frustration, not violence.",
    },
    {
        "content": "I want to murder this song, it's been stuck in my head all day!",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "Hyperbolic slang for an earworm — no genuine threat.",
    },
    {
        "content": "Yaar ye traffic mujhe maar hi dalega, roz 2 ghante waste hote hain.",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="comments", language="hinglish"),
        "reason": "Hinglish idiom meaning 'this traffic is going to be the death of me' — safe.",
    },
    {
        "content": "The new Coldplay album absolutely slaps. They killed it! 🔥",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "'Killed it' and 'slaps' are positive slang — no harm.",
    },
    # ---- Genuine threats / harmful ----
    {
        "content": "Kill yourself. No one would miss you.",
        "label": ModerationAction.ESCALATE,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="direct_messages", language="en"),
        "reason": "Direct self-harm instruction — critical, escalate immediately.",
    },
    {
        "content": "I know where you live. You'll pay for what you said.",
        "label": ModerationAction.ESCALATE,
        "metadata": ContentMetadata(user_history="repeat offender", platform="direct_messages", language="en"),
        "reason": "Implicit doxxing with personal threat.",
    },
    {
        "content": "Tu agar kal school aaya toh teri haddi todne ka plan ready hai.",
        "label": ModerationAction.ESCALATE,
        "metadata": ContentMetadata(user_history="1 prior warning", platform="direct_messages", language="hinglish"),
        "reason": "Hinglish physical threat — escalate.",
    },
    {
        "content": "I'll report every one of your posts until you're banned for good.",
        "label": ModerationAction.FLAG,
        "metadata": ContentMetadata(user_history="repeat offender", platform="comments", language="en"),
        "reason": "Harassment / trolling behaviour — flag for review.",
    },
    # ---- Polite wrapping of harmful content ----
    {
        "content": "With all due respect, people like you shouldn't be allowed to breed.",
        "label": ModerationAction.DELETE,
        "metadata": ContentMetadata(user_history="repeat offender", platform="comments", language="en"),
        "reason": "Hate speech disguised with polite framing.",
    },
    {
        "content": "I'm just asking questions — why is our country being replaced by immigrants?",
        "label": ModerationAction.FLAG,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "Dog-whistle rhetoric — suspicious, warrants review.",
    },
    # ---- Cross-language edge cases ----
    {
        "content": "Bhai ek kaam kar, is ladke ko block kar warna main tujhe personally handle karunga.",
        "label": ModerationAction.ESCALATE,
        "metadata": ContentMetadata(user_history="repeat offender", platform="direct_messages", language="hinglish"),
        "reason": "Hinglish: 'I'll handle you personally' is an implicit physical threat.",
    },
    {
        "content": "This is satire. The politician completely 'destroyed' the economy — chef's kiss.",
        "label": ModerationAction.ALLOW,
        "metadata": ContentMetadata(user_history="clean", platform="public_feed", language="en"),
        "reason": "Political satire with explicit satire label — not genuine threat.",
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_NAME = "contextual_moderation"
TOTAL_STEPS = len(_SAMPLES)

# For contextual moderation, partial credit is tighter: only directly adjacent
# actions get 0.5 and only when the correct action is not ESCALATE.
_ADJACENT: dict[ModerationAction, set[ModerationAction]] = {
    ModerationAction.ALLOW: {ModerationAction.FLAG},
    ModerationAction.FLAG: {ModerationAction.DELETE},
    ModerationAction.DELETE: {ModerationAction.FLAG},
    # ESCALATE must be matched exactly — no partial credit for missing a critical threat
    ModerationAction.ESCALATE: set(),
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
    Deterministic grader for contextual moderation.

    Misidentifying an ESCALATE case as anything else → 0.0 (no partial credit).
    Other adjacency misses → 0.5.
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
        "correct_action": correct.value,
        "agent_action": action.value,
        "grader_score": score,
        "context_tip": (
            "Remember: surface keywords can be misleading. "
            "Consider intent, idiom, and platform context."
        ),
    }
    return score, info
