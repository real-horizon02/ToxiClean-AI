"""
env/models.py

Typed Pydantic models for the ToxiClean AI OpenEnv interface.
These define the observation, action, and reward contracts used throughout the environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class ModerationAction(str, Enum):
    """
    The four moderation actions an agent can take.
    Ordered roughly by severity so numeric comparisons are meaningful.
    """
    ALLOW = "ALLOW"          # Content is safe — no action needed
    FLAG = "FLAG"            # Suspicious — queue for human review
    DELETE = "DELETE"        # Clearly harmful — remove immediately
    ESCALATE = "ESCALATE"   # Critical threat — escalate to safety team


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------

class ContentMetadata(BaseModel):
    """
    Metadata attached to every piece of content under review.

    Fields
    ------
    user_history : str
        Short description of the submitting user's moderation history.
        Examples: "clean", "1 prior warning", "repeat offender".
    platform : str
        The platform or product surface where the content appeared.
        Examples: "comments", "direct_messages", "public_feed".
    language : str
        BCP-47-ish language tag.
        Examples: "en", "hi", "hinglish".
    """
    user_history: str = Field(
        default="clean",
        description="Submitting user's prior moderation record.",
        max_length=256,
    )
    platform: str = Field(
        default="comments",
        description="Platform surface where content appeared.",
        max_length=128,
    )
    language: str = Field(
        default="en",
        description="Language / locale of the content.",
        max_length=32,
    )


class Observation(BaseModel):
    """
    A single observation returned by reset() or step().

    Fields
    ------
    content : str
        The raw text content to be moderated.
    metadata : ContentMetadata
        Contextual signals to aid decision-making.
    step_index : int
        Zero-based index of the current step within an episode.
    task_name : str
        Which task is currently active (spam_detection | toxicity_classification |
        contextual_moderation).
    """
    content: str = Field(
        description="Raw text content submitted for moderation.",
        max_length=4096,
    )
    metadata: ContentMetadata = Field(
        default_factory=ContentMetadata,
        description="Contextual information about the content.",
    )
    step_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based step index within the current episode.",
    )
    task_name: str = Field(
        default="spam_detection",
        description="Active task identifier.",
    )


# ---------------------------------------------------------------------------
# Action model (wrapper so the agent always sends a typed payload)
# ---------------------------------------------------------------------------

class AgentAction(BaseModel):
    """
    Wraps the enum action with optional chain-of-thought reasoning.

    Fields
    ------
    action : ModerationAction
        The chosen moderation action.
    reasoning : str, optional
        Short natural-language explanation (used for explainability / logging).
    """
    action: ModerationAction
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional chain-of-thought reasoning for the action.",
        max_length=1024,
    )


# ---------------------------------------------------------------------------
# Reward & Step result
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Structured breakdown of how a reward was calculated.

    Fields
    ------
    base_score : float
        Core correctness score (−1.0 … +1.0).
    reputation_modifier : float
        Adjustment based on the user's moderation history.
    total : float
        Final reward = base_score + reputation_modifier.
    """
    base_score: float = Field(description="Core correctness component.")
    reputation_modifier: float = Field(
        default=0.0,
        description="Reputation-based modifier.",
    )
    total: float = Field(description="Final aggregated reward.")


class StepResult(BaseModel):
    """
    The full result returned by environment.step().

    Mirrors the classic (obs, reward, done, info) tuple but fully typed.
    """
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment State (returned by state())
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """
    Complete snapshot of the environment at any point in time.
    Useful for serialisation, debugging, and reproducibility.
    """
    task_name: str
    step_index: int
    total_steps: int
    cumulative_reward: float
    done: bool
    current_observation: Optional[Observation] = None
    episode_actions: list[str] = Field(default_factory=list)
    episode_rewards: list[float] = Field(default_factory=list)
