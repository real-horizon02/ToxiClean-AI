"""
env/environment.py

ToxiCleanEnv — Production-ready OpenEnv RL environment for content moderation.

OpenEnv interface
-----------------
    reset()           → Observation
    step(action)      → (Observation, float, bool, dict)
    state()           → EnvironmentState

The environment cycles through three tasks in order:
    1. spam_detection        (easy)
    2. toxicity_classification (medium)
    3. contextual_moderation  (hard)

Each episode runs all samples from a single task; when the last sample is
exhausted, done=True is returned.

Reward design
-------------
    +1.0  correct moderation
    +0.5  partially correct
    -0.7  false positive (flagged/deleted safe content)
    -1.0  missed harmful content (allowed harmful)
    -0.5  over-moderation (escalated something that needed only a flag)

A small reputation modifier is also applied:
    repeat offender → reward scaled ×1.1 for correct, ×1.2 penalty for incorrect
    1 prior warning → reward scaled ×1.05 for correct, ×1.1 penalty
    clean           → no modifier
"""

from __future__ import annotations

import logging
from typing import Any

from core.models import (
    EnvironmentState,
    ModerationAction,
    Observation,
    StepResult,
)
from core.tasks import contextual, spam, toxicity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task registry — maps task name to its module
# ---------------------------------------------------------------------------
_TASK_REGISTRY = {
    spam.TASK_NAME: spam,
    toxicity.TASK_NAME: toxicity,
    contextual.TASK_NAME: contextual,
}

_TASK_ORDER = [spam.TASK_NAME, toxicity.TASK_NAME, contextual.TASK_NAME]

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
_REWARD_CORRECT = 1.0
_REWARD_PARTIAL = 0.5
_PENALTY_FALSE_POSITIVE = -0.7   # agent removed safe content
_PENALTY_MISSED_HARM = -1.0      # agent allowed harmful content
_PENALTY_OVER_MODERATION = -0.5  # agent escalated unnecessarily

# User-reputation modifiers
_REPUTATION_MODIFIER: dict[str, dict[str, float]] = {
    "repeat offender": {"correct": 0.1, "incorrect": -0.2},
    "1 prior warning": {"correct": 0.05, "incorrect": -0.1},
    "clean": {"correct": 0.0, "incorrect": 0.0},
}


class ToxiCleanEnv:
    """
    OpenEnv-compliant reinforcement learning environment for content moderation.

    Parameters
    ----------
    task_name : str, optional
        Which task to run.  Defaults to 'spam_detection'.
        Must be one of: 'spam_detection', 'toxicity_classification',
        'contextual_moderation'.
    """

    def __init__(self, task_name: str = "spam_detection") -> None:
        if task_name not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Choose from: {list(_TASK_REGISTRY.keys())}"
            )
        self._task_name = task_name
        self._task_module = _TASK_REGISTRY[task_name]
        self._step_index: int = 0
        self._total_steps: int = self._task_module.TOTAL_STEPS
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._current_obs: Observation | None = None
        self._episode_actions: list[str] = []
        self._episode_rewards: list[float] = []

    # -----------------------------------------------------------------------
    # OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment to the beginning of a new episode.

        Returns
        -------
        Observation
            The first observation of the episode.
        """
        self._step_index = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._episode_actions = []
        self._episode_rewards = []
        self._current_obs = self._task_module.get_observation(0)
        logger.info("Environment reset. Task: %s", self._task_name)
        return self._current_obs

    def step(
        self, action: ModerationAction | str
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Apply an action and advance the environment by one step.

        Parameters
        ----------
        action : ModerationAction | str
            The agent's chosen moderation action.  Accepts both the enum and
            its string value for LLM-agent convenience.

        Returns
        -------
        observation : Observation
            Next item to moderate (or the terminal observation if done).
        reward : float
            Immediate reward signal.
        done : bool
            True when all samples in the episode have been consumed.
        info : dict
            Explainability info (reason, correct action, score, etc.).
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        # Normalise action
        if isinstance(action, str):
            try:
                action = ModerationAction(action.upper())
            except ValueError:
                valid = [a.value for a in ModerationAction]
                raise ValueError(
                    f"Invalid action '{action}'. Must be one of {valid}."
                )

        # Grade the action
        grader_score, info = self._task_module.grade(self._step_index, action)

        # Convert grader score to reward
        reward = self._compute_reward(grader_score, info, action)

        # Apply reputation modifier
        reputation = (
            self._current_obs.metadata.user_history
            if self._current_obs
            else "clean"
        )
        reward = self._apply_reputation_modifier(reward, grader_score, reputation)

        # Track episode stats
        self._cumulative_reward += reward
        self._episode_actions.append(action.value)
        self._episode_rewards.append(round(reward, 4))

        # Enrich info
        info["reward"] = round(reward, 4)
        info["cumulative_reward"] = round(self._cumulative_reward, 4)
        info["step_index"] = self._step_index

        # Advance step
        self._step_index += 1
        done = self._step_index >= self._total_steps
        self._done = done

        if not done:
            next_obs = self._task_module.get_observation(self._step_index)
        else:
            # Terminal observation: repeat last with done marker
            next_obs = Observation(
                content="[EPISODE_COMPLETE]",
                step_index=self._step_index,
                task_name=self._task_name,
            )
            info["episode_summary"] = {
                "total_steps": self._total_steps,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "final_score": round(
                    self._cumulative_reward / self._total_steps, 4
                ),
                "actions": self._episode_actions,
            }
            logger.info(
                "Episode complete. Cumulative reward: %.4f | Final score: %.4f",
                self._cumulative_reward,
                self._cumulative_reward / self._total_steps,
            )

        self._current_obs = next_obs
        return next_obs, round(reward, 4), done, info

    def state(self) -> EnvironmentState:
        """
        Return a complete snapshot of the current environment state.

        Returns
        -------
        EnvironmentState
            A Pydantic model capturing all relevant state fields.
        """
        return EnvironmentState(
            task_name=self._task_name,
            step_index=self._step_index,
            total_steps=self._total_steps,
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            current_observation=self._current_obs,
            episode_actions=list(self._episode_actions),
            episode_rewards=list(self._episode_rewards),
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_reward(
        grader_score: float,
        info: dict[str, Any],
        agent_action: ModerationAction,
    ) -> float:
        """
        Map the grader score and action pair to a structured reward signal.

        The reward is richer than a simple pass/fail so the agent can learn
        *what kind* of mistake it made.
        """
        correct_action = ModerationAction(info["correct_action"])

        if grader_score == 1.0:
            return _REWARD_CORRECT

        if grader_score == 0.5:
            return _REWARD_PARTIAL

        # grader_score == 0.0 — determine the penalty type
        actually_harmful = correct_action in (
            ModerationAction.DELETE,
            ModerationAction.ESCALATE,
        )
        agent_was_permissive = agent_action == ModerationAction.ALLOW

        if actually_harmful and agent_was_permissive:
            # Worst case: let harmful content through
            return _PENALTY_MISSED_HARM

        # Check over-moderation BEFORE false-positive so a FLAG→ESCALATE
        # error gets the more specific -0.5 penalty, not the generic -0.7.
        if agent_action == ModerationAction.ESCALATE and correct_action in (
            ModerationAction.FLAG,
            ModerationAction.DELETE,
            ModerationAction.ALLOW,
        ):
            # Escalated something that didn't warrant the safety team
            return _PENALTY_OVER_MODERATION

        if not actually_harmful and agent_action in (
            ModerationAction.DELETE,
            ModerationAction.ESCALATE,
        ):
            # Incorrectly removed or over-actioned safe content
            return _PENALTY_FALSE_POSITIVE

        # Catch-all for any other 0-score case (e.g. DELETE instead of FLAG)
        return _PENALTY_FALSE_POSITIVE

    @staticmethod
    def _apply_reputation_modifier(
        reward: float, grader_score: float, user_history: str
    ) -> float:
        """
        Adjust reward based on the submitting user's reputation.

        Repeat offenders should be moderated more strictly; errors on their
        content carry a larger penalty (and correct actions a small bonus).
        """
        # Normalise to known keys
        key = user_history.lower()
        if key not in _REPUTATION_MODIFIER:
            key = "clean"

        modifiers = _REPUTATION_MODIFIER[key]
        if grader_score == 1.0:
            reward += modifiers["correct"]
        elif grader_score == 0.0:
            reward += modifiers["incorrect"]

        # Cap reward in a reasonable range
        return max(-2.0, min(2.0, reward))
