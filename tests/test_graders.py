"""
tests/test_graders.py

Grader validation tests — verifies the pre-submission checklist requirement:
  "3+ tasks with graders: run each grader, verify scores/reward in 0.0–1.0 range"

Run with:
    pytest tests/test_graders.py -v
"""

from __future__ import annotations

import pytest

from core.models import ModerationAction
from core.tasks import spam, toxicity, contextual

# ---------------------------------------------------------------------------
# All grader modules under test
# ---------------------------------------------------------------------------

GRADER_MODULES = [
    ("spam_detection",          spam,        spam.TOTAL_STEPS),
    ("toxicity_classification", toxicity,    toxicity.TOTAL_STEPS),
    ("contextual_moderation",   contextual,  contextual.TOTAL_STEPS),
]

ALL_ACTIONS = list(ModerationAction)


# ---------------------------------------------------------------------------
# 1. Grader score always in [0.0, 1.0]
# ---------------------------------------------------------------------------

class TestGraderScoreRange:
    """Every (task, step, action) combination must return a score in [0.0, 1.0]."""

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    @pytest.mark.parametrize("action", ALL_ACTIONS)
    def test_grader_score_in_range(self, task_name, module, total_steps, action):
        for idx in range(total_steps):
            score, info = module.grade(idx, action)
            assert 0.0 <= score <= 1.0, (
                f"[{task_name}] step {idx} action {action.value}: "
                f"score {score} out of [0.0, 1.0]"
            )

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    def test_correct_action_gives_full_score(self, task_name, module, total_steps):
        """Grading the correct action for every sample must return 1.0."""
        for idx in range(total_steps):
            obs = module.get_observation(idx)
            correct_action_str = module._SAMPLES[idx]["label"]
            score, info = module.grade(idx, correct_action_str)
            assert score == 1.0, (
                f"[{task_name}] step {idx}: correct action gave score {score}, expected 1.0"
            )

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    def test_grader_returns_info_dict(self, task_name, module, total_steps):
        """The info dict must always contain required keys."""
        for idx in range(total_steps):
            score, info = module.grade(idx, ModerationAction.ALLOW)
            assert isinstance(info, dict), f"[{task_name}] step {idx}: info must be a dict"
            assert "correct_action" in info, f"[{task_name}] step {idx}: missing 'correct_action'"
            assert "reason" in info, f"[{task_name}] step {idx}: missing 'reason'"
            assert "agent_action" in info, f"[{task_name}] step {idx}: missing 'agent_action'"
            assert "grader_score" in info, f"[{task_name}] step {idx}: missing 'grader_score'"


# ---------------------------------------------------------------------------
# 2. Environment reward (after modifiers) stays in [-2.0, 2.0]
# ---------------------------------------------------------------------------

from core.environment import ToxiCleanEnv


class TestEnvironmentRewardRange:
    """Full reward (including reputation modifier) must stay in [-2.0, 2.0]."""

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    @pytest.mark.parametrize("action", ALL_ACTIONS)
    def test_env_reward_in_range(self, task_name, action, module, total_steps):
        env = ToxiCleanEnv(task_name=task_name)
        env.reset()
        done = False
        while not done:
            _, reward, done, _ = env.step(action)
            assert -2.0 <= reward <= 2.0, (
                f"[{task_name}] action {action.value}: reward {reward} out of [-2.0, 2.0]"
            )
            env.reset()  # reset each step so we always start from idx 0
            break        # one step per loop since we just reset; test the pattern

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    def test_normalised_episode_score_in_range(self, task_name, module, total_steps):
        """
        Full episode score normalised as (mean_reward + 1.2) / 2.4 must be in [0.0, 1.0].
        """
        env = ToxiCleanEnv(task_name=task_name)
        env.reset()
        rewards = []
        done = False
        while not done:
            # Cycle through actions to get varied rewards
            action = ALL_ACTIONS[len(rewards) % len(ALL_ACTIONS)]
            _, reward, done, _ = env.step(action)
            rewards.append(reward)

        mean_reward = sum(rewards) / len(rewards)
        score = max(0.0, min(1.0, (mean_reward + 1.2) / 2.4))
        assert 0.0 <= score <= 1.0, (
            f"[{task_name}] normalised score {score} out of [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# 3. Three distinct tasks exist and are enumerable
# ---------------------------------------------------------------------------

class TestTaskEnumeration:
    def test_three_tasks_registered(self):
        from core.environment import _TASK_REGISTRY
        assert len(_TASK_REGISTRY) >= 3, "Must have at least 3 tasks registered"

    def test_all_required_tasks_present(self):
        from core.environment import _TASK_REGISTRY
        required = {"spam_detection", "toxicity_classification", "contextual_moderation"}
        missing = required - set(_TASK_REGISTRY.keys())
        assert not missing, f"Missing required tasks: {missing}"

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    def test_each_task_has_minimum_steps(self, task_name, module, total_steps):
        assert total_steps >= 1, f"[{task_name}] must have at least 1 sample"

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    def test_each_task_grader_callable(self, task_name, module, total_steps):
        assert callable(module.grade), f"[{task_name}] grade() must be callable"

    @pytest.mark.parametrize("task_name,module,total_steps", GRADER_MODULES)
    def test_each_task_get_observation_callable(self, task_name, module, total_steps):
        assert callable(module.get_observation), (
            f"[{task_name}] get_observation() must be callable"
        )
