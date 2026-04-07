"""
tests/test_environment.py

Smoke tests for the ToxiClean AI OpenEnv environment.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest

from env.environment import ToxiCleanEnv
from env.models import EnvironmentState, ModerationAction, Observation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["spam_detection", "toxicity_classification", "contextual_moderation"])
def env(request):
    """Parameterised fixture — runs each test for all three tasks."""
    return ToxiCleanEnv(task_name=request.param)


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestOpenEnvInterface:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.content
        assert obs.step_index == 0

    def test_step_returns_correct_tuple(self, env):
        env.reset()
        obs, reward, done, info = env.step(ModerationAction.ALLOW)
        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_environment_state(self, env):
        env.reset()
        state = env.state()
        assert isinstance(state, EnvironmentState)
        assert state.step_index == 0

    def test_done_after_all_steps(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(ModerationAction.ALLOW)
            steps += 1
        # All tasks have exactly 12 samples
        assert steps == 12
        assert done is True

    def test_step_after_done_raises(self, env):
        env.reset()
        for _ in range(12):
            env.step(ModerationAction.ALLOW)
        with pytest.raises(RuntimeError, match="Episode is already done"):
            env.step(ModerationAction.ALLOW)

    def test_reset_restarts_episode(self, env):
        env.reset()
        for _ in range(12):
            env.step(ModerationAction.ALLOW)
        # Should not raise after reset
        obs = env.reset()
        assert obs.step_index == 0
        state = env.state()
        assert state.cumulative_reward == 0.0


# ---------------------------------------------------------------------------
# Action handling
# ---------------------------------------------------------------------------

class TestActionHandling:
    def test_string_action_accepted(self, env):
        env.reset()
        # Should accept string "DELETE" as well as the enum
        obs, reward, done, info = env.step("DELETE")
        assert isinstance(reward, float)

    def test_invalid_action_raises(self, env):
        env.reset()
        with pytest.raises(ValueError):
            env.step("NUKE")  # not a valid action

    @pytest.mark.parametrize("action", list(ModerationAction))
    def test_all_actions_accepted(self, action):
        env = ToxiCleanEnv(task_name="spam_detection")
        env.reset()
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float)


# ---------------------------------------------------------------------------
# Reward bounds
# ---------------------------------------------------------------------------

class TestRewardBounds:
    def test_reward_in_expected_range(self, env):
        env.reset()
        done = False
        while not done:
            _, reward, done, _ = env.step(ModerationAction.ALLOW)
            # Rewards should stay within [-2, 2] (including reputation modifier)
            assert -2.0 <= reward <= 2.0

    def test_correct_action_gives_positive_reward(self):
        """
        For spam task sample 0, DELETE is the correct action.
        Verify we get a positive reward.
        """
        env = ToxiCleanEnv(task_name="spam_detection")
        env.reset()
        # Sample 0 is spam — correct action is DELETE
        _, reward, _, info = env.step(ModerationAction.DELETE)
        assert reward > 0, f"Expected positive reward, got {reward}"

    def test_wrong_action_on_harmful_gives_negative_reward(self):
        """
        Allowing spam should give a negative reward.
        """
        env = ToxiCleanEnv(task_name="spam_detection")
        env.reset()
        _, reward, _, info = env.step(ModerationAction.ALLOW)
        assert reward < 0, f"Expected negative reward for missed spam, got {reward}"


# ---------------------------------------------------------------------------
# Info dict completeness
# ---------------------------------------------------------------------------

class TestInfoDict:
    def test_info_has_reason(self, env):
        env.reset()
        _, _, _, info = env.step(ModerationAction.FLAG)
        assert "reason" in info

    def test_info_has_correct_action(self, env):
        env.reset()
        _, _, _, info = env.step(ModerationAction.FLAG)
        assert "correct_action" in info
        assert info["correct_action"] in [a.value for a in ModerationAction]

    def test_episode_summary_on_done(self, env):
        env.reset()
        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(ModerationAction.DELETE)
        assert "episode_summary" in info
        summary = info["episode_summary"]
        assert "final_score" in summary
        assert "cumulative_reward" in summary
        assert "total_steps" in summary


# ---------------------------------------------------------------------------
# Invalid task name
# ---------------------------------------------------------------------------

class TestTaskValidation:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            ToxiCleanEnv(task_name="flying_spaghetti_monster")
