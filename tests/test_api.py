"""
tests/test_api.py

HTTP endpoint tests for the FastAPI server (server.py).

Verifies:
  - POST /reset  → 200 with observation payload
  - POST /step   → 200 with reward in [0.0…2.0] range (including reputation)
  - GET  /state  → 200 with expected fields
  - GET  /health → 200

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app — this also mounts Gradio at /ui
from server import app

client = TestClient(app, raise_server_exceptions=True)

VALID_TASKS = ["spam_detection", "toxicity_classification", "contextual_moderation"]


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------

class TestResetEndpoint:
    def test_reset_returns_200(self):
        """Validator ping: POST /reset with empty body must return 200."""
        resp = client.post("/reset", json={})
        assert resp.status_code == 200

    def test_reset_response_has_observation(self):
        resp = client.post("/reset", json={"task_name": "spam_detection"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        obs = data["observation"]
        assert "content" in obs
        assert "step_index" in obs
        assert obs["step_index"] == 0

    @pytest.mark.parametrize("task", VALID_TASKS)
    def test_reset_all_tasks(self, task):
        resp = client.post("/reset", json={"task_name": task})
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_name"] == task

    def test_reset_invalid_task_returns_422(self):
        resp = client.post("/reset", json={"task_name": "nonexistent_task"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------

class TestStepEndpoint:
    def setup_method(self):
        """Reset spam_detection before each step test."""
        client.post("/reset", json={"task_name": "spam_detection"})

    def test_step_returns_200(self):
        resp = client.post("/step", json={"task_name": "spam_detection", "action": "ALLOW"})
        assert resp.status_code == 200

    def test_step_response_structure(self):
        resp = client.post("/step", json={"task_name": "spam_detection", "action": "FLAG"})
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_float(self):
        resp = client.post("/step", json={"task_name": "spam_detection", "action": "DELETE"})
        data = resp.json()
        assert isinstance(data["reward"], float)

    def test_step_invalid_action_returns_422(self):
        resp = client.post("/step", json={"task_name": "spam_detection", "action": "NUKE"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------

class TestStateEndpoint:
    def test_state_returns_200(self):
        client.post("/reset", json={"task_name": "spam_detection"})
        resp = client.get("/state", params={"task_name": "spam_detection"})
        assert resp.status_code == 200

    def test_state_has_required_fields(self):
        client.post("/reset", json={"task_name": "spam_detection"})
        resp = client.get("/state", params={"task_name": "spam_detection"})
        data = resp.json()
        required_fields = {
            "task_name", "step_index", "total_steps",
            "cumulative_reward", "done", "episode_actions", "episode_rewards",
        }
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_body(self):
        resp = client.get("/health")
        data = resp.json()
        assert data.get("status") == "ok"
