"""
server.py

ToxiClean AI — FastAPI REST server for the HF Space.

Exposes the OpenEnv HTTP interface so the submission validator can ping
    POST /reset  → 200 { observation, task_name, step_index, ... }
    POST /step   → 200 { observation, reward, done, info }
    GET  /state  → 200 { task_name, step_index, ... }
    GET  /health → 200 { status: "ok" }

Also mounts the Gradio web UI at /ui (root path redirects there).

Run locally:
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 7860

Deploy:
    Push to a HF Space with SDK=docker. CMD in Dockerfile runs this.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from core.environment import ToxiCleanEnv
from core.models import EnvironmentState, ModerationAction, Observation

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ToxiClean AI — OpenEnv API",
    description=(
        "REST interface for the ToxiClean AI reinforcement-learning environment. "
        "Implements the OpenEnv spec: /reset, /step, /state."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (keyed by task_name for simplicity)
# ---------------------------------------------------------------------------
_ENVS: dict[str, ToxiCleanEnv] = {}
_VALID_TASKS = ["spam_detection", "toxicity_classification", "contextual_moderation"]


def _get_or_create_env(task_name: str) -> ToxiCleanEnv:
    if task_name not in _VALID_TASKS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{task_name}'. Must be one of {_VALID_TASKS}.",
        )
    if task_name not in _ENVS:
        _ENVS[task_name] = ToxiCleanEnv(task_name=task_name)
    return _ENVS[task_name]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = Field(
        default="spam_detection",
        description="Which task to reset. One of: spam_detection, toxicity_classification, contextual_moderation.",
    )


class StepRequest(BaseModel):
    task_name: str = Field(
        default="spam_detection",
        description="Which task environment to step.",
    )
    action: str = Field(
        description="Moderation action: ALLOW | FLAG | DELETE | ESCALATE",
    )


class ObservationResponse(BaseModel):
    content: str
    metadata: dict[str, Any]
    step_index: int
    task_name: str


class ResetResponse(BaseModel):
    observation: ObservationResponse
    task_name: str
    total_steps: int
    message: str = "Episode reset successfully."


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float
    done: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    task_name: str
    step_index: int
    total_steps: int
    cumulative_reward: float
    done: bool
    episode_actions: list[str]
    episode_rewards: list[float]


def _obs_to_response(obs: Observation) -> ObservationResponse:
    return ObservationResponse(
        content=obs.content,
        metadata=obs.metadata.model_dump(),
        step_index=obs.step_index,
        task_name=obs.task_name,
    )


# ---------------------------------------------------------------------------
# OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse, summary="Reset environment episode")
async def reset(body: ResetRequest = ResetRequest()):
    """
    Reset the environment for the given task and return the first observation.

    The submission validator calls:
        POST /reset   (body may be empty `{}`)
    and checks for HTTP 200. This endpoint handles both empty and populated bodies.
    """
    task_name = body.task_name if body.task_name else "spam_detection"
    env = _get_or_create_env(task_name)
    obs: Observation = env.reset()
    state: EnvironmentState = env.state()

    return ResetResponse(
        observation=_obs_to_response(obs),
        task_name=task_name,
        total_steps=state.total_steps,
    )


@app.post("/step", response_model=StepResponse, summary="Take a moderation action")
async def step(body: StepRequest):
    """
    Apply a moderation action and advance the environment by one step.
    """
    env = _get_or_create_env(body.task_name)

    if env.state().done:
        raise HTTPException(
            status_code=409,
            detail="Episode is already done. Call /reset to start a new episode.",
        )

    try:
        action = ModerationAction(body.action.upper().strip())
    except ValueError:
        valid = [a.value for a in ModerationAction]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{body.action}'. Must be one of {valid}.",
        )

    next_obs, reward, done, info = env.step(action)

    return StepResponse(
        observation=_obs_to_response(next_obs),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse, summary="Get current environment state")
async def state(task_name: str = "spam_detection"):
    """
    Return a complete snapshot of the environment's current state.
    """
    env = _get_or_create_env(task_name)
    s = env.state()

    return StateResponse(
        task_name=s.task_name,
        step_index=s.step_index,
        total_steps=s.total_steps,
        cumulative_reward=s.cumulative_reward,
        done=s.done,
        episode_actions=s.episode_actions,
        episode_rewards=s.episode_rewards,
    )


@app.get("/health", summary="Health check")
async def health():
    """Simple liveness probe."""
    return {"status": "ok", "service": "toxiclean-ai"}


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the Gradio UI."""
    return RedirectResponse(url="/ui")


# ---------------------------------------------------------------------------
# Gradio UI (mounted at /ui)
# ---------------------------------------------------------------------------

from app import demo as gradio_demo  # noqa: E402

app = gr.mount_gradio_app(app, gradio_demo, path="/ui")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
