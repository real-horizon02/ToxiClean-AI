"""
app.py

Hugging Face Spaces entry point for ToxiClean AI.

Exposes a Gradio interface so judges can interact with the RL environment
directly in the browser — no terminal required.

Features:
    • Run a single moderation decision and see the reward + explanation
    • Run a full task episode and watch the score accumulate
    • Task selector: Spam / Toxicity / Contextual
    • Colour-coded reward feedback

Run locally:
    python app.py

Deploy:
    Push to a HF Space with SDK=gradio in README.md YAML header.
"""

from __future__ import annotations

import os
import json
from typing import Generator

import gradio as gr

from core.environment import ToxiCleanEnv
from core.models import ModerationAction, Observation

# ── Theme ─────────────────────────────────────────────────────────────────
_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
)

# ── Task config ────────────────────────────────────────────────────────────
_TASK_OPTIONS = {
    "🗑️  Spam Detection (Easy)": "spam_detection",
    "☣️  Toxicity Classification (Medium)": "toxicity_classification",
    "🧩  Contextual Moderation (Hard)": "contextual_moderation",
}

# ── Singleton env state (Gradio is stateless per session by default) ───────
_ENVS: dict[str, ToxiCleanEnv] = {}


def _get_env(task_label: str) -> ToxiCleanEnv:
    task_name = _TASK_OPTIONS[task_label]
    if task_name not in _ENVS:
        _ENVS[task_name] = ToxiCleanEnv(task_name=task_name)
    return _ENVS[task_name]


def start_episode(task_label: str):
    """Reset the environment and return the first observation."""
    task_name = _TASK_OPTIONS[task_label]
    _ENVS[task_name] = ToxiCleanEnv(task_name=task_name)
    env = _ENVS[task_name]
    obs: Observation = env.reset()
    state = env.state()

    obs_text = _format_observation(obs)
    status = f"**Episode started** — Task: `{task_name}` | Steps: 0 / {state.total_steps}"
    return obs_text, status, "", "0.00", "0.00"


def take_action(task_label: str, action_str: str):
    """Apply the selected action and return updated observation + metrics."""
    task_name = _TASK_OPTIONS[task_label]
    if task_name not in _ENVS:
        return (
            "⚠️ No active episode. Click **Start Episode** first.",
            "", "", "0.00", "0.00",
        )

    env = _ENVS[task_name]
    if env.state().done:
        return (
            "✅ Episode complete. Click **Start Episode** to run again.",
            "**Episode done.**", "", "0.00", "0.00",
        )

    try:
        action = ModerationAction(action_str.upper())
    except ValueError:
        return ("❌ Invalid action.", "", "", "0.00", "0.00")

    next_obs, reward, done, info = env.step(action)
    state = env.state()

    # Format reward with colour hint
    reward_str = f"{reward:+.2f}"
    colour = "🟢" if reward > 0 else ("🟡" if reward == 0 else "🔴")

    info_lines = [
        f"**Correct action:** `{info.get('correct_action', '?')}`",
        f"**Your action:** `{info.get('agent_action', '?')}`",
        f"**Grader score:** `{info.get('grader_score', '?')}`",
        f"**Reason:** {info.get('reason', '')}",
    ]
    if "context_tip" in info:
        info_lines.append(f"💡 *{info['context_tip']}*")

    explanation = "\n\n".join(info_lines)

    if done:
        summary = info.get("episode_summary", {})
        final_score = summary.get("final_score", 0.0)
        obs_display = f"✅ **Episode complete!**\n\nFinal score: `{final_score:.4f}`"
        status = (
            f"**Done** — Cumulative reward: `{state.cumulative_reward:.2f}` "
            f"| Final score: `{final_score:.4f}`"
        )
        return obs_display, status, explanation, reward_str, f"{final_score:.4f}"

    obs_display = _format_observation(next_obs)
    status = (
        f"Step {state.step_index} / {state.total_steps} | "
        f"Cumulative reward: `{state.cumulative_reward:.2f}` {colour}"
    )
    norm_score = max(0.0, min(1.0, (state.cumulative_reward / state.step_index + 1.2) / 2.4)) if state.step_index > 0 else 0.0
    return obs_display, status, explanation, reward_str, f"{norm_score:.4f}"


def _format_observation(obs: Observation) -> str:
    meta = obs.metadata
    return (
        f"### Content to Moderate\n\n"
        f"> {obs.content}\n\n"
        f"---\n"
        f"**Platform:** `{meta.platform}` &nbsp;|&nbsp; "
        f"**Language:** `{meta.language}` &nbsp;|&nbsp; "
        f"**User history:** `{meta.user_history}`\n\n"
        f"*Step {obs.step_index + 1} — Task: `{obs.task_name}`*"
    )


# ── Build UI ──────────────────────────────────────────────────────────────
with gr.Blocks(theme=_THEME, title="ToxiClean AI — OpenEnv Demo") as demo:

    gr.Markdown(
        """
# 🧹 ToxiClean AI
### OpenEnv RL Environment — Content Moderation

An AI agent learns to moderate online content across three tasks of increasing difficulty.
Select a task, start an episode, then pick a moderation action for each piece of content.
        """
    )

    with gr.Row():
        task_dd = gr.Dropdown(
            choices=list(_TASK_OPTIONS.keys()),
            value=list(_TASK_OPTIONS.keys())[0],
            label="Task",
            scale=3,
        )
        start_btn = gr.Button("▶ Start Episode", variant="primary", scale=1)

    obs_box = gr.Markdown(
        value="*Click **Start Episode** to begin.*",
        label="Current Observation",
    )

    status_box = gr.Markdown(value="", label="Status")

    with gr.Row():
        action_radio = gr.Radio(
            choices=["ALLOW", "FLAG", "DELETE", "ESCALATE"],
            value="FLAG",
            label="Your Moderation Action",
            scale=3,
        )
        step_btn = gr.Button("⚡ Take Action", variant="secondary", scale=1)

    with gr.Row():
        reward_box = gr.Textbox(label="Last Reward", interactive=False, scale=1)
        score_box = gr.Textbox(label="Running Score (0–1)", interactive=False, scale=1)

    explanation_box = gr.Markdown(value="", label="Explanation")

    gr.Markdown(
        """
---
### Action Space

| Action | Meaning |
|--------|---------|
| `ALLOW` | Safe content — no action needed |
| `FLAG` | Suspicious — queue for human review |
| `DELETE` | Harmful — remove immediately |
| `ESCALATE` | Critical threat — safety team needed |

### Reward Design

| Outcome | Reward |
|---------|--------|
| Correct | `+1.0` |
| Partial | `+0.5` |
| False positive | `−0.7` |
| Missed harm | `−1.0` |
| Over-moderation | `−0.5` |
        """
    )

    # Wire up events
    start_btn.click(
        fn=start_episode,
        inputs=[task_dd],
        outputs=[obs_box, status_box, explanation_box, reward_box, score_box],
    )
    step_btn.click(
        fn=take_action,
        inputs=[task_dd, action_radio],
        outputs=[obs_box, status_box, explanation_box, reward_box, score_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
