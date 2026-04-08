"""
app.py

ToxiClean AI — Visual Dashboard for HuggingFace Spaces.

A rich, real-time dashboard that lets any user:
  • Run an AI moderation agent live (using their LLM via HF_TOKEN)
  • Step through tasks manually and see rewards in real-time
  • View a full episode summary with score gauges and step history
  • Browse the task library across all 3 difficulty tiers
"""

from __future__ import annotations

import os
import time
import json
from typing import Optional

import gradio as gr
from dotenv import load_dotenv

from core.environment import ToxiCleanEnv
from core.models import ModerationAction, Observation

load_dotenv()

# ── Task config ─────────────────────────────────────────────────────────────
_TASK_OPTIONS = {
    "🗑️ Spam Detection  (Easy)":          "spam_detection",
    "☣️ Toxicity Classification  (Medium)": "toxicity_classification",
    "🧩 Contextual Moderation  (Hard)":     "contextual_moderation",
}

_ACTION_EMOJIS = {
    "ALLOW": "✅ ALLOW",
    "FLAG": "🚩 FLAG",
    "DELETE": "🗑️ DELETE",
    "ESCALATE": "🚨 ESCALATE",
}

_COLOUR = {
    "positive": "🟢",
    "partial":  "🟡",
    "negative": "🔴",
}

_ENVS: dict[str, ToxiCleanEnv] = {}
_HISTORY: dict[str, list[dict]] = {}


def _get_env(task_name: str) -> ToxiCleanEnv:
    if task_name not in _ENVS:
        _ENVS[task_name] = ToxiCleanEnv(task_name=task_name)
    return _ENVS[task_name]


def _reward_badge(reward: float) -> str:
    if reward > 0:
        return f"{_COLOUR['positive']} **+{reward:.2f}**"
    elif reward == 0:
        return f"{_COLOUR['partial']} **{reward:.2f}**"
    else:
        return f"{_COLOUR['negative']} **{reward:.2f}**"


def _score_bar(score: float) -> str:
    filled = int(score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"`[{bar}]` **{score:.1%}**"


def _format_obs(obs: Observation) -> str:
    meta = obs.metadata
    lang_flag = {"en": "🇬🇧", "hi": "🇮🇳", "hinglish": "🇮🇳🇬🇧"}.get(meta.language, "🌐")
    platform_icon = {"comments": "💬", "public_feed": "📢", "direct_messages": "✉️"}.get(meta.platform, "📄")
    history_badge = {
        "clean": "🟢 Clean",
        "1 prior warning": "🟡 1 Warning",
        "repeat offender": "🔴 Repeat Offender",
    }.get(meta.user_history, meta.user_history)

    return f"""### 📋 Content to Moderate

> {obs.content}

---
{platform_icon} **Platform:** `{meta.platform}` &nbsp;|&nbsp; {lang_flag} **Language:** `{meta.language}` &nbsp;|&nbsp; 👤 **User:** {history_badge}

*Step {obs.step_index + 1} · Task: `{obs.task_name}`*"""


def _history_table(history: list[dict]) -> str:
    if not history:
        return "*No steps taken yet.*"
    rows = ["| Step | Action | Reward | Correct | Result |",
            "|------|--------|--------|---------|--------|"]
    for h in history:
        reward = h["reward"]
        badge = "✅" if reward > 0 else ("⚠️" if reward == 0 else "❌")
        rows.append(
            f"| {h['step']} | `{h['action']}` | `{reward:+.2f}` | `{h['correct']}` | {badge} |"
        )
    return "\n".join(rows)


# ── Callbacks ────────────────────────────────────────────────────────────────

def start_episode(task_label: str):
    task_name = _TASK_OPTIONS[task_label]
    env = ToxiCleanEnv(task_name=task_name)
    _ENVS[task_name] = env
    _HISTORY[task_name] = []
    obs = env.reset()
    state = env.state()

    obs_md   = _format_obs(obs)
    status   = f"**Episode started** — {task_label.strip()} &nbsp;·&nbsp; Step **1 / {state.total_steps}**"
    score_md = _score_bar(0.0)
    hist_md  = "*No steps taken yet.*"
    reward_md = ""
    explain_md = ""
    progress = 0.0

    return obs_md, status, reward_md, explain_md, score_md, hist_md, progress


def take_action(task_label: str, action_str: str):
    task_name = _TASK_OPTIONS[task_label]

    if task_name not in _ENVS:
        return (
            "⚠️ *No active episode — click **▶ Start Episode** first.*",
            "", "", "", _score_bar(0.0), "*No steps taken yet.*", 0.0,
        )

    env = _ENVS[task_name]
    if env.state().done:
        return (
            "✅ *Episode complete — click **▶ Start Episode** to run again.*",
            "**Episode done.**", "", "", _score_bar(0.0), _history_table(_HISTORY.get(task_name, [])), 1.0,
        )

    try:
        action = ModerationAction(action_str.upper())
    except ValueError:
        return ("❌ *Invalid action.*", "", "", "", _score_bar(0.0), "*—*", 0.0)

    next_obs, reward, done, info = env.step(action)
    state = env.state()

    # Log step history
    hist = _HISTORY.setdefault(task_name, [])
    hist.append({
        "step":    state.step_index,
        "action":  action.value,
        "reward":  reward,
        "correct": info.get("correct_action", "?"),
        "done":    done,
    })

    # Build explanation card
    correct = info.get("correct_action", "?")
    grader  = info.get("grader_score", "?")
    reason  = info.get("reason", "")
    tip     = info.get("context_tip", "")

    correct_emoji = _ACTION_EMOJIS.get(correct, correct)
    your_emoji    = _ACTION_EMOJIS.get(action.value, action.value)

    explain_parts = [
        f"**Your action:** {your_emoji} &nbsp;→&nbsp; **Correct:** {correct_emoji}",
        f"**Grader score:** `{grader}` &nbsp;·&nbsp; **Reward:** {_reward_badge(reward)}",
        f"> {reason}",
    ]
    if tip:
        explain_parts.append(f"💡 *{tip}*")
    explain_md = "\n\n".join(explain_parts)

    # Normalised running score
    norm_score = max(0.0, min(1.0, (state.cumulative_reward / max(state.step_index, 1) + 1.2) / 2.4))

    if done:
        summary     = info.get("episode_summary", {})
        final_score = summary.get("final_score", 0.0)
        norm_score  = max(0.0, min(1.0, (final_score + 1.2) / 2.4))

        obs_md  = f"""### ✅ Episode Complete!

**Final score:** {_score_bar(norm_score)}

**Cumulative reward:** `{state.cumulative_reward:.2f}` across `{state.total_steps}` steps."""
        status  = f"**Done** — Final score **{norm_score:.1%}** · Cumulative reward `{state.cumulative_reward:.2f}`"
        return obs_md, status, _reward_badge(reward), explain_md, _score_bar(norm_score), _history_table(hist), norm_score

    obs_md = _format_obs(next_obs)
    status = (
        f"Step **{state.step_index} / {state.total_steps}** &nbsp;·&nbsp; "
        f"Cumulative reward `{state.cumulative_reward:.2f}` &nbsp;{_reward_badge(reward)}"
    )
    return obs_md, status, _reward_badge(reward), explain_md, _score_bar(norm_score), _history_table(hist), norm_score


# ── UI ───────────────────────────────────────────────────────────────────────

_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(
    theme=_THEME,
    title="ToxiClean AI — Content Moderation Dashboard",
    server_name="0.0.0.0",
    server_port=int(os.getenv("PORT", "7860")),
    css="""
    .reward-card { border-radius: 12px; padding: 12px; }
    .score-row { font-size: 1.1em; }
    footer { display: none !important; }
    .gr-prose h3 { margin-top: 0.5em; color: #7c3aed; }
    """,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────
    gr.Markdown("""
# 🧹 ToxiClean AI
### OpenEnv RL Environment — Content Moderation Dashboard

An AI agent learns to moderate online content across **3 tasks of increasing difficulty**.
Select a task, start an episode, then pick the right moderation action for each piece of content.
    """)

    # ── Controls row ──────────────────────────────────────────────────────
    with gr.Row():
        task_dd   = gr.Dropdown(
            choices=list(_TASK_OPTIONS.keys()),
            value=list(_TASK_OPTIONS.keys())[0],
            label="🎯 Task",
            scale=4,
        )
        start_btn = gr.Button("▶ Start Episode", variant="primary", scale=1)

    status_box = gr.Markdown(value="*Select a task and click **▶ Start Episode** to begin.*")

    # ── Main content ──────────────────────────────────────────────────────
    with gr.Row():
        # Left: observation + action
        with gr.Column(scale=3):
            obs_box = gr.Markdown(
                value="*Your content-to-moderate will appear here.*",
                label="Content",
            )

            with gr.Row():
                action_radio = gr.Radio(
                    choices=["ALLOW", "FLAG", "DELETE", "ESCALATE"],
                    value="FLAG",
                    label="⚡ Your Moderation Action",
                    scale=3,
                )
                step_btn = gr.Button("Submit Action", variant="secondary", scale=1, size="lg")

            explain_box = gr.Markdown(value="", label="Grader Feedback")

        # Right: score + history
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Running Score")
            score_md    = gr.Markdown(value=_score_bar(0.0))
            progress_sl = gr.Slider(minimum=0, maximum=1, value=0, interactive=False, label="Score (0 → 1)")
            reward_md   = gr.Markdown(value="", label="Last Reward")

            gr.Markdown("### 📜 Step History")
            hist_box = gr.Markdown(value="*No steps yet.*")

    # ── Reference tables ──────────────────────────────────────────────────
    with gr.Accordion("📖 Quick Reference", open=False):
        with gr.Row():
            gr.Markdown("""
**Action Space**

| Action | When to use |
|--------|-------------|
| ✅ `ALLOW` | Safe content |
| 🚩 `FLAG` | Suspicious — human review |
| 🗑️ `DELETE` | Clearly harmful |
| 🚨 `ESCALATE` | Critical threat |
            """)
            gr.Markdown("""
**Reward Design**

| Outcome | Reward |
|---------|--------|
| Correct | `+1.0` |
| Partial | `+0.5` |
| False positive | `−0.7` |
| Missed harm | `−1.0` |
| Over-moderation | `−0.5` |
            """)
            gr.Markdown("""
**Tasks**

| Task | Difficulty |
|------|-----------|
| Spam Detection | 🟢 Easy |
| Toxicity Classification | 🟡 Medium |
| Contextual Moderation | 🔴 Hard |
            """)

    # ── Wire events ───────────────────────────────────────────────────────
    _OUTPUTS = [obs_box, status_box, reward_md, explain_box, score_md, hist_box, progress_sl]

    start_btn.click(fn=start_episode, inputs=[task_dd], outputs=_OUTPUTS)
    step_btn.click(fn=take_action, inputs=[task_dd, action_radio], outputs=_OUTPUTS)


if __name__ == "__main__":
    demo.launch(show_error=True)
