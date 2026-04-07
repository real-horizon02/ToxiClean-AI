# ToxiClean AI 🧹

> **An OpenEnv reinforcement learning environment where an AI agent learns to moderate online content — intelligently, contextually, and at scale.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-orange)](https://huggingface.co/spaces)

---

## 🌍 Problem Statement

Content moderation is one of the most critical — and hardest — problems facing online platforms today.  Billions of pieces of content are published every day.  Human moderators face burnout, inconsistency, and cultural blind spots.

ToxiClean AI frames content moderation as a **reinforcement learning problem**: an agent learns *what to do* with a piece of content by receiving structured reward signals tied to real moderation policies.

Unlike toy classifiers, ToxiClean treats moderation as a **sequential decision-making problem** with:
- Multiple action choices with different consequences
- Contextual nuance (sarcasm, idioms, Hinglish code-switching)
- Reputation-aware penalties (stricter on repeat offenders)
- Explainable feedback for every decision

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│  OpenEnv Interface                               │
│  ─────────────────                               │
│  reset()  →  Observation                        │
│  step()   →  (Observation, reward, done, info)  │
│  state()  →  EnvironmentState                   │
└──────────────────┬───────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  ToxiCleanEnv       │
        │  environment.py     │
        └──────────┬──────────┘
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
  spam.py    toxicity.py  contextual.py
  (Easy)      (Medium)      (Hard)
```


---

## 🖥️ Web Interface (Gradio)

ToxiClean AI includes a built-in interactive dashboard developed with Gradio. This allows human judges and developers to:
- **Test in Real-Time**: Select from different moderation tasks and moderate samples manually.
- **Visual Rewards**: See immediate reward feedback (🟢/🔴/🟡) for every action.
- **Explainable Moderation**: View the reasoning behind every grader decision.
- **Running Scores**: Monitor normalized performance (0.0 to 1.0) throughout the episode.

---

## ⚙️ Action Space

| Action     | When to use                                              |
|------------|----------------------------------------------------------|
| `ALLOW`    | Content is safe — no action needed                      |
| `FLAG`     | Suspicious — queue for human review                     |
| `DELETE`   | Clearly harmful — remove immediately                    |
| `ESCALATE` | Critical threat (violence, doxxing, self-harm)          |

---

## 👁️ Observation Space

Every step returns a structured `Observation`:

```json
{
  "content": "Buy now!!! Limited offer!!! Click here → http://bit.ly/XrT9",
  "metadata": {
    "user_history": "repeat offender",
    "platform": "comments",
    "language": "en"
  },
  "step_index": 0,
  "task_name": "spam_detection"
}
```

---

## 🎯 Tasks

### Task 1 — Spam Detection (Easy)
- **Goal:** Identify promotional spam, phishing links, and get-rich-quick schemes.
- **Languages:** English + Hinglish
- **Samples:** 12 (6 spam, 6 clean)
- **Grader:** Exact match = 1.0 | FLAG for spam = 0.5 | Wrong = 0.0

### Task 2 — Toxicity Classification (Medium)
- **Goal:** Classify toxicity severity (clean / mild / severe / critical threat).
- **Languages:** English + Hinglish
- **Samples:** 12 across all severity levels
- **Grader:** Exact = 1.0 | Adjacent severity = 0.5 | Far off = 0.0

### Task 3 — Contextual Moderation (Hard)
- **Goal:** Understand intent, not just keywords.
  - `"This bug is killing me 😂"` → **ALLOW** (idiomatic frustration)
  - `"Kill yourself."` → **ESCALATE** (direct self-harm instruction)
  - Polite wording of hate speech → **DELETE**
- **Languages:** English + Hinglish
- **Samples:** 12 contextual edge cases
- **Grader:** Strict — missing an ESCALATE case gives 0.0 (no partial credit)

---

## 💰 Reward Design

| Outcome                        | Reward  |
|--------------------------------|---------|
| Correct moderation             | `+1.0`  |
| Partially correct              | `+0.5`  |
| False positive                 | `−0.7`  |
| Missed harmful content         | `−1.0`  |
| Over-moderation (unnecessary escalation) | `−0.5` |

### Reputation Modifier

The reward is further adjusted based on the submitting user's history:

| User History      | Correct Bonus | Incorrect Penalty |
|-------------------|---------------|-------------------|
| Repeat offender   | +0.10         | −0.20             |
| 1 prior warning   | +0.05         | −0.10             |
| Clean             | 0.00          | 0.00              |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-org/toxiclean-ai.git
cd toxiclean-ai
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY
```

### 3. Run the baseline

```bash
# Run all three tasks
python inference.py

# Run a specific task
python inference.py --task spam_detection
python inference.py --task toxicity_classification
python inference.py --task contextual_moderation

# Verbose mode (shows per-step reasoning)
python inference.py --verbose

# Use a different model
python inference.py --model gpt-4o
```

### 4. Launch the Web UI

Run the interactive Gradio demo locally:

```bash
python app.py
```

Visit `http://localhost:7860` in your browser to start moderating!

---

## 🐳 Docker

### Build

```bash
docker build -t toxiclean-ai .
```

### Run

```bash
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -e MODEL_NAME=gpt-4o-mini \
  toxiclean-ai
```

### With custom API base (e.g., Together AI, Groq)

```bash
docker run --rm \
  -e OPENAI_API_KEY=your-key \
  -e API_BASE_URL=https://api.together.xyz/v1 \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2 \
  toxiclean-ai
```

---

## 📊 Baseline Results

Results using `gpt-4o-mini` at temperature 0:

| Task                     | Final Score |
|--------------------------|-------------|
| Spam Detection           | 0.71        |
| Toxicity Classification  | 0.65        |
| Contextual Moderation    | 0.58        |
| **Overall**              | **0.65**    |

Scores are normalised: `(mean_reward + 1.2) / 2.4`, clamped to `[0, 1]`.

---

## 🧩 Programmatic Usage

```python
from env import ToxiCleanEnv, ModerationAction

env = ToxiCleanEnv(task_name="spam_detection")
obs = env.reset()

while True:
    # Your agent logic here
    action = ModerationAction.FLAG

    obs, reward, done, info = env.step(action)
    print(f"reward={reward} | reason={info['reason']}")

    if done:
        print(f"Final score: {info['episode_summary']['final_score']}")
        break
```

---

## 📁 Project Structure

```
ToxiClean AI/
├── env/
│   ├── __init__.py
│   ├── environment.py      ← Core ToxiCleanEnv (OpenEnv interface)
│   ├── models.py           ← Pydantic typed models
│   └── tasks/
│       ├── __init__.py
│       ├── spam.py         ← Task 1: Spam detection
│       ├── toxicity.py     ← Task 2: Toxicity classification
│       └── contextual.py   ← Task 3: Contextual moderation
│
├── execution/              ← Existing deterministic scripts
├── directives/             ← SOPs (AGENTS.md architecture)
│
├── inference.py            ← Baseline LLM agent runner
├── openenv.yaml            ← OpenEnv specification
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🌐 Multi-Language Support

ToxiClean natively supports:
- **English** — all three tasks
- **Hinglish** — Hindi + English code-switching, common in South Asian social media

The agent must recognise culturally-specific idioms, slang, and threat patterns in both languages.

---

## 🔐 Security

- All secrets stored in `.env` — never hardcoded
- Pydantic v2 input validation on every observation and action
- Non-root Docker user
- No arbitrary code execution
- LLM responses validated before use

---

## 🏆 Hackathon Highlights

| Feature                        | Status |
|-------------------------------|--------|
| Full OpenEnv compliance        | ✅     |
| Three difficulty tiers         | ✅     |
| Deterministic graders          | ✅     |
| Structured reward function     | ✅     |
| Multi-language (EN + Hinglish) | ✅     |
| Reputation-aware rewards       | ✅     |
| Explainable feedback in `info` | ✅     |
| Baseline inference script      | ✅     |
| Docker support                 | ✅     |
| Hugging Face Spaces ready      | ✅     |

---

## 📄 License

MIT © ToxiClean AI Team
