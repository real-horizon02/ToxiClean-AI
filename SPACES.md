---
title: ToxiClean AI
emoji: 🧹
colorFrom: purple
colorTo: violet
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: OpenEnv RL environment for intelligent content moderation
tags:
  - reinforcement-learning
  - content-moderation
  - openenv
  - nlp
  - hackathon
---

# ToxiClean AI

See [README.md](README.md) for full documentation.

## API Endpoints

The Space exposes the full OpenEnv REST interface:

| Method | Path     | Description                        |
|--------|----------|------------------------------------|
| POST   | /reset   | Reset episode, get first observation |
| POST   | /step    | Take a moderation action           |
| GET    | /state   | Get current environment state      |
| GET    | /health  | Liveness check                     |
| GET    | /ui      | Gradio interactive demo            |
