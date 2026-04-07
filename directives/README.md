# 📋 Directives — Layer 1 (What To Do)

This folder contains Standard Operating Procedures (SOPs) that define **what** the agent should do.

## Format

Each directive is a Markdown file with this structure:

```markdown
# [Task Name]

## Goal
What the end result should look like.

## Inputs
What data / files / APIs are required.

## Tools
Which scripts in `execution/` to call, in what order.

## Expected Output
What a successful run produces.

## Edge Cases
Known failure modes and how to handle them.

## Changelog
- YYYY-MM-DD: Initial version
```

## Naming Convention

`<domain>_<action>.md` — e.g. `moderation_classify_text.md`, `auth_refresh_token.md`

## Rules

- Directives are **living documents** — update when new constraints are discovered.
- Never overwrite without preserving the original intent.
- Every edge case discovered in production must be captured here.
