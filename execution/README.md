# ⚙️ Execution — Layer 3 (Doing the Work)

This folder contains **deterministic Python scripts** that perform the actual work.

## Rules

1. **Idempotent** — safe to retry; no duplicate side-effects.
2. **No hardcoded secrets** — always read from `.env` via `python-dotenv`.
3. **Fully commented** — every function has a docstring.
4. **Validate all inputs** — use Pydantic models or explicit checks.
5. **Log, don't print** — use the `logging` module; never log secrets.
6. **Graceful errors** — raise typed exceptions; never swallow silently.

## Structure

```
execution/
├── README.md          ← this file
├── utils/
│   ├── __init__.py
│   ├── logger.py      ← shared logging setup
│   ├── validator.py   ← shared input validation helpers
│   └── rate_limiter.py
├── moderation/        ← domain-specific scripts
├── auth/
└── ...
```

## Running a Script

```bash
python execution/<domain>/<script>.py --input <args>
```

All scripts must accept `--dry-run` to simulate without side-effects.
