# moderation_classify_text.md
# Layer 1 Directive — Text Toxicity Classification

## Goal
Classify a piece of user-submitted text as toxic / non-toxic using the configured LLM, and return a structured result with confidence score and reasoning.

## Inputs
| Field      | Type   | Required | Notes                              |
|------------|--------|----------|------------------------------------|
| `text`     | string | ✅       | Max 10,000 chars. UTF-8.           |
| `user_id`  | string | ✅       | For rate-limiting and audit logs.  |
| `language` | string | ❌       | ISO 639-1 code, default `"en"`.    |

## Tools
1. `execution/utils/validator.py` → validate input schema
2. `execution/utils/rate_limiter.py` → enforce per-user rate limit (60 req/min)
3. `execution/moderation/classify_text.py` → call LLM classification API

## Expected Output
```json
{
  "is_toxic": true,
  "confidence": 0.94,
  "categories": ["hate_speech", "harassment"],
  "reasoning": "...",
  "user_id": "u_123",
  "processed_at": "2026-04-07T17:00:00Z"
}
```

## Edge Cases
- **Empty text after strip**: Reject with `ValidationError` before hitting LLM.
- **Rate limit exceeded**: Return HTTP 429 with `Retry-After: 60` header.
- **LLM timeout**: Retry up to 3× with exponential backoff (1s, 2s, 4s).
- **LLM returns ambiguous result**: Default `is_toxic=false`, log for manual review.
- **Language not supported**: Proceed in English, flag `language_mismatch=true` in response.

## Changelog
- 2026-04-07: Initial version (instantiation)
