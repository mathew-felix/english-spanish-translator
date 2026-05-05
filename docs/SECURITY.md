# Security

## Secret Handling

Secrets must come from environment variables. Do not commit `.env`, W&B API keys,
OpenAI API keys, release tokens, or cloud credentials.

Document required variables in `.env.example` with empty or non-sensitive
placeholder values only.

## Known Remediation Required

The working tree no longer contains the previously exposed W&B key, but git
history still needs to be cleaned before publishing. Rotate the key in W&B and
rewrite history only after confirming the branch coordination impact.

## Dependency Checks

Run:

```bash
python -m pip_audit -r requirements.txt
```

Fix vulnerable pins before releasing a production build.

## Input Validation

The FastAPI request schema rejects empty input, whitespace-only input, extra JSON
fields, and text longer than 500 characters. Keep that validation in place for
all public endpoints.
