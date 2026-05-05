## Summary

- 

## Validation

- [ ] `python -m ruff check .`
- [ ] `python -m pytest tests -v --cov=serve --cov=scripts.download_model --cov-fail-under=60`
- [ ] `python -m pip_audit -r requirements.txt`

## Artifact Hygiene

- [ ] No secrets committed
- [ ] No generated datasets committed
- [ ] No model checkpoints committed
- [ ] Notebook outputs cleared

## Notes

Add deployment, reproduction, or migration notes here.
