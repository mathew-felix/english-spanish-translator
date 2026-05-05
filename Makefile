.PHONY: api audit clean dev-install docker-build docker-up lint test

PYTHON ?= python

dev-install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

api:
	$(PYTHON) -m uvicorn src.serve:app --host 127.0.0.1 --port 8000

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest tests -v --cov=src --cov=scripts.download_model --cov-fail-under=60

audit:
	$(PYTHON) -m pip_audit -r requirements.txt

docker-build:
	docker build -t english-spanish-translator:local .

docker-up:
	docker compose up --build

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ['.pytest_cache', '.ruff_cache', 'htmlcov']]"
