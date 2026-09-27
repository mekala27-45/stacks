.PHONY: setup evidence reports check api web build

setup:
	uv sync --frozen
	cd web && npm ci

evidence:
	uv run python scripts/crosscheck_obp.py
	uv run python scripts/build_evidence.py
	uv run python scripts/build_neural_evidence.py
	uv run python scripts/render_reports.py
	uv run python scripts/build_notebooks.py

reports:
	uv run python scripts/render_reports.py

check:
	uv run ruff check .
	uv run mypy
	uv run python scripts/check_no_em_dash.py
	uv run python scripts/check_vocabulary.py
	uv run python scripts/validate_palette.py
	uv run pytest --cov=packages --cov-branch --cov-fail-under=80
	uv run python scripts/render_reports.py --check
	uv run python scripts/check_notebooks.py
	cd web && npm run typecheck
	cd web && npm test
	node --test packages/edge/test-worker.mjs

api:
	uv run uvicorn packages.api.main:app --host 127.0.0.1 --port 8000 --no-access-log

web:
	cd web && npm run dev

build: check
	cd web && npm run build
