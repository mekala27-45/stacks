.PHONY: setup evidence reports check api web build

setup:
	uv sync --frozen
	cd web && npm ci

evidence:
	uv run python scripts/crosscheck_obp.py
	uv run python scripts/build_evidence.py
	uv run python scripts/render_reports.py

reports:
	uv run python scripts/render_reports.py

check:
	uv run pytest
	uv run python scripts/render_reports.py --check
	cd web && npm run typecheck
	cd web && npm test

api:
	uv run uvicorn packages.api.main:app --host 127.0.0.1 --port 8000 --no-access-log

web:
	cd web && npm run dev

build: check
	cd web && npm run build
