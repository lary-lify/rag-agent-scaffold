.PHONY: install test lint format build-index run docker-milvus

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install || true

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

build-index:
	python -m rag_agent.cli build-index

run:
	uvicorn app.main:app --reload --port 8000

docker-milvus:
	docker compose -f docker-compose.milvus.yml up --build
