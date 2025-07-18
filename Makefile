.PHONY: install install-dev run test lint docker-build docker-up

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

run:
	uvicorn app.main:app --reload

test:
	pytest -q

lint:
	ruff check .

docker-build:
	docker compose build

docker-up:
	docker compose up --build

