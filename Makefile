clean:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache && rm -f .coverage coverage.xml && find . -type d -name __pycache__ -exec rm -r {} +

format:
	black . && \
	ruff check --fix . && \
	ruff format . && \
	isort .

lint:
	isort . --check && \
	black . --check && \
	ruff check .

test:
	pytest -ra --cov=src --cov-report=term --cov-report=xml

.PHONY: clean lint format test