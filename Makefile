clean:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache && rm -f .coverage coverage.xml && find . -type d -name __pycache__ -exec rm -r {} +

format:
	ruff check --fix . && \
	ruff format .

lint:
	ruff check .

test:
	pytest -ra --cov=src --cov-report=term --cov-report=xml

.PHONY: clean lint format test