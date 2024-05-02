clean:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache && rm -f .coverage coverage.xml && find . -type d -name __pycache__ -exec rm -r {} +

test:
	rye run pytest -ra --cov=src --cov-report=term --cov-report=xml

.PHONY: clean test