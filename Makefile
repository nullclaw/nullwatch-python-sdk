.PHONY: install lint fmt test

install:
	pip install -e ".[rag,dev]"
	pip install ruff

lint:
	ruff check nullwatch/ tests/

fmt:
	ruff format nullwatch/ tests/ examples/

test:
	pytest
