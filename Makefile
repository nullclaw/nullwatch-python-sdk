.PHONY: install lint fmt test build check-package

install:
	pip install -r requirements-dev.txt

lint:
	ruff check nullwatch/ tests/

fmt:
	ruff format nullwatch/ tests/ examples/

test:
	pytest

build:
	python -m build

check-package:
	python -m build
	python -m twine check dist/*
