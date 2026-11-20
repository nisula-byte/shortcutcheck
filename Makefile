.PHONY: setup lint test check dry-run

setup:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e . --no-deps
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	mypy

test:
	pytest --cov=shortcutcheck --cov-report=term-missing

check: lint test

# Rehearse all of Phase 1 on synthetic data in a scratch copy (never report these numbers).
dry-run:
	rm -rf /tmp/sc-dry && mkdir -p /tmp/sc-dry && cp pyproject.toml /tmp/sc-dry/
	shortcutcheck synthetic /tmp/sc-dry/zip --config-out /tmp/sc-dry/configs/phase1.yaml
	cd /tmp/sc-dry && shortcutcheck data register zip/synthetic-dada.zip \
	  && shortcutcheck data audit && shortcutcheck data validity \
	  && shortcutcheck data split-rq12 --name rq12-syn --sets SYN-DR-AG \
	  && shortcutcheck data split-rq3 --candidate SYN-2014-vs-2017 --name rq3-syn \
	  && shortcutcheck baselines --split rq12-syn
