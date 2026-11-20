.PHONY: setup lint test check dry-run splits baselines baselines-vs-ag mr-applicability pilot

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

# Phase 1 on the real data (after `shortcutcheck data register <zip>`). Splits are frozen once.
splits:
	shortcutcheck data split-rq12 --name rq12-DR-AG
	shortcutcheck data split-rq12 --name rq12-DR-AG_deb
	shortcutcheck data split-rq12 --name rq12-VS-AG
	shortcutcheck data split-rq12 --name rq12-VS-AG_deb
	shortcutcheck data split-rq3

baselines:
	shortcutcheck baselines --split rq12-DR-AG
	shortcutcheck baselines --split rq12-DR-AG_deb
	shortcutcheck baselines --split rq12-VS-AG_deb
	shortcutcheck baselines --split rq3-AndroCT-2012-14-vs-2017-19

# About 100k training apps: run on a machine with several cores (1-2 hours).
baselines-vs-ag:
	shortcutcheck baselines --split rq12-VS-AG

# Phase 2, step 2.2: which core MRs apply to each frozen split (training partitions only).
mr-applicability:
	shortcutcheck mrs applicability

# Phase 2, step 2.7: pilot RQ1 on the Drebin mixed set (plan: docs/pilot_plan.md).
pilot:
	shortcutcheck pilot --split rq12-DR-AG
