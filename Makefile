PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_STREAMLIT := $(VENV)/bin/streamlit

.PHONY: setup install generate train score pipeline test lint check dashboard

setup: $(VENV_PYTHON)

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

install: $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install -r requirements.txt

generate:
	$(VENV_PYTHON) -m src.generate_data

train:
	$(VENV_PYTHON) -m src.train

score:
	$(VENV_PYTHON) -m src.score --backend duckdb

pipeline: generate train score

test:
	$(VENV_PYTHON) -m pytest -q

lint:
	$(VENV_PYTHON) -m ruff check src dashboard tests

check: test lint
	$(VENV_PYTHON) -m pip check

dashboard:
	$(VENV_STREAMLIT) run dashboard/app.py
