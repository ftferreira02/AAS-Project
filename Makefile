PYTHON = ./venv/bin/python3
PIP = ./venv/bin/pip

.PHONY: setup install train run clean

setup:
	python3 -m venv venv

install:
	$(PIP) install -r api/requirements.txt

train:
	$(PYTHON) ml/train.py ml/data/dataset2.csv

run:
	$(PYTHON) api/app.py

clean:
	rm -rf venv ml/__pycache__ api/__pycache__ ml/model.pkl
