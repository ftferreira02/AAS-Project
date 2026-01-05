PYTHON = ./venv/bin/python3
PIP = ./venv/bin/pip
DATASET = ml/data/dataset.csv

.PHONY: setup install train train-lexical train-cnn run clean benchmark

setup:
	python3 -m venv venv

install:
	$(PIP) install -r api/requirements.txt

# Train both models (Hybrid Ensemble)
train: train-lexical train-cnn

train-lexical:
	$(PYTHON) ml/train.py $(DATASET) --model xgb_calibrated

train-cnn:
	$(PYTHON) ml/train.py $(DATASET) --model char_cnn

# Run the API
run:
	$(PYTHON) api/app.py

# Run the Benchmark
benchmark:
	$(PYTHON) ml/benchmark_ensemble.py $(DATASET) --lexical-model ml/runs/xgb_calibrated/model.pkl --cnn-model ml/runs/char_cnn

# Clean artifacts
clean:
	rm -rf venv ml/__pycache__ api/__pycache__ ml/runs/xgb_calibrated ml/runs/char_cnn ml/data/*cache.csv
