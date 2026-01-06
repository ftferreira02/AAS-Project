# Phishing URL Detection

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

## Introduction

This repository contains the source code for the Practical Project of the **Aprendizagem Aplicada à Segurança (AAS)** course.

The objective of this project is to **classify URLs as benign or phishing using only the URL string** (lexical analysis), without fetching page content. We compare several lightweight classical ML models trained on engineered URL features (LogReg, RF, XGBoost and calibrated variants), and include a **character-level CNN baseline** and an **optional hybrid ensemble** for comparison.

**Authors:**
* Francisco Ferreira 124467
* Sara Almeida 108796

## Prerequisites

* Python 3.12+ (Required for TensorFlow 2.16 compatibility)
* Pip (Python Package Installer)
* Make (Optional, for simplified commands)

## Installation

It is strongly recommended to run this project inside a virtual environment to avoid dependency conflicts.

### 1. Extract the compressed file
```bash
# If provided as tarball
tar xvf aas-project.tar.xz 
cd aas-project
```

### 2. Create and Activate Virtual Environment

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### 3. Install Dependencies

```bash
```bash
pip install -r api/requirements.txt
```

## Dataset

This project utilizes the **Malicious and Benign URLs** dataset from Kaggle. 

*   **Source:** [Malicious and Benign URLs Dataset](https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls)
*   **Setup:** Download the dataset and place the `dataset.csv` file in the `ml/data/` folder.
*   **Preprocessing:** The project automatically handles feature extraction and caching upon the first run.

The dataset contains 450,176 labelled URLs, with 345,738 benign (76.8\%) and 104,438 malicious (23.2\%) samples. The report experiments were conducted with hostname-grouped test set of 78,214 samples.

## Usage

### 1. Train the Model

To retrain the lexical model:

```bash
# Using Makefile (Recommended)
make train

# Or manually:
python ml/train.py ml/data/dataset.csv --model xgb_calibrated
```

**Available Models and how to run them:**
*   `rf`: Random Forest
```bash
python ml/train.py ml/data/dataset.csv --model rf
```
*   `xgb`: XGBoost
```bash
python ml/train.py ml/data/dataset.csv --model xgb
```
*   `logreg`: Logistic Regression 
```bash
python ml/train.py ml/data/dataset.csv --model logreg
```
*   `rf_calibrated`: Random Forest with Probability Calibration
```bash
python ml/train.py ml/data/dataset.csv --model rf_calibrated
```
*   `xgb_calibrated`: XGBoost with Probability Calibration (Recommended Lexical)
```bash
python ml/train.py ml/data/dataset.csv --model xgb_calibrated
```
*   `char_cnn`: Character-Level CNN (Recommended Visual)
```bash
python ml/train.py ml/data/dataset.csv --model char-cnn
```
*  Hybrid Emsemble (Lexical XGB Calibrated + Char-CNN) - Combine both models.
```bash
python ml/train.py ml/data/dataset.csv --model xgb_calibrated
python ml/train.py ml/data/dataset.csv --model char_cnn
```

### 2. Run the API (Backend)

Start the Flask API to serve predictions:

```bash
make run
# or
python api/app.py
```

The API exposes:
- `GET /health`
- `POST /predict` (expects a JSON payload containing a URL and returns a score + risk tier)

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/login"}'
```

### 3. Advanced Configuration (Optional)

You can customize the API behavior by setting environment variables **in your terminal** before running the app.

**Option A: Disable the Neural Network (Lexical Only)**
Use this if you want faster performance (at the cost of some detection power).
```bash
# Set the flag before the command:
ENABLE_CNN=false make run
```

**Option B: Use a Different Model**
If you want to test the Random Forest or Logistic Regression models instead of the default XGBoost:
```bash
# Point to the specific model file:
MODEL_PATH=ml/runs/rf/model.pkl make run
```

### 3. Browser Extension

1.  Navigate to `chrome://extensions` in Chromium/Chrome.
2.  Enable **Developer Mode**.
3.  Click **Load Unpacked** and select the `extension/` directory.


## Major Results

All results below are reported on the **hostname-grouped test set** using a **0.5 threshold** for fair comparison across models.

### Best deployed model (lexical-only): `XGB_calibrated`

- Accuracy: **0.9873**
- Phishing precision (class 1): **0.9999**
- Phishing recall (class 1): **0.9512**
- Phishing F1 (class 1): **0.9750**
- Confusion matrix: **TN=57897, FP=1, FN=992, TP=19324**

### Deep learning baseline (Char-CNN, comparison only)

- Accuracy: **0.9971**
- Confusion matrix: **TN=57814, FP=84, FN=143, TP=20173**

### Hybrid ensemble (0.4 lexical + 0.6 Char-CNN, comparison only)

- Accuracy: **0.9959**
- Confusion matrix: **TN=57885, FP=13, FN=311, TP=20005**

### Key findings

- Calibrated XGBoost yields an extremely low **false positive** count at threshold 0.5 (FP=1), which is desirable for user-facing warning/blocking actions.
- Char-CNN and the hybrid approach show strong classification performance, but introduce heavier runtime dependencies and operational complexity.
- The deployed extension uses a three-level policy (SAFE/WARNING/UNSAFE) with stricter thresholds than 0.5 to reduce disruption.

---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
