# Hybrid Ensemble Phishing URL Detection

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)

## Introduction

This repository contains the source code for the Practical Project of the **Aprendizagem Aplicada à Segurança (AAS)** course.

The objective of this project is to **detect phishing URLs in real-time** using a **Hybrid Ensemble Model** that combines:
1.  **Lexical Analysis**: An XGBoost classifier trained on 20+ extracted features (entropy, dot counts, keyword presence).
2.  **Visual Pattern Analysis**: A Character-Level Convolutional Neural Network (Char-CNN) that learns suspicious string patterns.

This approach balances **high detection rates** (Recall) with **extremely low false positives** (Safety), making it suitable for a user-facing browser extension.

**Authors:**
* Francisco Ferreira 124467
* Sara Almeida 108796

## Prerequisities

* Python 3.12 or higher
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
pip install -r requirements.txt
```

## Dataset

## Dataset

This project utilizes the **Malicious and Benign URLs** dataset from Kaggle.

*   **Source:** [Malicious and Benign URLs Dataset](https://www.kaggle.com/datasets/siddharthkumar25/malicious-and-benign-urls)
*   **Setup:** Download the dataset and place the `dataset.csv` file in the `ml/data/` folder.
*   **Preprocessing:** The project automatically handles feature extraction and caching upon the first run.

## Usage

### 1. Train the Model

To retrain the complete Hybrid Ensemble (Lexical + CNN):

```bash
# Using Makefile (Recommended)
make train

# Or manually:
python ml/train.py ml/data/dataset2.csv --model xgb_calibrated
python ml/train.py ml/data/dataset2.csv --model char_cnn
```

### 2. Run the API (Backend)

Start the Flask API to serve predictions:

```bash
make run
# or
python api/app.py
```

### 3. Browser Extension

1.  Navigate to `chrome://extensions` in Chromium/Chrome.
2.  Enable **Developer Mode**.
3.  Click **Load Unpacked** and select the `extension/` directory.

## Major Results

The following table summarizes the performance of our best model (**Hybrid Ensemble - Tuned**) on the test set (~78,000 URLs).

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 0.99 | 1.00 | 0.99 | 57,898 |
| **Phishing** | 1.00 | 0.96 | 0.98 | 20,316 |
| **Accuracy** | | | **0.9896** | 78,214 |

**Key Findings:**

  * **Extremely Safe**: We achieved a **False Positive count of only 2** (out of ~58,000 safe sites).
  * **High Detection**: The system caught **96.00%** of all phishing attacks (Recall).
  * **Efficiency**: The ensemble allows for real-time inference suitable for browser navigation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
