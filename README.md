# Phishing URL Detector

A privacy-preserving browser extension that detects phishing attempts using a state-of-the-art **Hybrid Ensemble** of Machine Learning and Deep Learning models.

## Overview
This project implements a system to detect phishing websites in real-time. Unlike traditional blacklist methods, our approach analyzes the *structure* of the URL itself (length, special characters, entropy) using a Lexical Model, and the *visual character patterns* using a Deep Learning model.

## Core Architecture

### 1. Hybrid Ensemble Model
We combine two powerful models to achieve maximum safety and detection power:
*   **Lexical Model (XGBoost Calibrated)**: Analyzes 20+ feature counts (e.g., dot count, entropy, length). Excellent at statistical detection.
*   **Visual Model (Char-CNN)**: A Deep Learning Convolutional Neural Network that reads the URL character-by-character to detect patterns (e.g., `g0ogle`, `secure-login`) that standard features miss.
*   **Ensemble Policy**: `Final Score = (0.4 * Lexical) + (0.6 * CNN)`.

### 2. Probabilistic Decision Policy
Instead of a binary Safe/Unsafe check, we use a calibrated probability system:
*   **Safe (< 60%)**: Allow access (Badges Green).
*   **Warning (60% - 85%)**: Show warning UI, allow proceed (Badge Orange).
*   **Unsafe (> 85%)**: Block access immediately (Badge Red).

## Performance
*   **Accuracy**: **99.20%**
*   **Phishing Recall**: **96.52%** (Caught 19,304 of 20,000 phishing sites)
*   **False Positive Rate**: **0.013%** (Only 9 false alarms out of 68,000 safe sites).

## Installation

### 1. Setup (Makefile)
The project includes a `Makefile` for one-shot setup and running.

```bash
# 1. Setup Environment & Install Dependencies
make setup
make install

# 2. Train the Models (XGBoost + Char-CNN)
# Expects ml/data/dataset2.csv
make train

# 3. Run the API (Serves the Ensemble)
make run
```

### 2. Extension Setup
1.  Open Chrome and navigate to `chrome://extensions`.
2.  Enable **Developer Mode** (top right toggle).
3.  Click **Load Unpacked**.
4.  Select the `extension/` directory of this project.

## Methodology
We extract **lexical features** from the URL string to train our model. The content of the webpage is *never* accessed, making this approach fast and secure.

**Key Features Extracted:**
*   **Structural**: URL length, domain length, path depth.
*   **Statistical**: Count of special characters (`@`, `-`, `.`), digit-to-letter ratios.
*   **Complexity**: Shannon entropy (to detect random gibberish).
*   **Patterns**: Suspicious TLDs, IP address usage, direct login keywords.
*   **Deep Learning**: Character-level embedding vectors (Char-CNN).
