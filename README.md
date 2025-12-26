# Phishing URL Detector

A privacy-preserving browser extension that detects phishing attempts using a lightweight Machine Learning model.

## Overview
This project implements a **Lexical URL Analysis** system to detect phishing websites in real-time. Unlike traditional blacklist methods, our approach analyzes the *structure* of the URL itself (length, special characters, entropy) to identify malicious patterns without needing to download page content or compromise user privacy.

### Core Architecture
The system consists of three main components:

1.  **Chrome Extension**: Intercepts navigation events and queries the analysis engine.
2.  **Analysis API (Flask)**: A Python backend that extracts features and runs the ML model.
3.  **ML Model (Random Forest)**: A trained classifier that predicts if a URL is safe or phishing based on 20+ lexical features.

# Technology Stack
*   **Frontend**: HTML, CSS, JavaScript (Chrome Manifest V3)
*   **Backend**: Python, Flask
*   **Machine Learning**: Scikit-Learn (Random Forest), Pandas
*   **Feature Extraction**: `tldextract`, Entropy analysis, RegEx

## Methodology
We extract **lexical features** from the URL string to train our model. The content of the webpage is *never* accessed, making this approach fast and secure.

**Key Features Extracted:**
*   **Structural**: URL length, domain length, path depth.
*   **Statistical**: Count of special characters (`@`, `-`, `.`), digit-to-letter ratios.
*   **Complexity**: Shannon entropy (to detect random gibberish).
*   **Patterns**: Suspicious TLDs, IP address usage, direct login keywords.

## Installation

### 1. Backend Setup
```bash
# Install dependencies
pip install -r api/requirements.txt

# Train the model (if not already trained)
# Ensure you have your dataset.csv in ml/data/
python ml/train.py ml/data/dataset.csv

# Start the API Server
python api/app.py
```
*The API will run on http://127.0.0.1:5000*

### 2. Extension Setup
1.  Open Chrome and navigate to `chrome://extensions`.
2.  Enable **Developer Mode** (top right toggle).
3.  Click **Load Unpacked**.
4.  Select the `extension/` directory of this project.

##  Usage
1.  Ensure the Python API is running.
2.  Browse the web normally.
3.  The extension icon will show the current status.
4.  If a **Phishing Site** is detected, a red warning overlay will block access (with an option to proceed if you trust the site).

## Performance
*   **Model**: Random Forest Classifier
*   **Accuracy**: ~94% (on validation set)
*   **Inference Time**: < 50ms per URL
