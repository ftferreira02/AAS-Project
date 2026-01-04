# Model Benchmark Report

This document compares the performance of 5 different model configurations for Phishing Detection.

## Executive Summary
| Metric | Winner | Result |
| :--- | :--- | :--- |
| **User Experience** | `xgb_calibrated` | **36** False Positives |
| **Detection Power** | `xgb` | **19,741** Phishing Sites Caught |
| **Production Choice** | **Hybrid (XGB+CNN)** | **Best Balanace** (9 FP, 96.5% Recall) |

---

## Detailed Metrics

| Model | Accuracy | False Positives (Lower is Better) | False Negatives (Lower is Better) | Phishing Recall | Precision (Class 1) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 99.48% | 91 | 364 | 98.18% | 99.54% |
| **XGBoost (Raw)** | **99.55%** | 135 | **258** | **98.71%** | 99.32% |
| **Logistic Regression** | 99.32% | 273 | 320 | 98.40% | 98.63% |
| **RF Calibrated** | 99.35% | 48 | 525 | 97.37% | 99.75% |
| **XGB Calibrated** | 99.24% | 36 | 634 | 96.83% | 99.81% |
| **Hybrid (XGB+CNN)** | 99.20% | **9** | 695 | 96.52% | **99.95%** |

---

## Analysis

### 1. Hybrid Superiority (The "Safety King")
*   **False Positives**: Dropped from **36** (XGB-Calibrated) to just **9** (Hybrid).
    *   This is a **75% reduction** in false alarms.
    *   Out of ~68,000 safe sites, only 9 were wrongly flagged. This is near-perfect safety.
*   **Trade-off**: Slightly higher miss rate (695 misses vs 634). We trade ~60 missed phishing sites to save ~27 innocent sites. In a user-facing tool, this is the correct trade.

### 2. Calibration vs Raw
*   **Impact**: Calibration significantly reduces False Positives but reduces Phishing Recall.
*   **Example**: `xgb` -> `xgb_calibrated`:
    *   **False Positives**: Dropped from **135** to **36** (-73% noise).
    *   **False Negatives**: Rose from **258** to **634** (+145% missed).
*   **Why?**: The calibration step realizes that many of the sites `xgb` was "confident" about were actually borderline, so it lowered their probability scores. This prevents blocking legitimate sites that look slightly weird.

### 3. Random Forest vs XGBoost
*   **Uncalibrated**: `xgb` beats `rf` in detection (258 misses vs 364 misses).
*   **Calibrated**: `xgb_calibrated` beats `rf_calibrated` in safety (36 FP vs 48 FP).

---

## Recommendation
We should continue using **Hybrid (XGB + CNN)** for the production API.

*   **Reason**: In a browser extension, **False Positives are fatal**. If we block Google or YouTube once, the user uninstalls. 9 False Positives (out of 67,000 safe sites) is an incredible 0.01% error rate.
*   **Trade-off**: Missing ~695 phishing sites (out of 20,000) is acceptable (96.5% catch rate) to ensure a smooth user experience.