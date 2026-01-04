# False Negative Analysis

We analyzed the phishing URLs that the Hybrid System missed (False Negatives). These "misses" allow a user to visit a malicious site without a warning.

## Key Findings
The ~695 missed sites generally fall into three distinct categories:

### 1. The "Split Verdict" (Model Disagreement)
**Scenario**: One model flags it, but the other votes it down.
*   **Example**: `https://sercure-pagealert.cf/?checkpoint=facebook.com/standart`
    *   **CNN Score**: `0.57` (Warning) -> Saw "sercure" and "facebook".
    *   **Lexical Score**: `0.35` (Safe) -> Trusted the `.cf` domain / lack of URL features.
    *   **Final Score**: `0.48` (Safe).
*   **Fix**: Increasing CNN weight or using a "Max" policy (`max(lex, cnn)`) would catch these but drastically increase False Positives.

### 2. Compromised Legitimate Sites (The Hardest Case)
**Scenario**: High-reputation legitimate domains hacked to host phishing content.
*   **Example**: `https://www.boozyfoodie.co.za/en-US/mpp/security-check-IDPP00C452/`
*   **Scores**: Lexical `0.00`, CNN `0.00`.
*   **Why Missed?**:
    *   Domain (`boozyfoodie.co.za`) is a real, high-reputation food blog.
    *   URL structure is standard.
    *   Phishing content is hidden deep in subfolders.
*   **Solution**: **HTML Content Analysis**. We must scan the page content (e.g., looking for "Login" forms on a blog) to detect this. URL analysis is insufficient.

### 3. "Clean" Phishing URLs
**Scenario**: Phishing sites that strictly mimic legitimate business domains without obvious tricks.
*   **Example**: `https://chinabitmain.com/`
    *   **Analysis**: Looks like a standard business URL. "China" and "Bitmain" are legitimate words.
*   **Example**: `https://pigce.edu.in/id/`
    *   **Analysis**: Educational domain (`.edu.in`), likely compromised.

## Summary

| Category | Frequency | Difficulty to Fix | Proposed Solution |
| :--- | :--- | :--- | :--- |
| **Split Verdict** | Moderate | Low | Adjust ensemble weights (risk of higher FP). |
| **Compromised Site** | High | High | Implement HTML Content Scanning. |
| **Clean Phishing** | Low | High | Domain Age / WHOIS Lookup integration. |

**Verdict**: The current system has reached the limit of **URL-only analysis**. To improve beyond 96.5% recall, future versions must inspect page content.