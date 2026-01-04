False Negative Analysis
We analyzed the phishing URLs that the Hybrid System missed (False Negatives). These are dangerous because they allow a user to visit a malicious site.

key Findings
The "missed" sites generally fall into three categories:

1. The "Split Verdict" (Model Disagreement)
In these cases, one model correctly flagged the site, but the other "voted it down" because the weights (0.4 / 0.6) weren't enough to push it over 0.60.

URL: https://sercure-pagealert.cf/?checkpoint=facebook.com/standart
CNN Score: 0.57 (Warning) -> Correctly saw "sercure" and "facebook".
Lexical Score: 0.35 (Safe) -> Incorrectly trusted the .cf domain or lack of other features.
Final: 0.48 (Safe).
Fix: Increasing the CNN weight or using a "Max" policy (instead of Average) would catch this, but might increase False Positives.
2. Compromised Legitimate Sites
The hardest category. These are real, high-reputation domains (like blogs or small business sites) that were hacked to host a phishing page.

URL: https://www.boozyfoodie.co.za/en-US/mpp/security-check-IDPP00C452/
Scores: Lexical 0.00, CNN 0.00.
Why: boozyfoodie.co.za is a real food blog. The domain age, reputation, and character patterns look completely benign. The phishing content is deep in the subfolder /mpp/....
Solution: URL-only models struggle here. We would need Content Analysis (scanning the page HTML for "Login" forms) to catch this.
3. "Clean" Phishing URLs
URLs that legitimately look safe and don't use obvious tricks (no typosquatting, no "secure-login" keywords).

URL: https://chinabitmain.com/
Scores: Lexical 0.30, CNN 0.10.
Why: It looks like a standard business domain. "China" and "Bitmain" are legitimate words.
URL: https://pigce.edu.in/id/
Why: An educational domain (.edu.in). Likely a compromised university site used to host a phishing page.
Summary
Total Misses: ~695 out of 20,000 phishing sites (3.5% miss rate).
Dominant Cause: Compromised legitimate domains (WordPress hacks, etc.).
Mitigation: The URLs themselves are "safe"; the content is not. To fix this in the future, the extension should likely inspect the HTML content (e.g., look for password fields on a non-password domain) in addition to just the URL.