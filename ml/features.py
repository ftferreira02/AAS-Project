import re
import math
from urllib.parse import urlparse, urlsplit, urlunsplit, parse_qsl, urlencode
import tldextract

class FeatureExtractor:
    def __init__(self, url):
        u = strip_tracking_params(url)
        if not u.startswith(("http://", "https://")):
            u = "http://" + u
        self.url = u
        self.parsed = urlparse(self.url)
        self.domain_info = tldextract.extract(self.url)


    def get_features(self):
        """Returns a dictionary of all extracted features."""
        features = {}
        
        # Structure
        features['url_length'] = len(self.url)
        features['domain_length'] = len(self.domain_info.domain)
        features['path_length'] = len(self.parsed.path)
        features['query_length'] = len(self.parsed.query)
        
        # Character Counts
        features['count_dots'] = self.url.count('.')
        features['count_hyphens'] = self.url.count('-')
        features['count_underscores'] = self.url.count('_')
        features['count_digits'] = sum(c.isdigit() for c in self.url)
        features['count_at'] = self.url.count('@')
        features['count_special'] = sum(not c.isalnum() for c in self.url)

        # Ratios
        total_chars = len(self.url)
        features['ratio_digits'] = features['count_digits'] / total_chars if total_chars > 0 else 0
        features['ratio_special'] = features['count_special'] / total_chars if total_chars > 0 else 0

        # Domain/Protocol
        features['is_https'] = 1 if self.parsed.scheme == 'https' else 0
        features['has_ip'] = 1 if self._has_ip_address() else 0
        features['suspicious_tld'] = 1 if self.domain_info.suffix in ['tk', 'xyz', 'top', 'club', 'info'] else 0
        
        # Advanced Features
        features['count_subdomains'] = len(self.domain_info.subdomain.split('.')) if self.domain_info.subdomain else 0
        features['is_punycode'] = 1 if 'xn--' in self.url else 0
        
        suspicious_keywords = ['login', 'verify', 'update', 'secure', 'account', 'password', 'confirm', 'signin', 'banking']
        u = self.url.lower()
        host = self.parsed.netloc.lower()
        path = self.parsed.path.lower()
        query = self.parsed.query.lower()

        # Split host into parts using tldextract
        registered_domain = (self.domain_info.domain or "").lower()          # e.g., "slbenfica"
        subdomain = (self.domain_info.subdomain or "").lower()               # e.g., "mybenfica"
        suffix = (self.domain_info.suffix or "").lower()                     # e.g., "pt"

        # More precise keyword location features
        features['kw_in_netloc'] = 1 if any(kw in host for kw in suspicious_keywords) else 0
        features['kw_in_registered_domain'] = 1 if any(kw in registered_domain for kw in suspicious_keywords) else 0
        features['kw_in_subdomain'] = 1 if (subdomain and any(kw in subdomain for kw in suspicious_keywords)) else 0

        # Keep the old names too if you want backwards compatibility (optional)
        features['kw_in_domain'] = features['kw_in_netloc']

        features['kw_in_path']  = 1 if any(kw in path for kw in suspicious_keywords) else 0
        features['kw_in_query'] = 1 if any(kw in query for kw in suspicious_keywords) else 0

        # Count total occurrences in full URL (same as before)
        features['kw_count'] = sum(u.count(kw) for kw in suspicious_keywords)
        
        features['count_params'] = len(self.parsed.query.split('&')) if self.parsed.query else 0
        
        # Domain Specific Ratios
        domain_str = self.domain_info.domain
        features['domain_digit_ratio'] = sum(c.isdigit() for c in domain_str) / len(domain_str) if domain_str else 0
        
        # Complexity
        # Entropy split (more informative than entropy of full URL alone)
        features['entropy_url'] = self._calculate_entropy(self.url)
        features['entropy_host'] = self._calculate_entropy(self.parsed.netloc or "")
        features['entropy_path_query'] = self._calculate_entropy((self.parsed.path or "") + "?" + (self.parsed.query or ""))

        # Optional: keep old key name for compatibility (if you don't want to change downstream yet)
        features['entropy'] = features['entropy_url']

        return features

    def _has_ip_address(self):
        # Regex to check for IP address pattern
        ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'
        domain = self.domain_info.domain
        # Check if domain itself looks like an IP or part of the netloc
        if re.search(ip_pattern, self.parsed.netloc):
            return True
        return False

    def _calculate_entropy(self, string):
        """Calculates Shannon entropy of the string."""
        if not string:
            return 0
        
        entropy = 0
        length = len(string)
        
        # Count occurrences of each char
        counts = {}
        for char in string:
            counts[char] = counts.get(char, 0) + 1
            
        for count in counts.values():
            p_x = count / length
            entropy += - p_x * math.log2(p_x)
            
        return entropy


TRACKING_KEYS = {
    "gclid", "fbclid", "gbraid", "wbraid", "gad_source", "gclsrc",
    "gad_campaignid", "dplnk", "ds_rl"
}

def strip_tracking_params(url: str) -> str:
    # Ensure scheme/netloc are present for urlsplit to work reliably on partial URLs
    # But usually we expect valid URLs here.
    s = urlsplit(url)
    q = [
        (k, v) for (k, v) in parse_qsl(s.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in TRACKING_KEYS)
    ]
    new_query = urlencode(q, doseq=True)
    # drop fragment too (it is not sent to servers and often noisy)
    return urlunsplit((s.scheme, s.netloc, s.path, new_query, ""))


if __name__ == "__main__":
    # Test
    test_url = "http://secure-login.paypal.com.fake.site/login?q=123"
    extractor = FeatureExtractor(test_url)
    print(f"Features for {test_url}:")
    for k, v in extractor.get_features().items():
        print(f"{k}: {v}")
