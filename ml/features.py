import re
import math
from urllib.parse import urlparse
import tldextract

class FeatureExtractor:
    def __init__(self, url):
        self.url = url
        self.parsed = urlparse(url)
        self.domain_info = tldextract.extract(url)

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
        features['is_https'] = 1 if self.parsed.scheme == 'https' else 0 # Disabled due to dataset bias (missing schemes)
        features['has_ip'] = 1 if self._has_ip_address() else 0
        features['suspicious_tld'] = 1 if self.domain_info.suffix in ['tk', 'xyz', 'top', 'club', 'info'] else 0
        
        # Advanced Features
        features['count_subdomains'] = len(self.domain_info.subdomain.split('.')) if self.domain_info.subdomain else 0
        features['is_punycode'] = 1 if 'xn--' in self.url else 0
        
        suspicious_keywords = ['login', 'verify', 'update', 'secure', 'account', 'password', 'confirm', 'signin', 'banking']
        features['has_suspicious_keyword'] = 1 if any(kw in self.url.lower() for kw in suspicious_keywords) else 0
        
        features['count_params'] = len(self.parsed.query.split('&')) if self.parsed.query else 0
        
        # Domain Specific Ratios
        domain_str = self.domain_info.domain
        features['domain_digit_ratio'] = sum(c.isdigit() for c in domain_str) / len(domain_str) if domain_str else 0
        
        # Complexity
        # Analyze Domain Entropy separately (Structure is more important than Path randomness)
        features['entropy'] = self._calculate_entropy(self.url)
        features['domain_entropy'] = self._calculate_entropy(self.domain_info.fqdn)

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

if __name__ == "__main__":
    # Test
    test_url = "http://secure-login.paypal.com.fake.site/login?q=123"
    extractor = FeatureExtractor(test_url)
    print(f"Features for {test_url}:")
    for k, v in extractor.get_features().items():
        print(f"{k}: {v}")
