"""
Feature extraction module for URLs and emails
Extracts various features that help identify phishing attempts
"""
import re
import urllib.parse
from urllib.parse import urlparse
import ipaddress
import tldextract

class FeatureExtractor:
    """
    Extract features from URLs and emails for phishing detection
    Implements 30 features commonly used in phishing detection research
    """
    
    def __init__(self):
        """Initialize feature extractor with phishing indicators"""
        # Common phishing keywords found in malicious URLs/emails
        self.phishing_keywords = [
            'verify', 'update', 'confirm', 'account', 'suspended',
            'urgent', 'security', 'login', 'password', 'bank',
            'paypal', 'amazon', 'ebay', 'click', 'here', 'secure',
            'validate', 'restore', 'unlock', 'limited', 'expire'
        ]
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.xyz', '.click', '.download']
    
    def extract_url_features(self, url):
        """
        Extract 30 features from a URL
        
        Args:
            url: URL string to analyze
        
        Returns:
            list: Feature vector with 30 numerical features
        """
        features = []
        
        # Ensure URL has protocol for parsing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split('/')[0]
            path = parsed.path
            query = parsed.query
            
            # Feature 1: URL Length
            features.append(len(url))
            
            # Feature 2: Domain Length
            features.append(len(domain))
            
            # Feature 3: Path Length
            features.append(len(path))
            
            # Feature 4: Query Length
            features.append(len(query))
            
            # Feature 5: Number of dots in domain
            features.append(domain.count('.'))
            
            # Feature 6: Number of hyphens in domain
            features.append(domain.count('-'))
            
            # Feature 7: Number of underscores in domain
            features.append(domain.count('_'))
            
            # Feature 8: Number of slashes in URL
            features.append(url.count('/'))
            
            # Feature 9: Number of question marks
            features.append(url.count('?'))
            
            # Feature 10: Number of equals signs
            features.append(url.count('='))
            
            # Feature 11: Number of ampersands
            features.append(url.count('&'))
            
            # Feature 12: Number of percent signs (URL encoding)
            features.append(url.count('%'))
            
            # Feature 13: Has HTTPS (secure connection)
            features.append(1 if url.startswith('https://') else 0)
            
            # Feature 14: Has IP address in domain (suspicious)
            try:
                ipaddress.ip_address(domain.split(':')[0])  # Remove port if present
                features.append(1)
            except:
                features.append(0)
            
            # Feature 15: Number of subdomains
            ext = tldextract.extract(url)
            subdomain_count = len([s for s in ext.subdomain.split('.') if s]) if ext.subdomain else 0
            features.append(subdomain_count)
            
            # Feature 16: Suspicious TLD
            tld = '.' + ext.suffix if ext.suffix else ''
            features.append(1 if tld.lower() in self.suspicious_tlds else 0)
            
            # Feature 17: Short URL service (often used for phishing)
            short_url_services = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'short.link']
            features.append(1 if any(service in domain.lower() for service in short_url_services) else 0)
            
            # Feature 18: Number of digits in domain
            features.append(sum(c.isdigit() for c in domain))
            
            # Feature 19: Number of digits in URL
            features.append(sum(c.isdigit() for c in url))
            
            # Feature 20: Phishing keywords in URL
            url_lower = url.lower()
            keyword_count = sum(1 for keyword in self.phishing_keywords if keyword in url_lower)
            features.append(keyword_count)
            
            # Feature 21: Has port number
            features.append(1 if ':' in domain and any(c.isdigit() for c in domain.split(':')[-1]) else 0)
            
            # Feature 22: URL depth (path depth)
            features.append(path.count('/') - 1 if path and path != '/' else 0)
            
            # Feature 23: Has @ symbol (suspicious in URL)
            features.append(1 if '@' in url else 0)
            
            # Feature 24: Has double slash in path
            features.append(1 if '//' in path else 0)
            
            # Feature 25: Number of special characters
            special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/', '~', '`']
            features.append(sum(1 for char in url if char in special_chars))
            
            # Feature 26: Entropy of domain (randomness measure)
            features.append(self._calculate_entropy(domain))
            
            # Feature 27: Has suspicious file extension
            suspicious_extensions = ['.exe', '.zip', '.scr', '.bat', '.cmd', '.com', '.pif', '.vbs', '.js']
            features.append(1 if any(ext in path.lower() for ext in suspicious_extensions) else 0)
            
            # Feature 28: URL shortening ratio (domain length / total URL length)
            features.append(len(domain) / max(len(url), 1))
            
            # Feature 29: Has login page keywords
            login_keywords = ['login', 'signin', 'sign-in', 'account', 'verify', 'authenticate', 'signup']
            features.append(1 if any(keyword in path.lower() for keyword in login_keywords) else 0)
            
            # Feature 30: Number of redirects (approximated by redirect keywords in query)
            redirect_keywords = ['redirect', 'url=', 'goto=', 'link=', 'destination=', 'next=']
            features.append(sum(1 for keyword in redirect_keywords if keyword in query.lower()))
            
        except Exception as e:
            # If parsing fails, return default features (likely suspicious)
            features = [0] * 30
            features[0] = len(url)  # At least preserve URL length
            features[13] = 0  # No HTTPS
        
        return features
    
    def extract_email_features(self, email_text):
        """
        Extract features from email text
        Uses URL features if URLs are found in email, otherwise text-based features
        
        Args:
            email_text: Email content to analyze
        
        Returns:
            list: Feature vector with 30 numerical features
        """
        # Extract URLs from email text
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, email_text)
        
        if urls:
            # If URLs found, use URL features from first URL (most common case)
            return self.extract_url_features(urls[0])
        else:
            # Text-based features for email without URLs
            features = []
            
            # Features 1-5: Length features
            features.append(len(email_text))
            features.append(len(email_text.split()))
            features.append(len(email_text.split('\n')))
            features.append(email_text.count(' '))
            features.append(email_text.count('\n'))
            
            # Features 6-10: Character counts
            features.append(sum(c.isdigit() for c in email_text))
            features.append(sum(c.isupper() for c in email_text))
            features.append(sum(c.islower() for c in email_text))
            features.append(email_text.count('!'))
            features.append(email_text.count('?'))
            
            # Features 11-15: Phishing indicators
            email_lower = email_text.lower()
            features.append(sum(1 for keyword in self.phishing_keywords if keyword in email_lower))
            features.append(1 if 'urgent' in email_lower else 0)
            features.append(1 if 'click here' in email_lower or 'clickhere' in email_lower else 0)
            features.append(1 if 'verify' in email_lower else 0)
            features.append(1 if 'suspended' in email_lower else 0)
            
            # Features 16-20: Email structure
            features.append(1 if '@' in email_text else 0)
            features.append(email_text.count('$'))
            features.append(email_text.count('%'))
            features.append(email_text.count('&'))
            features.append(email_text.count('#'))
            
            # Features 21-25: Suspicious patterns
            features.append(1 if re.search(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}', email_text) else 0)  # Credit card pattern
            features.append(1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', email_text) else 0)  # IP address
            features.append(email_text.count('http'))
            features.append(email_text.count('www'))
            features.append(self._calculate_entropy(email_text))
            
            # Features 26-30: Additional phishing indicators
            features.append(1 if 'password' in email_lower else 0)
            features.append(1 if 'account' in email_lower else 0)
            features.append(1 if 'security' in email_lower else 0)
            features.append(1 if 'update' in email_lower else 0)
            features.append(1 if 'confirm' in email_lower else 0)
            
            return features
    
    def _calculate_entropy(self, text):
        """
        Calculate simplified entropy of a string
        Higher entropy indicates more randomness (potentially suspicious)
        
        Args:
            text: Input string
        
        Returns:
            float: Entropy value
        """
        if not text or len(text) == 0:
            return 0
        
        # Calculate character frequency
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy using Shannon entropy formula
        entropy = 0
        length = len(text)
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)  # Simplified entropy calculation
        
        return entropy

