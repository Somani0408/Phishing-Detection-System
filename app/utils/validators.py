"""
Input validation utilities
Validates URLs and email text to ensure they meet expected formats
"""
import re
from urllib.parse import urlparse

def validate_url(url):
    """
    Validate URL format
    
    Args:
        url: URL string to validate
    
    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url or len(url) > 2048:  # Max URL length per RFC
        return False
    
    # Add protocol if missing for validation
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    try:
        result = urlparse(url)
        # Check if scheme and netloc are present
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_email_text(email_text):
    """
    Validate email text content
    
    Args:
        email_text: Email text to validate
    
    Returns:
        bool: True if email text is valid, False otherwise
    """
    if not email_text:
        return False
    
    # Check reasonable length (not too short, not too long)
    if len(email_text) < 10 or len(email_text) > 10000:
        return False
    
    # Check for basic email structure indicators
    # Should contain some text content
    if len(email_text.strip()) < 10:
        return False
    
    return True

