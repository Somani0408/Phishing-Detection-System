"""
Input sanitization utilities
Sanitizes user input to prevent injection attacks (XSS, SQL injection, etc.)
"""
import re
import html

def sanitize_input(input_value):
    """
    Sanitize user input to prevent XSS and injection attacks
    
    Args:
        input_value: Raw input string
    
    Returns:
        str: Sanitized input string
    """
    if not input_value:
        return ""
    
    # HTML escape to prevent XSS attacks
    sanitized = html.escape(input_value)
    
    # Remove null bytes (can be used in injection attacks)
    sanitized = sanitized.replace('\x00', '')
    
    # Remove control characters except newlines and tabs
    # This prevents various injection techniques
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    # Limit length to prevent DoS attacks
    max_length = 2048
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()

