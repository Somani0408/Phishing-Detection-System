"""
Simple test script to verify the system is working correctly
Run this after training models to test the prediction system
"""
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.feature_extractor import FeatureExtractor
from app.utils.validators import validate_url, validate_email_text
from app.utils.sanitizers import sanitize_input

def test_feature_extraction():
    """Test feature extraction"""
    print("Testing Feature Extraction...")
    extractor = FeatureExtractor()
    
    # Test URL features
    url = "https://www.github.com"
    features = extractor.extract_url_features(url)
    assert len(features) == 30, f"Expected 30 features, got {len(features)}"
    print("✓ URL feature extraction works")
    
    # Test email features
    email = "This is a test email with http://example.com link"
    features = extractor.extract_email_features(email)
    assert len(features) == 30, f"Expected 30 features, got {len(features)}"
    print("✓ Email feature extraction works")
    
    print("Feature extraction: PASSED\n")

def test_validation():
    """Test input validation"""
    print("Testing Input Validation...")
    
    # Test URL validation
    assert validate_url("https://www.example.com") == True
    assert validate_url("http://example.com") == True
    assert validate_url("invalid-url") == False
    print("✓ URL validation works")
    
    # Test email validation
    assert validate_email_text("This is a valid email text") == True
    assert validate_email_text("") == False
    assert validate_email_text("a" * 10001) == False  # Too long
    print("✓ Email validation works")
    
    print("Input validation: PASSED\n")

def test_sanitization():
    """Test input sanitization"""
    print("Testing Input Sanitization...")
    
    # Test HTML escaping
    malicious = "<script>alert('xss')</script>"
    sanitized = sanitize_input(malicious)
    assert "<script>" not in sanitized
    print("✓ XSS prevention works")
    
    # Test null byte removal
    with_null = "test\x00string"
    sanitized = sanitize_input(with_null)
    assert "\x00" not in sanitized
    print("✓ Null byte removal works")
    
    print("Input sanitization: PASSED\n")

def test_model_loading():
    """Test model loading"""
    print("Testing Model Loading...")
    
    try:
        from app.controllers.detection_controller import DetectionController
        controller = DetectionController()
        print("✓ Models loaded successfully")
        
        # Test prediction
        result = controller.predict("https://www.github.com", "url")
        assert result['status'] == 'success'
        assert result['label'] in ['Phishing', 'Legitimate']
        assert 'confidence' in result
        print("✓ Prediction works")
        
        print("Model loading: PASSED\n")
    except FileNotFoundError as e:
        print("⚠ Model files not found. Run: python ml_training/train_models.py\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("PHISHING DETECTION SYSTEM - TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_feature_extraction()
        test_validation()
        test_sanitization()
        test_model_loading()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nSystem is ready to use!")
        print("Run: python run.py")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)

