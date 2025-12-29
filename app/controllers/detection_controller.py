"""
Controller for handling phishing detection logic
Separates business logic from routes (MVC pattern)
"""
import os
import joblib
import numpy as np
from app.models.feature_extractor import FeatureExtractor
from app.config import Config

class DetectionController:
    """Controller class for phishing detection operations"""
    
    def __init__(self):
        """Initialize the controller and load models"""
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self._load_models()
    
    def _load_models(self):
        """
        Load the trained model and scaler from disk
        Raises error if models are not found
        """
        try:
            model_path = Config.MODEL_PATH
            scaler_path = Config.SCALER_PATH
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(
                    f"Model files not found. Please train the model first.\n"
                    f"Expected files:\n"
                    f"  - {model_path}\n"
                    f"  - {scaler_path}\n"
                    f"Run: python ml_training/train_models.py"
                )
            
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
        except FileNotFoundError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")
    
    def predict(self, input_value, input_type='url'):
        """
        Predict if input is phishing or legitimate
        
        Args:
            input_value: URL or email text to analyze
            input_type: 'url' or 'email'
        
        Returns:
            dict: Prediction result with label, confidence, and metadata
        """
        try:
            # Extract features based on input type
            if input_type == 'url':
                features = self.feature_extractor.extract_url_features(input_value)
            else:  # email
                features = self.feature_extractor.extract_email_features(input_value)
            
            # Convert to numpy array and reshape for single prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features using the same scaler used during training
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get prediction probability for confidence score
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = float(max(probabilities)) * 100
            else:
                # For models without predict_proba, use decision function if available
                if hasattr(self.model, 'decision_function'):
                    decision = self.model.decision_function(features_scaled)[0]
                    # Normalize decision score to 0-100 range
                    confidence = min(max(float(abs(decision)) * 10, 0), 100)
                else:
                    confidence = 85.0  # Default confidence if neither available
            
            # Map prediction (1 = phishing, 0 = legitimate)
            label = "Phishing" if prediction == 1 else "Legitimate"
            
            return {
                'status': 'success',
                'label': label,
                'confidence': round(confidence, 2),
                'input_type': input_type,
                'input_value': input_value[:100]  # Truncate for display
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")

