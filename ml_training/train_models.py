"""
Machine Learning Model Training Script
Trains and compares multiple ML models for phishing detection
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import feature extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.feature_extractor import FeatureExtractor

def download_dataset():
    """
    Download or load phishing dataset
    Uses a publicly available phishing URL dataset
    """
    print("=" * 60)
    print("Downloading Phishing Dataset...")
    print("=" * 60)
    
    # Dataset URLs - using multiple sources for robustness
    dataset_urls = [
        "https://raw.githubusercontent.com/incertum/cyber-matrix-ai/master/Malicious-URL-Detection-Deep-Learning/data/url_data_mega_deep_learning.csv",
        "https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset"
    ]
    
    # Try to load from local file first
    local_path = "data/phishing_dataset.csv"
    if os.path.exists(local_path):
        print(f"Loading dataset from local file: {local_path}")
        try:
            df = pd.read_csv(local_path)
            if 'url' in df.columns and 'label' in df.columns:
                return df
        except:
            pass
    
    # Try to download from GitHub
    try:
        print("Attempting to download from GitHub...")
        df = pd.read_csv(dataset_urls[0])
        if 'url' in df.columns and 'label' in df.columns:
            # Save locally for future use
            os.makedirs('data', exist_ok=True)
            df.to_csv(local_path, index=False)
            print(f"Dataset downloaded and saved to {local_path}")
            return df
    except Exception as e:
        print(f"Failed to download from GitHub: {e}")
    
    # If download fails, create synthetic dataset for demonstration
    print("\nCreating synthetic dataset for demonstration...")
    return create_synthetic_dataset()

def create_synthetic_dataset():
    """
    Create a synthetic phishing dataset for demonstration
    In production, use a real dataset
    """
    print("Generating synthetic phishing and legitimate URLs...")
    
    # Legitimate URLs
    legitimate_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.wikipedia.org",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.linkedin.com",
        "https://www.youtube.com",
        "https://www.reddit.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.paypal.com",
        "https://www.ebay.com",
        "https://www.bbc.com",
        "https://www.cnn.com",
        "https://www.nytimes.com",
        "https://www.medium.com",
        "https://www.udemy.com",
        "https://www.coursera.org",
        "https://www.khanacademy.org",
        "https://www.codecademy.com",
        "https://www.freecodecamp.org",
        "https://www.python.org",
        "https://www.nodejs.org",
        "https://www.reactjs.org",
        "https://www.vuejs.org",
        "https://www.angular.io",
        "https://www.docker.com",
        "https://www.kubernetes.io",
        "https://www.aws.amazon.com",
        "https://www.cloud.google.com",
        "https://www.azure.microsoft.com",
        "https://www.digitalocean.com",
        "https://www.heroku.com",
        "https://www.netlify.com",
        "https://www.vercel.com",
        "https://www.mongodb.com",
        "https://www.postgresql.org",
        "https://www.mysql.com",
        "https://www.redis.io",
        "https://www.elastic.co",
        "https://www.kibana.org",
        "https://www.grafana.com",
        "https://www.prometheus.io",
        "https://www.jenkins.io",
        "https://www.gitlab.com"
    ]
    
    # Phishing URLs (synthetic examples)
    phishing_urls = [
        "http://verify-account-security.tk/login",
        "https://update-paypal-account.ga/confirm",
        "http://secure-banking-login.ml/verify",
        "https://amazon-account-suspended.cf/restore",
        "http://urgent-security-update.tk/click-here",
        "https://ebay-account-locked.gq/unlock",
        "http://facebook-security-alert.top/validate",
        "https://google-account-verify.xyz/confirm",
        "http://microsoft-security-update.click/update",
        "https://apple-id-verification.download/verify",
        "http://netflix-account-suspended.tk/restore",
        "https://spotify-premium-expired.ga/renew",
        "http://paypal-payment-failed.ml/resolve",
        "https://amazon-order-confirmation.cf/verify",
        "http://bank-security-alert.tk/login",
        "https://credit-card-verification.gq/confirm",
        "http://urgent-action-required.top/click",
        "https://account-compromised.xyz/secure",
        "http://password-reset-required.click/reset",
        "https://suspicious-activity-detected.download/verify",
        "http://verify-your-identity.tk/confirm",
        "https://update-payment-method.ga/update",
        "http://account-will-be-suspended.ml/prevent",
        "https://security-breach-detected.cf/secure",
        "http://immediate-action-required.tk/act-now",
        "https://phishing-example-1.gq/login",
        "http://malicious-url-1.top/verify",
        "https://suspicious-link-1.xyz/confirm",
        "http://fake-bank-login.click/secure",
        "https://scam-website-1.download/update",
        "http://verify-now-urgent.tk/immediate",
        "https://account-locked-security.ga/unlock",
        "http://payment-failed-update.ml/pay",
        "https://suspended-account-restore.cf/restore",
        "http://security-alert-verify.tk/verify",
        "https://phishing-attempt-1.gq/login",
        "http://malicious-site-1.top/click",
        "https://fake-login-page.xyz/enter",
        "http://scam-verification.click/verify",
        "https://suspicious-domain.download/update",
        "http://bit.ly/suspicious-link-1",
        "https://tinyurl.com/fake-bank-login",
        "http://goo.gl/malicious-redirect",
        "https://t.co/phishing-attempt",
        "http://ow.ly/scam-link",
        "https://is.gd/fake-verification",
        "http://short.link/malicious-url",
        "https://bit.ly/urgent-update-required",
        "http://tinyurl.com/account-suspended"
    ]
    
    # Create DataFrame
    data = {
        'url': legitimate_urls + phishing_urls,
        'label': [0] * len(legitimate_urls) + [1] * len(phishing_urls)
    }
    
    df = pd.DataFrame(data)
    
    # Save to local file
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/phishing_dataset.csv', index=False)
    print(f"Synthetic dataset created with {len(df)} samples")
    print(f"  - Legitimate URLs: {len(legitimate_urls)}")
    print(f"  - Phishing URLs: {len(phishing_urls)}")
    
    return df

def extract_features(df, feature_extractor):
    """
    Extract features from URLs in the dataset
    
    Args:
        df: DataFrame with 'url' column
        feature_extractor: FeatureExtractor instance
    
    Returns:
        numpy array: Feature matrix
    """
    print("\nExtracting features from URLs...")
    features_list = []
    
    for idx, url in enumerate(df['url']):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} URLs...")
        features = feature_extractor.extract_url_features(str(url))
        features_list.append(features)
    
    print("Feature extraction complete!")
    return np.array(features_list)

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train a model and evaluate its performance
    
    Args:
        model: Scikit-learn compatible model
        model_name: Name of the model
        X_train, X_test: Training and test feature matrices
        y_train, y_test: Training and test labels
    
    Returns:
        dict: Model performance metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  {'':<15} {'Predicted Legitimate':<25} {'Predicted Phishing':<25}")
    print(f"  {'Actual Legitimate':<15} {cm[0][0]:<25} {cm[0][1]:<25}")
    print(f"  {'Actual Phishing':<15} {cm[1][0]:<25} {cm[1][1]:<25}")
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def main():
    """Main training function"""
    print("\n" + "="*60)
    print("PHISHING DETECTION MODEL TRAINING")
    print("="*60 + "\n")
    
    # Step 1: Load dataset
    df = download_dataset()
    print(f"\nDataset loaded: {len(df)} samples")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Label distribution:")
    print(f"    Legitimate (0): {sum(df['label'] == 0)}")
    print(f"    Phishing (1):   {sum(df['label'] == 1)}")
    
    # Step 2: Extract features
    feature_extractor = FeatureExtractor()
    X = extract_features(df, feature_extractor)
    y = df['label'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  Number of samples: {X.shape[0]}")
    print(f"  Number of features: {X.shape[1]}")
    
    # Step 3: Split data
    print("\nSplitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    
    # Step 4: Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Step 5: Train and evaluate models
    models_to_train = [
        (LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression"),
        (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), "Random Forest"),
        (xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'), "XGBoost")
    ]
    
    results = []
    for model, name in models_to_train:
        result = train_and_evaluate_model(
            model, name, X_train_scaled, X_test_scaled, y_train, y_test
        )
        results.append(result)
    
    # Step 6: Select best model (based on F1-score)
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    best_model = max(results, key=lambda x: x['f1'])
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['name']:<20} {result['accuracy']:<12.4f} {result['precision']:<12.4f} "
              f"{result['recall']:<12.4f} {result['f1']:<12.4f}")
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model['name']}")
    print(f"  F1-Score: {best_model['f1']:.4f}")
    print(f"{'='*60}\n")
    
    # Step 7: Save best model
    model_path = 'models/best_model.pkl'
    joblib.dump(best_model['model'], model_path)
    print(f"Best model saved to {model_path}")
    print("\nTraining complete! You can now run the Flask application.")
    print("Run: python run.py")

if __name__ == "__main__":
    main()

