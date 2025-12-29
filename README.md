# 🛡️ Phishing Detection System

A complete Machine Learning-based Phishing Detection System built with Python, Flask, and advanced ML algorithms. This system can detect phishing attempts in URLs and email content using trained machine learning models.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Security Features](#security-features)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multi-Model Support**: Trains and compares 3 ML models (Logistic Regression, Random Forest, XGBoost)
- **Dual Input Types**: Supports both URL and email text analysis
- **Real-time Detection**: Fast prediction with confidence scores
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- **Secure Web Interface**: Input validation and sanitization
- **MVC Architecture**: Clean, maintainable code structure
- **Production Ready**: Error handling, logging, and security best practices

## 📁 Project Structure

```
Phishing Detection System/
│
├── app/                          # Flask application
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # Configuration settings
│   ├── routes.py                # URL routes and endpoints
│   │
│   ├── controllers/             # Business logic (MVC)
│   │   ├── __init__.py
│   │   └── detection_controller.py
│   │
│   ├── models/                  # ML models and feature extraction
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── validators.py        # Input validation
│   │   └── sanitizers.py        # Input sanitization
│   │
│   ├── templates/               # HTML templates
│   │   └── index.html
│   │
│   └── static/                  # Static files
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── main.js
│
├── ml_training/                  # ML training scripts
│   ├── __init__.py
│   └── train_models.py          # Model training script
│
├── models/                      # Saved ML models (created after training)
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── data/                        # Dataset storage (created after download)
│   └── phishing_dataset.csv
│
├── run.py                       # Application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Architecture Explanation

The project follows **MVC (Model-View-Controller)** architecture:

- **Model** (`app/models/`): Feature extraction logic and ML model definitions
- **View** (`app/templates/`, `app/static/`): HTML templates and frontend assets
- **Controller** (`app/controllers/`): Business logic that connects models and views

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
cd "C:\Project\Phishing Detection System"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Step 1: Train the Models

Before running the web application, you need to train the ML models:

```bash
python ml_training/train_models.py
```

This script will:
1. Download or create a phishing dataset
2. Extract 30 features from each URL
3. Train 3 ML models (Logistic Regression, Random Forest, XGBoost)
4. Compare models and select the best one
5. Save the best model and scaler to `models/` directory

**Expected Output:**
```
PHISHING DETECTION MODEL TRAINING
============================================================

Downloading Phishing Dataset...
Dataset loaded: 100 samples
  Columns: ['url', 'label']
  Label distribution:
    Legitimate (0): 50
    Phishing (1): 50

Extracting features from URLs...
Feature extraction complete!

Feature matrix shape: (100, 30)
  Number of samples: 100
  Number of features: 30

Splitting dataset into train/test sets...
  Training set: 80 samples
  Test set: 20 samples

Training Logistic Regression...
Training Random Forest...
Training XGBoost...

MODEL COMPARISON SUMMARY
============================================================
Best Model: XGBoost
  F1-Score: 0.9500

Best model saved to models/best_model.pkl
Training complete!
```

### Step 2: Run the Web Application

```bash
python run.py
```

The application will start on `http://localhost:5000`

### Step 3: Use the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Choose between **URL Detection** or **Email Detection** tab
3. Enter a URL or email text
4. Click "Analyze" to get the prediction
5. View the result with confidence score

## 🔬 Model Training

### Dataset

The training script automatically:
- Downloads a real phishing dataset from GitHub (if available)
- Creates a synthetic dataset if download fails (for demonstration)
- Extracts 30 features from each URL

### Features Extracted

The system extracts 30 features including:
- URL length and structure
- Domain characteristics (dots, hyphens, subdomains)
- Security indicators (HTTPS, IP addresses)
- Suspicious patterns (short URLs, suspicious TLDs)
- Phishing keywords
- Entropy and randomness measures

### Models Trained

1. **Logistic Regression**: Fast, interpretable baseline model
2. **Random Forest**: Ensemble method with good generalization
3. **XGBoost**: Gradient boosting with high performance

The best model (based on F1-score) is automatically selected and saved.

### Evaluation Metrics

Each model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## 🔒 Security Features

### Input Validation

- **URL Validation**: Checks URL format and length
- **Email Validation**: Validates email text structure
- **Length Limits**: Prevents DoS attacks with oversized inputs

### Input Sanitization

- **HTML Escaping**: Prevents XSS attacks
- **Control Character Removal**: Removes dangerous characters
- **Null Byte Removal**: Prevents injection attacks

### Security Best Practices

- Input validation at route level
- Sanitization before processing
- Error handling without information leakage
- Secure session management

## 📸 Screenshots

### Main Interface
The web interface features a clean, modern design with:
- Tab-based navigation (URL/Email)
- Real-time analysis
- Visual confidence indicators
- Responsive design

### Results Display
Results show:
- Clear "Phishing" or "Legitimate" label
- Confidence percentage
- Visual confidence bar
- Input details

## 🛠️ Technologies Used

### Backend
- **Flask 3.0**: Web framework
- **scikit-learn 1.3**: Machine learning library
- **XGBoost 2.0**: Gradient boosting framework
- **pandas 2.1**: Data manipulation
- **numpy 1.26**: Numerical computing
- **joblib 1.3**: Model serialization

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling (no frameworks)
- **JavaScript**: Client-side logic

### Security
- Input validation
- HTML sanitization
- Error handling

## 📊 Example Predictions

### Legitimate URL
```
Input: https://www.github.com
Result: ✅ LEGITIMATE
Confidence: 92.5%
```

### Phishing URL
```
Input: http://verify-account-security.tk/login
Result: ⚠️ PHISHING DETECTED
Confidence: 87.3%
```

## 🔧 Configuration

Edit `app/config.py` to customize:
- Secret key (change in production!)
- Model paths
- File upload limits
- Other settings

## 🐛 Troubleshooting

### Model Not Found Error
If you see "Model files not found":
1. Run `python ml_training/train_models.py` first
2. Ensure `models/best_model.pkl` and `models/scaler.pkl` exist

### Import Errors
If you see import errors:
1. Activate your virtual environment
2. Run `pip install -r requirements.txt`
3. Check Python version (3.8+)

### Port Already in Use
If port 5000 is busy:
1. Edit `run.py`
2. Change `port=5000` to another port (e.g., `port=5001`)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Built by a Senior Machine Learning + Cybersecurity Engineer

## 📚 Additional Resources

- [Phishing Detection Research](https://www.usenix.org/conference/usenixsecurity21)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Note**: This system is for educational and demonstration purposes. For production use, ensure:
- Use a larger, real-world dataset
- Implement proper logging
- Add rate limiting
- Use a production WSGI server (Gunicorn, uWSGI)
- Set up proper security headers
- Use environment variables for secrets

