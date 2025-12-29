# 📁 Folder Structure Explanation

This document explains the complete folder structure of the Phishing Detection System.

## Root Directory

```
Phishing Detection System/
├── app/                    # Main Flask application package
├── ml_training/            # Machine learning training scripts
├── models/                 # Saved ML models (created after training)
├── data/                   # Dataset storage (created after download)
├── run.py                  # Application entry point
├── test_system.py          # System test script
├── requirements.txt        # Python dependencies
├── README.md              # Main documentation
├── QUICKSTART.md          # Quick start guide
├── FOLDER_STRUCTURE.md     # This file
└── .gitignore             # Git ignore rules
```

## 📂 app/ - Flask Application

The main application package following MVC architecture.

### app/__init__.py
- **Purpose**: Flask application factory
- **Function**: Creates and configures the Flask app instance
- **Key Feature**: Uses factory pattern for better testability

### app/config.py
- **Purpose**: Configuration management
- **Contains**: 
  - Secret keys
  - Model paths
  - File upload limits
  - Other app settings

### app/routes.py
- **Purpose**: URL routing and request handling
- **Endpoints**:
  - `/` - Main page
  - `/predict` - Prediction API endpoint
  - `/health` - Health check endpoint
- **Security**: Input validation at route level

## 📂 app/controllers/ - Business Logic (MVC Controller)

### app/controllers/detection_controller.py
- **Purpose**: Business logic for phishing detection
- **Responsibilities**:
  - Load trained models
  - Coordinate feature extraction
  - Make predictions
  - Format results
- **Pattern**: Separates business logic from routes

## 📂 app/models/ - ML Models (MVC Model)

### app/models/feature_extractor.py
- **Purpose**: Feature extraction from URLs and emails
- **Features Extracted**: 30 features including:
  - URL structure (length, depth, special characters)
  - Domain characteristics (TLD, subdomains, IP addresses)
  - Security indicators (HTTPS, suspicious patterns)
  - Phishing keywords and entropy
- **Methods**:
  - `extract_url_features()` - Extract features from URLs
  - `extract_email_features()` - Extract features from emails
  - `_calculate_entropy()` - Calculate string entropy

## 📂 app/utils/ - Utility Functions

### app/utils/validators.py
- **Purpose**: Input validation
- **Functions**:
  - `validate_url()` - Validate URL format
  - `validate_email_text()` - Validate email text

### app/utils/sanitizers.py
- **Purpose**: Input sanitization for security
- **Functions**:
  - `sanitize_input()` - Sanitize user input to prevent XSS/injection

## 📂 app/templates/ - HTML Templates (MVC View)

### app/templates/index.html
- **Purpose**: Main web interface
- **Features**:
  - Tab-based navigation (URL/Email)
  - Input forms
  - Result display
  - Error handling UI

## 📂 app/static/ - Static Files

### app/static/css/style.css
- **Purpose**: Styling for web interface
- **Features**:
  - Modern gradient design
  - Responsive layout
  - Animations and transitions
  - No external frameworks (pure CSS)

### app/static/js/main.js
- **Purpose**: Client-side JavaScript
- **Features**:
  - Tab switching
  - API communication
  - Result display
  - Error handling

## 📂 ml_training/ - ML Training Scripts

### ml_training/train_models.py
- **Purpose**: Train and compare ML models
- **Functions**:
  - Download/create dataset
  - Extract features
  - Train 3 models (Logistic Regression, Random Forest, XGBoost)
  - Evaluate and compare models
  - Save best model
- **Output**: 
  - `models/best_model.pkl` - Best trained model
  - `models/scaler.pkl` - Feature scaler

## 📂 models/ - Saved Models (Created After Training)

This directory is created automatically when you run the training script.

- **models/best_model.pkl**: Best performing ML model (joblib format)
- **models/scaler.pkl**: StandardScaler used for feature normalization

## 📂 data/ - Dataset Storage (Created After Training)

This directory is created automatically when you run the training script.

- **data/phishing_dataset.csv**: Phishing dataset (URLs and labels)

## 🔄 Data Flow

```
User Input (URL/Email)
    ↓
routes.py (Validation & Sanitization)
    ↓
detection_controller.py (Business Logic)
    ↓
feature_extractor.py (Feature Extraction)
    ↓
Trained Model (Prediction)
    ↓
detection_controller.py (Format Result)
    ↓
routes.py (JSON Response)
    ↓
Frontend (Display Result)
```

## 🏗️ Architecture Pattern: MVC

### Model (app/models/)
- Feature extraction logic
- Data processing

### View (app/templates/, app/static/)
- HTML templates
- CSS styling
- JavaScript interactions

### Controller (app/controllers/)
- Business logic
- Model coordination
- Result formatting

### Routes (app/routes.py)
- HTTP request handling
- Input validation
- Response formatting

## 🔒 Security Layers

1. **Route Level** (`app/routes.py`): Input type validation
2. **Validator** (`app/utils/validators.py`): Format validation
3. **Sanitizer** (`app/utils/sanitizers.py`): XSS/injection prevention
4. **Controller** (`app/controllers/detection_controller.py`): Safe processing

## 📊 File Sizes (Approximate)

- **Code Files**: ~15-20 KB total
- **Templates**: ~3 KB
- **Static Files**: ~8 KB
- **Models** (after training): ~500 KB - 2 MB
- **Dataset**: ~100 KB - 10 MB (varies)

## 🎯 Key Design Decisions

1. **MVC Separation**: Clear separation of concerns
2. **Factory Pattern**: Flask app creation
3. **Singleton Pattern**: Controller instance
4. **Feature Extraction**: Reusable across training and prediction
5. **Security First**: Validation and sanitization at multiple layers

## 📝 Notes

- All `__init__.py` files make directories Python packages
- Models directory is gitignored (generated files)
- Data directory is gitignored (dataset files)
- Virtual environment should be in `.gitignore`

