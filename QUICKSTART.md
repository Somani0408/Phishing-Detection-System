# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Models
```bash
python ml_training/train_models.py
```

This will:
- Download/create the dataset
- Extract features
- Train 3 ML models
- Save the best model

### Step 3: Run the Web App
```bash
python run.py
```

Then open: **http://localhost:5000**

## 📝 Testing the System

### Test with Legitimate URL
```
https://www.github.com
```

### Test with Phishing URL
```
http://verify-account-security.tk/login
```

### Test with Email Text
```
Paste email content with suspicious links or phishing keywords
```

## ⚠️ Troubleshooting

**Error: Model files not found**
- Solution: Run `python ml_training/train_models.py` first

**Error: Port already in use**
- Solution: Change port in `run.py` (line 10)

**Error: Import errors**
- Solution: Activate virtual environment and install requirements

## 📚 For More Details
See [README.md](README.md) for complete documentation.

