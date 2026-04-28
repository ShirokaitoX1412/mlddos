# DDoS Detection using Machine Learning

## Overview

Enhanced DDoS attack detection and classification system using the **CICDDoS2019** dataset.
This project implements a multi-class classification pipeline that identifies various DDoS attack
types from network traffic using five machine learning models.

**Reference**: Based on the methodology from
[Kaggle: DDoS Detection using Machine Learning](https://www.kaggle.com/code/rakibhossainsajib/ddos-detection-using-machine-learning)

## Models

| # | Model | Description |
|---|-------|-------------|
| 1 | **Random Forest** | Ensemble of decision trees with bagging |
| 2 | **KNN** | K-Nearest Neighbors (k=10) |
| 3 | **Extra Trees** | Extremely Randomized Trees |
| 4 | **MLP Classifier** | Multi-Layer Perceptron neural network |
| 5 | **XGBoost** | Gradient boosted decision trees |

## DDoS Attack Types (Multi-class)

The dataset contains these attack categories:
- **TCP SYN Flood** - SYN packet flooding attack
- **UDP Flood** - Generic UDP flooding attack
- **UDP-Lag Flood** - UDP-based with induced lag
- **LDAP Flood** - LDAP amplification attack
- **MSSQL Flood** - MSSQL-targeted attack
- **NetBIOS Flood** - NetBIOS-related attack
- **Portmap Flood** - Portmapper-based attack
- **Benign** - Normal, non-attack traffic

## Project Structure

```
ML_DDOS/
├── data_loader.py      # Dataset download and loading
├── preprocessor.py     # Cleaning, feature engineering, scaling
├── models.py           # 5 classifier definitions
├── main.py             # Pipeline orchestrator
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API (for dataset download)
# Place your kaggle.json in ~/.kaggle/
```

## Usage

```bash
# Run Steps 1 & 2: Data loading + Preprocessing
python main.py

# Run full pipeline (when Step 3 is implemented)
python main.py --full
```

## Dataset

**CICDDoS2019** from the Canadian Institute for Cybersecurity.
- Source: https://www.kaggle.com/datasets/dhoogla/cicddos2019
- Format: Parquet files (training and testing splits per attack type)
- Features: 78 network traffic features

## Evaluation Metrics

For each model:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve (per-class)
- Feature Importance plot
