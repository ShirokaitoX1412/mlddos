# DDoS Detection using Machine Learning

## Overview

Enhanced DDoS attack detection and classification system using the **CICDDoS2019** dataset.
This project implements a multi-class classification pipeline that identifies various DDoS attack
types from network traffic using five machine learning models.

**Reference**: Based on the methodology from
[Kaggle: DDoS Detection using Machine Learning](https://www.kaggle.com/code/rakibhossainsajib/ddos-detection-using-machine-learning)

## Models

| # | Model | Description | Hyperparameters |
|---|-------|-------------|-----------------|
| 1 | **Random Forest** | Ensemble of decision trees with bagging | n_estimators=200, max_depth=25, min_samples_split=5 |
| 2 | **KNN** | K-Nearest Neighbors (distance-weighted) | n_neighbors=7, weights=distance, metric=minkowski |
| 3 | **Extra Trees** | Extremely Randomized Trees | n_estimators=200, max_depth=25, min_samples_split=5 |
| 4 | **MLP Classifier** | Multi-Layer Perceptron neural network | hidden_layers=(128,64), adam, early_stopping |
| 5 | **XGBoost** | Gradient boosted decision trees | n_estimators=200, max_depth=10, lr=0.1, subsample=0.8 |

## DDoS Attack Types (Multi-class, 7 Classes)

| Class | Description |
|-------|-------------|
| **TCP SYN Flood** | SYN packet flooding attack |
| **UDP Flood** | Generic UDP flooding attack |
| **UDP-Lag Flood** | UDP-based with induced lag |
| **LDAP Flood** | LDAP amplification attack |
| **MSSQL Flood** | MSSQL-targeted attack |
| **NetBIOS Flood** | NetBIOS-related attack |
| **Benign** | Normal, non-attack traffic |

## Model Comparison Results

### Validation Set Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest **[BEST]** | 0.993198 | 0.993193 | 0.993198 | 0.993107 |
| KNN | 0.992171 | 0.992094 | 0.992171 | 0.992113 |
| Extra Trees | 0.991743 | 0.992021 | 0.991743 | 0.991625 |
| MLP Classifier | 0.989604 | 0.989367 | 0.989604 | 0.989448 |
| XGBoost | 0.992042 | 0.992076 | 0.992042 | 0.992057 |

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.739866 | 0.812034 | 0.739866 | 0.644947 |
| KNN **[BEST]** | 0.745760 | 0.922684 | 0.745760 | 0.711882 |
| Extra Trees | 0.737590 | 0.811133 | 0.737590 | 0.642605 |
| MLP Classifier | 0.747285 | 0.687910 | 0.747285 | 0.709916 |
| XGBoost | 0.745346 | 0.820256 | 0.745346 | 0.682224 |

**Best Model (Validation F1):** Random Forest (F1 = 0.9931)
**Best Model (Test F1):** KNN (F1 = 0.7119)

> **Note on Test Set Performance**: The test set has a significantly different class distribution
> compared to the training set (e.g., UDP-Lag Flood is 0.05% of training but 22.94% of test),
> which explains the performance gap. This reflects a realistic scenario where attack distributions
> shift over time. The validation set (same distribution as training) shows all models achieve >99% F1.

## Generated Reports & Visualizations

All output files are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `*_confusion_matrix.png` | Confusion matrix heatmap for each model |
| `*_roc_curve.png` | Per-class ROC curves (One-vs-Rest) for each model |
| `*_feature_importance.png` | Top-20 feature importance (RF, Extra Trees, XGBoost) |
| `model_comparison.png` | Grouped bar chart comparing all models |
| `*_classification_report.txt` | Detailed per-class precision/recall/F1 |
| `validation_scores.csv` / `test_scores.csv` | Metrics in CSV format |

Trained models are saved in `saved_models/` as `.pkl` files.

## Project Structure

```
ML_DDOS/
├── data_loader.py      # Dataset download and loading
├── preprocessor.py     # Cleaning, feature engineering, scaling
├── models.py           # 5 classifier definitions + training + evaluation
├── main.py             # Pipeline orchestrator
├── results/            # Generated plots, reports, and metrics
├── saved_models/       # Trained model files (.pkl)
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
# Run full pipeline (all 4 steps)
python main.py

# Run only data loading + preprocessing (Steps 1 & 2)
python main.py --steps-1-2
```

## Dataset

**CICDDoS2019** from the Canadian Institute for Cybersecurity.
- Source: https://www.kaggle.com/datasets/dhoogla/cicddos2019
- Format: Parquet files (training and testing splits per attack type)
- Raw features: 78 network traffic features
- After preprocessing: 32 features (removed 12 single-value + 33 highly correlated)

## Preprocessing Pipeline

1. **Label Harmonization** — Align label names between train/test sets
2. **Duplicate Removal** — 3,195 duplicates removed from training
3. **Invalid Value Handling** — Replace inf/NaN with column medians
4. **Single-value Column Removal** — 12 columns with constant values
5. **High-correlation Removal** — 33 columns with correlation > 0.8
6. **Label Encoding** — LabelEncoder for 7 target classes
7. **Feature Scaling** — MinMaxScaler fitted on training data
8. **Stratified Split** — 80/20 train/validation split

## Evaluation Metrics

For each model:
- Accuracy, Precision, Recall, F1-Score (weighted average)
- Confusion Matrix heatmap
- ROC Curve per class (One-vs-Rest)
- Feature Importance plot (tree-based models)
