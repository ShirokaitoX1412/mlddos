# DDoS Detection & IPS using Machine Learning

## Overview

Enhanced DDoS attack detection, classification, and **real-time intrusion prevention** system
using the **CICDDoS2019** dataset. This project implements a multi-class ML pipeline with
5 models, SHAP explainability, a live Scapy-based IPS engine, and a professional Streamlit dashboard.

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

## System Architecture

```
ML_DDOS/
├── data_loader.py        # Dataset download from Kaggle + loading
├── preprocessor.py       # Cleaning, feature engineering, scaling
├── models.py             # 5 classifier definitions + training + evaluation
├── shap_explainer.py     # SHAP-based Explainable AI (XAI)
├── live_ips.py           # Real-time IPS engine (Scapy sniffer)
├── mitigation.py         # OS-level firewall commands (iptables/netsh)
├── app.py                # Streamlit dashboard (dark-mode UI)
├── main.py               # Pipeline orchestrator
├── Dockerfile            # Container deployment
├── results/              # Generated plots, reports, SHAP explanations
├── saved_models/         # Trained model files (.pkl)
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

## Features

### Explainable AI (SHAP)
- SHAP summary plots showing which features drive attack classification
- Per-class beeswarm plots explaining feature impact for each DDoS type
- Force plots for individual prediction explanations

### Real-Time IPS Engine (`live_ips.py`)
- **Scapy-based** live packet sniffing and flow aggregation
- Classifies network flows using the trained ML model
- **Auto-mitigation**: blocks attacker IPs when attack confidence > 95%
  - Linux: `iptables -A INPUT -s <IP> -j DROP`
  - Windows: `netsh advfirewall firewall add rule ...`
- **SIMULATION_MODE** (default: `True`) — prints commands without executing

### Professional Dashboard (`app.py`)
- **Dark-mode, high-tech UI** built with Streamlit
- **Analytics Tab**: Model comparison, ROC curves, confusion matrices, SHAP plots
- **Live Monitor Tab**: Real-time traffic log with RED attack alerts and block status
- **Telegram Integration**: Instant attack notifications via Telegram bot

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
# 1. Train models & generate all reports
python main.py

# 2. Generate SHAP explanations
python shap_explainer.py

# 3. Launch the Dashboard
streamlit run app.py

# 4. Start IPS in simulation mode (safe)
sudo python live_ips.py

# 5. Start IPS in LIVE mode (actually blocks IPs!)
sudo python live_ips.py --live

# 6. Run only preprocessing (Steps 1 & 2)
python main.py --steps-1-2
```

### Docker Deployment

```bash
docker build -t ddos-ips .
docker run -p 8501:8501 ddos-ips
```

## Generated Reports & Visualizations

All output files are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `*_confusion_matrix.png` | Confusion matrix heatmap for each model |
| `*_roc_curve.png` | Per-class ROC curves (One-vs-Rest) for each model |
| `*_feature_importance.png` | Top-20 feature importance (RF, Extra Trees, XGBoost) |
| `*_shap_summary_bar.png` | SHAP global feature importance |
| `*_shap_<class>.png` | SHAP beeswarm per attack class |
| `*_shap_force_plot.png` | SHAP force plot for single prediction |
| `model_comparison.png` | Grouped bar chart comparing all models |
| `*_classification_report.txt` | Detailed per-class precision/recall/F1 |
| `validation_scores.csv` / `test_scores.csv` | Metrics in CSV format |

Trained models are saved in `saved_models/` as `.pkl` files.

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
- SHAP explainability plots
