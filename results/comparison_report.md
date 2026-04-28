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

**Best Model (Validation F1):** Random Forest
**Best Model (Test F1):** KNN