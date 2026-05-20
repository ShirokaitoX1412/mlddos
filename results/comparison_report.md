## Model Comparison Results

### Validation Set Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest **[BEST]** | 0.993625 | 0.993587 | 0.993625 | 0.993578 |
| KNN | 0.992256 | 0.992184 | 0.992256 | 0.992200 |
| Extra Trees | 0.991914 | 0.992165 | 0.991914 | 0.991793 |
| MLP Classifier | 0.988449 | 0.988344 | 0.988449 | 0.988331 |
| XGBoost | 0.992256 | 0.992304 | 0.992256 | 0.992277 |

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.740124 | 0.814562 | 0.740124 | 0.646413 |
| KNN **[BEST]** | 0.745915 | 0.901966 | 0.745915 | 0.712075 |
| Extra Trees | 0.737616 | 0.809255 | 0.737616 | 0.642486 |
| MLP Classifier | 0.746406 | 0.686445 | 0.746406 | 0.708714 |
| XGBoost | 0.749405 | 0.868290 | 0.749405 | 0.709566 |

**Best Model (Validation F1):** Random Forest
**Best Model (Test F1):** KNN