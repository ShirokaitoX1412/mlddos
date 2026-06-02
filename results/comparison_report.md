## Robust Model Comparison

### Cross-Validation Performance
| Model | CV Accuracy Mean | CV Accuracy Std | CV F1-Score Mean | CV F1-Score Std | CV F1-Macro Mean |
| --- | --- | --- | --- | --- | --- |
| Extra Trees | 0.993095 | 0.000255 | 0.993028 | 0.000304 | 0.878783 |
| XGBoost | 0.991358 | 0.000232 | 0.991640 | 0.000233 | 0.879868 |
| KNN | 0.991538 | 0.000095 | 0.991504 | 0.000145 | 0.879816 |
| Random Forest | 0.992907 | 0.000087 | 0.992730 | 0.000091 | 0.841814 |
| MLP Classifier | 0.988363 | 0.000409 | 0.988190 | 0.000407 | 0.820987 |

### Held-Out Test Performance
| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test F1-Macro | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Extra Trees | 0.739685 | 0.812233 | 0.739685 | 0.645347 | 0.615672 | 0.866525 |
| XGBoost | 0.737358 | 0.683817 | 0.737358 | 0.648907 | 0.638285 | 0.932562 |
| KNN | 0.746510 | 0.922959 | 0.746510 | 0.712416 | 0.668477 | 0.865474 |
| Random Forest | 0.739814 | 0.811988 | 0.739814 | 0.644868 | 0.615645 | 0.857130 |
| MLP Classifier | 0.746484 | 0.688213 | 0.746484 | 0.709449 | 0.659968 | 0.855695 |

Selection prioritizes stable cross-validation F1, macro F1, and low overfitting risk.