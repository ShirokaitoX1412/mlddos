## Robust Model Comparison

### Cross-Validation Performance
| Model | CV Accuracy Mean | CV Accuracy Std | CV F1-Score Mean | CV F1-Score Std | CV F1-Macro Mean |
| --- | --- | --- | --- | --- | --- |
| MLP Classifier | 0.970200 | 0.000705 | 0.969594 | 0.000670 | 0.937653 |
| XGBoost | 0.968700 | 0.001087 | 0.968319 | 0.001098 | 0.927153 |
| Extra Trees + SMOTE | 0.958725 | 0.001405 | 0.958746 | 0.001282 | 0.902982 |
| Extra Trees | 0.957363 | 0.000692 | 0.957445 | 0.000624 | 0.900505 |
| Random Forest | 0.956787 | 0.000726 | 0.957496 | 0.000644 | 0.885682 |
| Random Forest + SMOTE | 0.957987 | 0.002526 | 0.958561 | 0.002123 | 0.893230 |
| KNN | 0.952450 | 0.002332 | 0.951494 | 0.002276 | 0.921287 |

### Held-Out Test Performance
| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test F1-Macro | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| MLP Classifier | 0.974547 | 0.974909 | 0.974547 | 0.974019 | 0.937719 | 0.996866 |
| XGBoost | 0.972986 | 0.974251 | 0.972986 | 0.972693 | 0.923634 | 0.997275 |
| Extra Trees + SMOTE | 0.964379 | 0.965903 | 0.964379 | 0.964531 | 0.898869 | 0.995924 |
| Extra Trees | 0.963648 | 0.965293 | 0.963648 | 0.963884 | 0.897152 | 0.995952 |
| Random Forest | 0.961588 | 0.966048 | 0.961588 | 0.962870 | 0.869480 | 0.995444 |
| Random Forest + SMOTE | 0.966971 | 0.968483 | 0.966971 | 0.967105 | 0.902051 | 0.995080 |
| KNN | 0.952085 | 0.955575 | 0.952085 | 0.953426 | 0.918547 | 0.989940 |

Selection prioritizes stable cross-validation F1, macro F1, and low overfitting risk.