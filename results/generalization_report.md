# Generalization Report

Selected model: **MLP Classifier**

Selection is based on cross-validation weighted F1, macro F1, fold stability, and overfit penalty. The held-out test set is reported as an external generalization check, not as the sole selector.

## Why the Original Pipeline Likely Overfit

- A single validation split can be too similar to the training data and hide distribution shift.
- Precomputing preprocessing outside cross-validation risks fitting medians, scalers, or feature filters on validation data.
- Weighted metrics can hide poor recall for minority DDoS classes.
- Random row splitting can leak near-duplicate flows across train and validation.

## Mitigations in This Pipeline

- All preprocessing is inside sklearn/imblearn Pipelines and is refit per CV fold.
- StratifiedKFold is used by default; StratifiedGroupKFold is used when session/IP/source grouping columns exist.
- Class imbalance is detected and handled with class weights, balanced sample weights, and optional SMOTE variants.
- Model selection penalizes high train-CV gaps and unstable fold performance.
- Potential identifier/leakage columns and highly correlated features are removed using training-fold statistics only.

## Overfitting Warnings

No large F1 gaps exceeded the configured threshold.


## Summary Table

| Model | Train Accuracy | Train F1-Score | CV Accuracy Mean | CV Accuracy Std | CV F1-Score Mean | CV F1-Score Std | CV F1-Macro Mean | CV ROC-AUC Mean | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test F1-Macro | Test ROC-AUC | Overfit Gap F1 | Generalization Gap F1 | Selection Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MLP Classifier | 0.970700 | 0.970125 | 0.970200 | 0.000705 | 0.969594 | 0.000670 | 0.937653 | 0.996555 | 0.974547 | 0.974909 | 0.974547 | 0.974019 | 0.937719 | 0.996866 | 0.000531 | -0.004425 | 1.295371 |
| XGBoost | 0.968900 | 0.968502 | 0.968700 | 0.001087 | 0.968319 | 0.001098 | 0.927153 | 0.997303 | 0.972986 | 0.974251 | 0.972986 | 0.972693 | 0.923634 | 0.997275 | 0.000183 | -0.004374 | 1.290260 |
| Extra Trees + SMOTE | 0.959113 | 0.959061 | 0.958725 | 0.001405 | 0.958746 | 0.001282 | 0.902982 | 0.995825 | 0.964379 | 0.965903 | 0.964379 | 0.964531 | 0.898869 | 0.995924 | 0.000316 | -0.005786 | 1.271593 |
| Extra Trees | 0.958037 | 0.958184 | 0.957363 | 0.000692 | 0.957445 | 0.000624 | 0.900505 | 0.995630 | 0.963648 | 0.965293 | 0.963648 | 0.963884 | 0.897152 | 0.995952 | 0.000739 | -0.006439 | 1.269896 |
| Random Forest | 0.956925 | 0.957610 | 0.956787 | 0.000726 | 0.957496 | 0.000644 | 0.885682 | 0.995160 | 0.961588 | 0.966048 | 0.961588 | 0.962870 | 0.869480 | 0.995444 | 0.000114 | -0.005374 | 1.265967 |
| Random Forest + SMOTE | 0.960838 | 0.960869 | 0.957987 | 0.002526 | 0.958561 | 0.002123 | 0.893230 | 0.994805 | 0.966971 | 0.968483 | 0.966971 | 0.967105 | 0.902051 | 0.995080 | 0.002308 | -0.008543 | 1.262331 |
| KNN | 0.992725 | 0.992690 | 0.952450 | 0.002332 | 0.951494 | 0.002276 | 0.921287 | 0.990519 | 0.952085 | 0.955575 | 0.952085 | 0.953426 | 0.918547 | 0.989940 | 0.041196 | -0.001932 | 1.187000 |