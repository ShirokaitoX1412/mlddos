# Generalization Report

Selected model: **Extra Trees**

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

- Extra Trees: train-CV gap=0.0016, CV-test gap=0.3477. Likely causes include duplicate/near-duplicate flows, class distribution shift, or attack-source/domain shift between train and test.
- XGBoost: train-CV gap=0.0009, CV-test gap=0.3427. Likely causes include duplicate/near-duplicate flows, class distribution shift, or attack-source/domain shift between train and test.
- KNN: train-CV gap=0.0049, CV-test gap=0.2791. Likely causes include duplicate/near-duplicate flows, class distribution shift, or attack-source/domain shift between train and test.
- Random Forest: train-CV gap=0.0013, CV-test gap=0.3479. Likely causes include duplicate/near-duplicate flows, class distribution shift, or attack-source/domain shift between train and test.
- MLP Classifier: train-CV gap=0.0007, CV-test gap=0.2787. Likely causes include duplicate/near-duplicate flows, class distribution shift, or attack-source/domain shift between train and test.

## Summary Table

| Model | Train Accuracy | Train F1-Score | CV Accuracy Mean | CV Accuracy Std | CV F1-Score Mean | CV F1-Score Std | CV F1-Macro Mean | CV ROC-AUC Mean | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test F1-Macro | Test ROC-AUC | Overfit Gap F1 | Generalization Gap F1 | Selection Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Extra Trees | 0.994618 | 0.994598 | 0.993095 | 0.000255 | 0.993028 | 0.000304 | 0.878783 | 0.999858 | 0.739685 | 0.812233 | 0.739685 | 0.645347 | 0.615672 | 0.866525 | 0.001570 | 0.347681 | 1.122971 |
| XGBoost | 0.992119 | 0.992518 | 0.991358 | 0.000232 | 0.991640 | 0.000233 | 0.879868 | 0.999876 | 0.737358 | 0.683817 | 0.737358 | 0.648907 | 0.638285 | 0.932562 | 0.000878 | 0.342733 | 1.122509 |
| KNN | 0.996432 | 0.996428 | 0.991538 | 0.000095 | 0.991504 | 0.000145 | 0.879816 | 0.998060 | 0.746510 | 0.922959 | 0.746510 | 0.712416 | 0.668477 | 0.865474 | 0.004924 | 0.279088 | 1.118407 |
| Random Forest | 0.994105 | 0.993983 | 0.992907 | 0.000087 | 0.992730 | 0.000091 | 0.841814 | 0.999869 | 0.739814 | 0.811988 | 0.739814 | 0.644868 | 0.615645 | 0.857130 | 0.001253 | 0.347861 | 1.117657 |
| MLP Classifier | 0.989082 | 0.988906 | 0.988363 | 0.000409 | 0.988190 | 0.000407 | 0.820987 | 0.999452 | 0.746484 | 0.688213 | 0.746484 | 0.709449 | 0.659968 | 0.855695 | 0.000716 | 0.278741 | 1.110214 |