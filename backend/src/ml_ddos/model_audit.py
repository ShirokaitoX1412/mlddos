"""
model_audit.py - Diagnostics for DDoS model evaluation quality.

Checks:
  - stratified split and class distribution drift
  - per-class precision/recall/F1, not only weighted averages
  - confusion matrices for train/validation/test
  - duplicate feature rows across splits
  - validation-test generalization gap
  - optional cross-validation on a leakage-safe sklearn pipeline

Usage:
    python model_audit.py --models random_forest knn
    python model_audit.py --models random_forest --cv-folds 5
"""

import argparse
import os
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MinMaxScaler

from .data_loader import load_dataset
from .models import get_classifiers
from .paths import AUDIT_DIR as PROJECT_AUDIT_DIR
from .preprocessor import (
    TARGET_COL,
    find_highly_correlated_columns,
    find_single_value_columns,
    harmonize_labels,
    preprocess,
    remove_duplicates,
)


AUDIT_DIR = str(PROJECT_AUDIT_DIR)

MODEL_ALIASES = {
    "random_forest": "Random Forest",
    "knn": "KNN",
    "extra_trees": "Extra Trees",
    "mlp_classifier": "MLP Classifier",
    "xgboost": "XGBoost",
}


class TrainOnlyFeatureDropper(BaseEstimator, TransformerMixin):
    """Drop constant and highly correlated columns based on each training fold."""

    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        self.drop_columns_: list[str] = []

    def fit(self, X, y=None):
        X_df = _as_dataframe(X)
        single_value = find_single_value_columns(X_df)
        X_reduced = X_df.drop(columns=single_value, errors="ignore")
        high_corr = find_highly_correlated_columns(
            X_reduced,
            threshold=self.correlation_threshold,
        )
        self.drop_columns_ = list(dict.fromkeys(single_value + high_corr))
        return self

    def transform(self, X):
        X_df = _as_dataframe(X)
        return X_df.drop(columns=self.drop_columns_, errors="ignore")


def _as_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def _replace_inf_with_nan(X):
    return np.where(np.isfinite(X), X, np.nan)


def _class_distribution(y: pd.Series, split_name: str) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    return pd.DataFrame({
        "split": split_name,
        "class": counts.index,
        "count": counts.values,
        "percent": (counts.values / len(y) * 100).round(4),
    })


def _duplicate_feature_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    left_hash = pd.util.hash_pandas_object(left, index=False)
    right_hash = pd.util.hash_pandas_object(right, index=False)
    return int(left_hash.isin(set(right_hash)).sum())


def _metrics_row(model_name: str, split_name: str, y_true, y_pred) -> dict:
    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _save_confusion_matrix(y_true, y_pred, class_names, model_name: str, split_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_name} - {split_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(
        AUDIT_DIR,
        f"{_safe_name(model_name)}_{split_name.lower()}_confusion_matrix.png",
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _write_classification_report(model_name: str, split_name: str,
                                 y_true, y_pred, class_names):
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    path = os.path.join(
        AUDIT_DIR,
        f"{_safe_name(model_name)}_{split_name.lower()}_per_class_report.txt",
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Per-class report: {model_name} - {split_name}\n")
        f.write("=" * 70 + "\n")
        f.write(report_txt)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    rows = []
    for class_name in class_names:
        rows.append({
            "model": model_name,
            "split": split_name,
            "class": class_name,
            "precision": report_dict[class_name]["precision"],
            "recall": report_dict[class_name]["recall"],
            "f1_score": report_dict[class_name]["f1-score"],
            "support": report_dict[class_name]["support"],
        })
    return rows


def _prepare_cv_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df, _ = harmonize_labels(train_df, test_df)
    train_df = remove_duplicates(train_df, "training set for CV")
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    le = LabelEncoder()
    return X, le.fit_transform(y), list(le.classes_)


def _run_cross_validation(model, X: pd.DataFrame, y: np.ndarray,
                          folds: int, correlation_threshold: float) -> dict:
    pipeline = Pipeline([
        ("inf_to_nan", FunctionTransformer(_replace_inf_with_nan, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_dropper", TrainOnlyFeatureDropper(correlation_threshold)),
        ("scaler", MinMaxScaler()),
        ("model", deepcopy(model)),
    ])
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "f1_macro": "f1_macro",
        },
        n_jobs=1,
        return_train_score=True,
    )
    return {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_f1_weighted_mean": scores["test_f1_weighted"].mean(),
        "cv_f1_weighted_std": scores["test_f1_weighted"].std(),
        "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "cv_train_f1_weighted_mean": scores["train_f1_weighted"].mean(),
    }


def _write_summary(distribution: pd.DataFrame, overlap_checks: dict,
                   gaps_df: pd.DataFrame, cv_df: pd.DataFrame | None):
    summary_path = os.path.join(AUDIT_DIR, "audit_summary.md")
    train_dist = distribution[distribution["split"] == "train"].set_index("class")
    test_dist = distribution[distribution["split"] == "test"].set_index("class")

    drift_rows = []
    for class_name in sorted(set(train_dist.index) | set(test_dist.index)):
        train_pct = train_dist["percent"].get(class_name, 0)
        test_pct = test_dist["percent"].get(class_name, 0)
        drift_rows.append((class_name, train_pct, test_pct, abs(train_pct - test_pct)))
    drift_rows = sorted(drift_rows, key=lambda row: row[3], reverse=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo kiểm tra đánh giá mô hình\n\n")
        f.write("## Kết luận nhanh\n\n")
        f.write("- Pipeline đã được chỉnh để chia stratified train/validation trước khi fit imputer, chọn đặc trưng và scaler.\n")
        f.write("- Train và validation gần nhau, nhưng test thấp hơn nhiều; dấu hiệu chính là phân phối lớp giữa train và test bị lệch mạnh.\n")
        f.write("- Nên đọc thêm `per_class_metrics.csv` vì weighted average che mất các lớp hiếm như `UDP-Lag Flood`.\n\n")

        f.write("## Kiểm tra split/leakage\n\n")
        f.write(f"- Stratified split: `{overlap_checks['stratified_split']}`\n")
        f.write(f"- Cột single-value bị loại: `{overlap_checks['dropped_single_value_columns']}`\n")
        f.write(f"- Cột tương quan cao bị loại: `{overlap_checks['dropped_high_correlation_columns']}`\n")
        f.write(f"- Dòng feature trùng train/validation: `{overlap_checks['train_val_duplicate_feature_rows']}`\n")
        f.write(f"- Dòng feature trùng train/test: `{overlap_checks['train_test_duplicate_feature_rows']}`\n")
        f.write(f"- Dòng feature trùng validation/test: `{overlap_checks['val_test_duplicate_feature_rows']}`\n\n")

        f.write("## Lớp lệch phân phối mạnh nhất\n\n")
        f.write("| Lớp | Train % | Test % | Chênh lệch |\n")
        f.write("|-----|---------|--------|------------|\n")
        for class_name, train_pct, test_pct, drift in drift_rows:
            f.write(f"| {class_name} | {train_pct:.4f} | {test_pct:.4f} | {drift:.4f} |\n")

        f.write("\n## Khoảng cách validation-test\n\n")
        f.write("| Mô hình | Train F1 | Validation F1 | Test F1 | Gap Val-Test |\n")
        f.write("|---------|----------|---------------|---------|--------------|\n")
        for _, row in gaps_df.iterrows():
            f.write(
                f"| {row['model']} | {row['train_f1_weighted']:.6f} "
                f"| {row['validation_f1_weighted']:.6f} "
                f"| {row['test_f1_weighted']:.6f} "
                f"| {row['validation_test_f1_gap']:.6f} |\n"
            )

        if cv_df is not None and not cv_df.empty:
            f.write("\n## Cross-validation\n\n")
            f.write("| Mô hình | CV F1 weighted mean | CV F1 weighted std | CV F1 macro mean |\n")
            f.write("|---------|---------------------|--------------------|------------------|\n")
            for _, row in cv_df.iterrows():
                f.write(
                    f"| {row['model']} | {row['cv_f1_weighted_mean']:.6f} "
                    f"| {row['cv_f1_weighted_std']:.6f} "
                    f"| {row['cv_f1_macro_mean']:.6f} |\n"
                )

        f.write("\n## Hướng kiểm tra tiếp theo\n\n")
        f.write("- Huấn luyện thêm biến thể `class_weight='balanced'` cho Random Forest/Extra Trees.\n")
        f.write("- Thử oversampling/undersampling cho các lớp hiếm, đặc biệt `UDP-Lag Flood`.\n")
        f.write("- Tách validation theo thời gian hoặc theo file/tấn công để mô phỏng test thực tế hơn.\n")
        f.write("- Không chỉ tối ưu F1 weighted; theo dõi thêm macro F1 và per-class recall.\n")


def run_audit(model_keys: list[str], cv_folds: int,
              correlation_threshold: float = 0.8):
    os.makedirs(AUDIT_DIR, exist_ok=True)

    raw_train_df, raw_test_df = load_dataset()
    preprocessed = preprocess(
        raw_train_df,
        raw_test_df,
        correlation_threshold=correlation_threshold,
    )

    label_map = preprocessed["label_map"]
    class_names = [label_map[i] for i in range(len(label_map))]
    classifiers = get_classifiers(n_classes=len(class_names))

    distribution = pd.concat([
        _class_distribution(preprocessed["y_train_labels"], "train"),
        _class_distribution(preprocessed["y_val_labels"], "validation"),
        _class_distribution(preprocessed["y_test_labels"], "test"),
    ])
    distribution.to_csv(os.path.join(AUDIT_DIR, "class_distribution.csv"), index=False)

    overlap_checks = {
        "train_val_duplicate_feature_rows": _duplicate_feature_overlap(
            preprocessed["X_train_df"], preprocessed["X_val_df"],
        ),
        "train_test_duplicate_feature_rows": _duplicate_feature_overlap(
            preprocessed["X_train_df"], preprocessed["X_test_df"],
        ),
        "val_test_duplicate_feature_rows": _duplicate_feature_overlap(
            preprocessed["X_val_df"], preprocessed["X_test_df"],
        ),
        "stratified_split": preprocessed["stratified_split"],
        "dropped_single_value_columns": len(preprocessed["dropped_single_val"]),
        "dropped_high_correlation_columns": len(preprocessed["dropped_high_corr"]),
    }
    pd.DataFrame([overlap_checks]).to_csv(
        os.path.join(AUDIT_DIR, "split_leakage_checks.csv"),
        index=False,
    )

    metrics_rows = []
    per_class_rows = []
    cv_rows = []

    X_cv = y_cv = None
    if cv_folds > 1:
        X_cv, y_cv, _ = _prepare_cv_data(raw_train_df, raw_test_df)

    for key in model_keys:
        model_name = MODEL_ALIASES[key]
        model = classifiers[model_name]
        print(f"[audit] Training {model_name} ...")
        model.fit(preprocessed["X_train"], preprocessed["y_train"])

        split_data = {
            "train": (preprocessed["X_train"], preprocessed["y_train"]),
            "validation": (preprocessed["X_val"], preprocessed["y_val"]),
            "test": (preprocessed["X_test"], preprocessed["y_test"]),
        }

        for split_name, (X_split, y_split) in split_data.items():
            y_pred = model.predict(X_split)
            metrics_rows.append(_metrics_row(model_name, split_name, y_split, y_pred))
            per_class_rows.extend(
                _write_classification_report(
                    model_name,
                    split_name,
                    y_split,
                    y_pred,
                    class_names,
                )
            )
            _save_confusion_matrix(y_split, y_pred, class_names, model_name, split_name)

        if cv_folds > 1 and X_cv is not None and y_cv is not None:
            print(f"[audit] Running {cv_folds}-fold CV for {model_name} ...")
            cv_result = _run_cross_validation(
                model,
                X_cv,
                y_cv,
                folds=cv_folds,
                correlation_threshold=correlation_threshold,
            )
            cv_result["model"] = model_name
            cv_rows.append(cv_result)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(AUDIT_DIR, "weighted_metrics_by_split.csv"), index=False)

    gap_rows = []
    for model_name, group in metrics_df.groupby("model"):
        split_scores = group.set_index("split")
        gap_rows.append({
            "model": model_name,
            "train_f1_weighted": split_scores.loc["train", "f1_weighted"],
            "validation_f1_weighted": split_scores.loc["validation", "f1_weighted"],
            "test_f1_weighted": split_scores.loc["test", "f1_weighted"],
            "validation_test_f1_gap": (
                split_scores.loc["validation", "f1_weighted"]
                - split_scores.loc["test", "f1_weighted"]
            ),
            "train_validation_f1_gap": (
                split_scores.loc["train", "f1_weighted"]
                - split_scores.loc["validation", "f1_weighted"]
            ),
        })
    gaps_df = pd.DataFrame(gap_rows)
    gaps_df.to_csv(
        os.path.join(AUDIT_DIR, "generalization_gaps.csv"),
        index=False,
    )

    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(AUDIT_DIR, "per_class_metrics.csv"),
        index=False,
    )

    cv_df = pd.DataFrame(cv_rows)
    if not cv_df.empty:
        cv_df.to_csv(os.path.join(AUDIT_DIR, "cross_validation.csv"), index=False)

    _write_summary(distribution, overlap_checks, gaps_df, cv_df)

    print(f"[audit] Done. Reports saved to: {AUDIT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit DDoS ML model evaluation.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_ALIASES),
        default=["random_forest", "knn", "extra_trees", "mlp_classifier", "xgboost"],
        help="Models to train and audit.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Run stratified cross-validation with this many folds. Use 0 to skip.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
        help="Correlation threshold for train-only feature dropping.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_audit(args.models, args.cv_folds, args.correlation_threshold)
