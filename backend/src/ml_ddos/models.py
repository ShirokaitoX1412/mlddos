"""
Leakage-safe training, tuning, and evaluation for DDoS classification.

The public entrypoint is ``train_and_evaluate_from_raw``. It accepts raw
train/test DataFrames, builds sklearn-compatible pipelines, performs robust
cross-validation and randomized search, then evaluates final models on the
held-out test set.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
    cross_validate,
    learning_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .paths import MODELS_DIR as PROJECT_MODELS_DIR
from .paths import RESULTS_DIR as PROJECT_RESULTS_DIR
from .preprocessor import TARGET_COL, harmonize_labels, remove_duplicates

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


RESULTS_DIR = str(PROJECT_RESULTS_DIR)
MODELS_DIR = str(PROJECT_MODELS_DIR)
RANDOM_STATE = 42

LOGGER = logging.getLogger("ml_ddos.training")


class CappedUnderSamplingStrategy:
    """Pickle-safe callable sampling strategy for RandomUnderSampler.

    imbalanced-learn calls this object inside each CV fold with the fold labels.
    It caps only large classes and keeps rare classes unchanged, avoiding the
    leakage-prone pattern of computing resampling counts before cross-validation.
    """

    def __init__(self, cap: int):
        self.cap = int(cap)

    def __call__(self, y: np.ndarray) -> dict[int, int]:
        counts = pd.Series(y).value_counts().sort_index()
        return {
            int(label): int(min(count, self.cap))
            for label, count in counts.items()
        }


LEAKAGE_COLUMN_PATTERNS = (
    r"^label$",
    r"target",
    r"attack",
    r"class",
    r"category",
    r"flow\s*id",
    r"session",
    r"timestamp",
    r"time\s*stamp",
    r"src\s*ip",
    r"source\s*ip",
    r"dst\s*ip",
    r"destination\s*ip",
    r"src\s*port",
    r"source\s*port",
    r"dst\s*port",
    r"destination\s*port",
    r"sport",
    r"dport",
    r"source[_\s-]*file",
    r"filename",
    r"file[_\s-]*name",
    r"__source",
)

GROUP_COLUMN_PATTERNS = (
    r"flow\s*id",
    r"session",
    r"src\s*ip",
    r"source\s*ip",
    r"dst\s*ip",
    r"destination\s*ip",
    r"src\s*port",
    r"source\s*port",
    r"dst\s*port",
    r"destination\s*port",
    r"source[_\s-]*file",
    r"filename",
    r"file[_\s-]*name",
)


@dataclass
class TrainingConfig:
    """Configuration for robust training."""

    random_state: int = RANDOM_STATE
    cv_folds: int = 5
    search_iter: int = 20
    correlation_threshold: float = 0.9
    enable_smote: bool = True
    n_jobs: int = -1
    max_shap_samples: int = 500
    max_learning_curve_samples: int = 25000
    overfit_gap_warning: float = 0.10
    balance_training: bool = True
    attack_to_benign_ratio: float = 1.0
    max_binary_group_samples: int = 40000


class TrafficFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Train-only preprocessing for network flow features.

    Fitted state includes leakage-like columns to drop, constant columns,
    high-correlation columns, imputers, scalers, and encoders. Because this
    object lives inside each sklearn Pipeline, every CV fold learns these
    choices from its own training fold only.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.9,
        leakage_patterns: tuple[str, ...] = LEAKAGE_COLUMN_PATTERNS,
    ):
        self.correlation_threshold = correlation_threshold
        self.leakage_patterns = leakage_patterns

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        X_df = self._to_dataframe(X)

        self.original_columns_ = list(X_df.columns)
        self.leakage_columns_ = self._find_leakage_columns(X_df)
        X_work = X_df.drop(columns=self.leakage_columns_, errors="ignore")

        self.constant_columns_ = [
            col for col in X_work.columns if X_work[col].nunique(dropna=False) <= 1
        ]
        X_work = X_work.drop(columns=self.constant_columns_, errors="ignore")

        self.numeric_columns_ = list(X_work.select_dtypes(include=[np.number]).columns)
        self.categorical_columns_ = [
            col for col in X_work.columns if col not in self.numeric_columns_
        ]

        self.high_corr_columns_ = self._find_high_corr_columns(X_work)
        X_work = X_work.drop(columns=self.high_corr_columns_, errors="ignore")
        self.numeric_columns_ = [
            col for col in self.numeric_columns_ if col not in self.high_corr_columns_
        ]
        self.categorical_columns_ = [
            col for col in self.categorical_columns_ if col not in self.high_corr_columns_
        ]

        if self.numeric_columns_:
            X_num = self._finite_numeric(X_work[self.numeric_columns_])
            self.numeric_imputer_ = SimpleImputer(strategy="median")
            self.numeric_scaler_ = MinMaxScaler()
            X_num_imp = self.numeric_imputer_.fit_transform(X_num)
            self.numeric_scaler_.fit(X_num_imp)
        else:
            self.numeric_imputer_ = None
            self.numeric_scaler_ = None

        if self.categorical_columns_:
            X_cat = X_work[self.categorical_columns_].astype("object")
            self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
            self.categorical_encoder_ = self._new_one_hot_encoder()
            X_cat_imp = self.categorical_imputer_.fit_transform(X_cat)
            self.categorical_encoder_.fit(X_cat_imp)
        else:
            self.categorical_imputer_ = None
            self.categorical_encoder_ = None

        self.feature_names_out_ = self._build_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_df = self._to_dataframe(X)
        X_work = X_df.drop(
            columns=self.leakage_columns_ + self.constant_columns_ + self.high_corr_columns_,
            errors="ignore",
        )

        parts = []
        if self.numeric_columns_:
            X_num = self._finite_numeric(X_work.reindex(columns=self.numeric_columns_))
            X_num_imp = self.numeric_imputer_.transform(X_num)
            parts.append(self.numeric_scaler_.transform(X_num_imp))

        if self.categorical_columns_:
            X_cat = X_work.reindex(columns=self.categorical_columns_).astype("object")
            X_cat_imp = self.categorical_imputer_.transform(X_cat)
            parts.append(self.categorical_encoder_.transform(X_cat_imp))

        if not parts:
            raise ValueError("No usable features remain after preprocessing.")
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)

    def dropped_columns(self) -> dict[str, list[str]]:
        return {
            "leakage": list(self.leakage_columns_),
            "constant": list(self.constant_columns_),
            "high_correlation": list(self.high_corr_columns_),
        }

    def _to_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        columns = getattr(self, "original_columns_", None)
        return pd.DataFrame(X, columns=columns)

    def _find_leakage_columns(self, X: pd.DataFrame) -> list[str]:
        dropped = []
        for col in X.columns:
            normalized = re.sub(r"[_\-]+", " ", str(col).strip().lower())
            if any(re.search(pattern, normalized) for pattern in self.leakage_patterns):
                dropped.append(col)
        return dropped

    def _find_high_corr_columns(self, X: pd.DataFrame) -> list[str]:
        if len(self.numeric_columns_) < 2:
            return []
        numeric = self._finite_numeric(X[self.numeric_columns_])
        numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0)
        corr_matrix = numeric.corr().abs()
        upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper = corr_matrix.where(upper_mask)
        return [col for col in upper.columns if (upper[col] > self.correlation_threshold).any()]

    def _finite_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X.apply(pd.to_numeric, errors="coerce")
        return X_num.replace([np.inf, -np.inf], np.nan)

    def _new_one_hot_encoder(self) -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _build_feature_names(self) -> list[str]:
        names = list(self.numeric_columns_)
        if self.categorical_columns_:
            cat_names = self.categorical_encoder_.get_feature_names_out(self.categorical_columns_)
            names.extend(cat_names.tolist())
        return names


def configure_logging(results_dir: str = RESULTS_DIR) -> None:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training.log")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def get_classifiers(n_classes: int = 7, random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Base models retained for compatibility with older scripts."""
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_kwargs = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 5.0,
        "objective": objective,
        "random_state": random_state,
        "n_jobs": -1,
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        "tree_method": "hist",
    }
    if n_classes > 2:
        xgb_kwargs["num_class"] = n_classes

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            max_samples=0.75,
            random_state=random_state,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=20,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(**xgb_kwargs),
    }


def train_and_evaluate_from_raw(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_dir: str = RESULTS_DIR,
    models_dir: str = MODELS_DIR,
    config: TrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the production-oriented training pipeline from raw DataFrames."""
    config = config or TrainingConfig()
    configure_logging(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    LOGGER.info("Starting leakage-safe DDoS training pipeline")
    raw_train, raw_test, label_encoder, label_map = prepare_raw_splits(train_df, test_df)
    raw_train, raw_test = remove_cross_source_feature_duplicates(raw_train, raw_test)
    X_train = raw_train.drop(columns=[TARGET_COL])
    y_train = label_encoder.transform(raw_train[TARGET_COL])
    X_test = raw_test.drop(columns=[TARGET_COL])
    y_test = label_encoder.transform(raw_test[TARGET_COL])
    class_names = [label_map[i] for i in range(len(label_map))]

    X_train, y_train = balance_training_distribution(
        X_train,
        y_train,
        class_names,
        config,
        results_dir,
    )
    groups = infer_groups(X_train)
    cv = make_cv(y_train, groups, config)
    imbalance = write_class_balance_report(y_train, y_test, class_names, results_dir)

    LOGGER.info("Training rows=%s, test rows=%s, classes=%s", len(X_train), len(X_test), class_names)
    LOGGER.info("CV strategy=%s", type(cv).__name__)
    if not IMBLEARN_AVAILABLE and config.enable_smote:
        LOGGER.warning("imbalanced-learn is not installed; SMOTE candidates are skipped")

    candidates = fit_all_candidates(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        cv=cv,
        n_classes=len(class_names),
        config=config,
        imbalance=imbalance,
        results_dir=results_dir,
    )

    summary_rows = []
    per_class_rows = []
    final_models = {}
    for candidate in candidates:
        LOGGER.info("Fitting final model: %s", candidate["name"])
        final_model = fit_final_model(
            candidate["estimator"],
            X_train,
            y_train,
            config,
            use_xgb_early_stopping=candidate["base_name"] == "XGBoost",
        )
        final_models[candidate["name"]] = final_model
        save_model(final_model, candidate["name"], models_dir)

        train_metrics = evaluate_split(
            final_model,
            X_train,
            y_train,
            class_names,
            candidate["name"],
            "train",
            results_dir,
            save_artifacts=False,
        )
        test_metrics = evaluate_split(
            final_model,
            X_test,
            y_test,
            class_names,
            candidate["name"],
            "test",
            results_dir,
            save_artifacts=True,
        )
        per_class_rows.extend(test_metrics.pop("per_class_rows"))
        train_metrics.pop("per_class_rows", None)

        cv_metrics = candidate["cv_metrics"]
        row = {
            "Model": candidate["name"],
            "Train Accuracy": train_metrics["accuracy"],
            "Train F1-Score": train_metrics["f1_weighted"],
            "CV Accuracy Mean": cv_metrics["cv_accuracy_mean"],
            "CV Accuracy Std": cv_metrics["cv_accuracy_std"],
            "CV F1-Score Mean": cv_metrics["cv_f1_weighted_mean"],
            "CV F1-Score Std": cv_metrics["cv_f1_weighted_std"],
            "CV F1-Macro Mean": cv_metrics["cv_f1_macro_mean"],
            "CV ROC-AUC Mean": cv_metrics["cv_roc_auc_ovr_weighted_mean"],
            "Test Accuracy": test_metrics["accuracy"],
            "Test Precision": test_metrics["precision_weighted"],
            "Test Recall": test_metrics["recall_weighted"],
            "Test F1-Score": test_metrics["f1_weighted"],
            "Test F1-Macro": test_metrics["f1_macro"],
            "Test ROC-AUC": test_metrics["roc_auc_ovr_weighted"],
            "Overfit Gap F1": train_metrics["f1_weighted"] - cv_metrics["cv_f1_weighted_mean"],
            "Generalization Gap F1": cv_metrics["cv_f1_weighted_mean"] - test_metrics["f1_weighted"],
            "Selection Score": selection_score(train_metrics, cv_metrics),
        }
        summary_rows.append(row)

        plot_feature_importance(final_model, candidate["name"], results_dir)
        maybe_generate_shap(final_model, X_test, candidate["name"], class_names, results_dir, config)

    summary = pd.DataFrame(summary_rows).sort_values("Selection Score", ascending=False)
    summary.to_csv(os.path.join(results_dir, "robust_model_summary.csv"), index=False)
    pd.DataFrame(per_class_rows).to_csv(os.path.join(results_dir, "test_per_class_metrics.csv"), index=False)

    selected_name = summary.iloc[0]["Model"]
    selected_model = final_models[selected_name]
    save_model(selected_model, "selected_model", models_dir)

    plot_train_cv_test(summary, results_dir)
    plot_learning_curve_safe(selected_model, X_train, y_train, cv, groups, selected_name, results_dir, config)
    write_generalization_report(summary, selected_name, results_dir, config)

    cv_scores_df = summary.set_index("Model")[
        ["CV Accuracy Mean", "CV Accuracy Std", "CV F1-Score Mean", "CV F1-Score Std", "CV F1-Macro Mean"]
    ]
    test_scores_df = summary.set_index("Model")[
        ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test F1-Macro", "Test ROC-AUC"]
    ]
    cv_scores_df.to_csv(os.path.join(results_dir, "validation_scores.csv"))
    test_scores_df.to_csv(os.path.join(results_dir, "test_scores.csv"))

    LOGGER.info("Selected model=%s", selected_name)
    LOGGER.info("Artifacts saved to %s and %s", results_dir, models_dir)
    close_logging_handlers()
    return cv_scores_df, test_scores_df


def prepare_raw_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, dict[int, str]]:
    train_df, test_df = harmonize_labels(train_df, test_df)
    train_df = remove_duplicates(train_df, "training set")
    test_df = remove_duplicates(test_df, "testing set")

    train_labels = set(train_df[TARGET_COL].dropna().unique())
    unknown_test = sorted(set(test_df[TARGET_COL].dropna().unique()) - train_labels)
    if unknown_test:
        LOGGER.warning("Dropping test labels not seen in training: %s", unknown_test)
        test_df = test_df[test_df[TARGET_COL].isin(train_labels)].copy()

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[TARGET_COL])
    label_map = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), label_encoder, label_map


def remove_cross_source_feature_duplicates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact feature-duplicate test rows already present in training.

    This is deliberately applied after ordinary per-split duplicate removal and
    before X/y separation. The target label is excluded from the hash, so the
    audit catches duplicated flows even when labels disagree.
    """
    if TARGET_COL not in train_df.columns or TARGET_COL not in test_df.columns:
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    train_features = train_df.drop(columns=[TARGET_COL], errors="ignore")
    test_features = test_df.drop(columns=[TARGET_COL], errors="ignore")
    common_columns = [col for col in train_features.columns if col in test_features.columns]
    if not common_columns:
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    train_hash = pd.util.hash_pandas_object(train_features[common_columns], index=False)
    test_hash = pd.util.hash_pandas_object(test_features[common_columns], index=False)
    duplicate_mask = test_hash.isin(set(train_hash.to_numpy()))
    n_removed = int(duplicate_mask.sum())
    if n_removed:
        LOGGER.warning(
            "Removed %s exact feature-duplicate rows from test set before evaluation.",
            f"{n_removed:,}",
        )
    return train_df.reset_index(drop=True), test_df.loc[~duplicate_mask].reset_index(drop=True)


def infer_groups(X: pd.DataFrame) -> pd.Series | None:
    for col in X.columns:
        normalized = re.sub(r"[_\-]+", " ", str(col).strip().lower())
        if any(re.search(pattern, normalized) for pattern in GROUP_COLUMN_PATTERNS):
            groups = X[col].astype(str).fillna("missing")
            if groups.nunique() > 1:
                LOGGER.info("Using group-aware CV based on column: %s", col)
                return groups
    return None


def make_cv(y: np.ndarray, groups: pd.Series | None, config: TrainingConfig):
    class_counts = np.bincount(y)
    min_class_count = int(class_counts[class_counts > 0].min())
    n_splits = max(2, min(config.cv_folds, min_class_count))
    if groups is not None and groups.nunique() >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)


def balance_training_distribution(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    class_names: list[str],
    config: TrainingConfig,
    results_dir: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Downsample the training split to reduce benign/attack imbalance.

    The test set is intentionally untouched. Balancing is applied only to the
    training split before CV/final fitting so reported test performance remains
    a real unseen-distribution measurement.
    """
    if not config.balance_training:
        LOGGER.info("Training balancing disabled.")
        return X_train.reset_index(drop=True), y_train

    benign_indices = [idx for idx, name in enumerate(class_names) if name.lower() == "benign"]
    if not benign_indices:
        LOGGER.warning("Could not find Benign class; skipping binary train balancing.")
        return X_train.reset_index(drop=True), y_train

    benign_label = benign_indices[0]
    rng = np.random.default_rng(config.random_state)
    y_series = pd.Series(y_train, index=X_train.index)
    benign_idx = y_series[y_series == benign_label].index.to_numpy()
    attack_idx = y_series[y_series != benign_label].index.to_numpy()

    if len(benign_idx) == 0 or len(attack_idx) == 0:
        LOGGER.warning("Cannot balance train split because one binary group is empty.")
        return X_train.reset_index(drop=True), y_train

    benign_target = min(len(benign_idx), config.max_binary_group_samples)
    attack_target = min(
        len(attack_idx),
        int(round(benign_target * config.attack_to_benign_ratio)),
        config.max_binary_group_samples,
    )
    benign_sample = rng.choice(benign_idx, size=benign_target, replace=False)
    attack_sample = stratified_attack_sample(
        y_series,
        attack_idx,
        attack_target,
        benign_label,
        rng,
    )

    selected_idx = np.concatenate([benign_sample, attack_sample])
    rng.shuffle(selected_idx)

    before = class_distribution_frame(y_train, class_names, split="train_before_balance")
    after_y = y_series.loc[selected_idx].to_numpy()
    after = class_distribution_frame(after_y, class_names, split="train_after_balance")
    pd.concat([before, after], ignore_index=True).to_csv(
        os.path.join(results_dir, "training_balance_before_after.csv"),
        index=False,
    )

    summary = pd.DataFrame([
        {
            "metric": "rows_before",
            "value": int(len(y_train)),
        },
        {
            "metric": "rows_after",
            "value": int(len(selected_idx)),
        },
        {
            "metric": "benign_before",
            "value": int(len(benign_idx)),
        },
        {
            "metric": "attack_before",
            "value": int(len(attack_idx)),
        },
        {
            "metric": "benign_after",
            "value": int((after_y == benign_label).sum()),
        },
        {
            "metric": "attack_after",
            "value": int((after_y != benign_label).sum()),
        },
        {
            "metric": "attack_to_benign_ratio_after",
            "value": float((after_y != benign_label).sum() / max((after_y == benign_label).sum(), 1)),
        },
    ])
    summary.to_csv(os.path.join(results_dir, "training_balance_summary.csv"), index=False)
    LOGGER.info(
        "Balanced training split: rows %s -> %s, benign=%s, attack=%s",
        len(y_train),
        len(selected_idx),
        int((after_y == benign_label).sum()),
        int((after_y != benign_label).sum()),
    )
    return X_train.loc[selected_idx].reset_index(drop=True), after_y


def stratified_attack_sample(
    y_series: pd.Series,
    attack_idx: np.ndarray,
    target_total: int,
    benign_label: int,
    rng: np.random.Generator,
) -> np.ndarray:
    attack_counts = y_series.loc[attack_idx].value_counts().sort_index()
    attack_counts = attack_counts[attack_counts.index != benign_label]
    if target_total >= int(attack_counts.sum()):
        return attack_idx

    labels = attack_counts.index.to_numpy()
    counts = attack_counts.to_dict()
    base_quota = max(1, target_total // max(len(labels), 1))
    targets = {label: min(int(counts[label]), base_quota) for label in labels}

    while sum(targets.values()) < target_total:
        remaining = {
            label: int(counts[label]) - targets[label]
            for label in labels
            if int(counts[label]) > targets[label]
        }
        if not remaining:
            break
        capacity_total = sum(remaining.values())
        leftover = target_total - sum(targets.values())
        progressed = False
        for label, capacity in sorted(remaining.items(), key=lambda item: item[1], reverse=True):
            add = max(1, int(round(leftover * capacity / capacity_total)))
            add = min(add, capacity, target_total - sum(targets.values()))
            if add > 0:
                targets[label] += add
                progressed = True
            if sum(targets.values()) >= target_total:
                break
        if not progressed:
            break

    sampled = []
    for label, target in targets.items():
        label_idx = y_series[y_series == label].index.to_numpy()
        sampled.append(rng.choice(label_idx, size=target, replace=False))
    return np.concatenate(sampled)


def class_distribution_frame(y: np.ndarray, class_names: list[str], split: str) -> pd.DataFrame:
    counts = np.bincount(y, minlength=len(class_names))
    return pd.DataFrame(
        {
            "split": split,
            "class": class_names,
            "count": counts.astype(int),
            "percent": counts / max(len(y), 1) * 100,
        }
    )


def write_class_balance_report(
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    results_dir: str,
) -> dict[str, Any]:
    rows = []
    for split_name, y in [("train", y_train), ("test", y_test)]:
        counts = np.bincount(y, minlength=len(class_names))
        for idx, count in enumerate(counts):
            rows.append(
                {
                    "split": split_name,
                    "class": class_names[idx],
                    "count": int(count),
                    "percent": float(count / len(y) * 100),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "class_distribution.csv"), index=False)
    train_counts = np.bincount(y_train, minlength=len(class_names))
    nonzero = train_counts[train_counts > 0]
    imbalance_ratio = float(nonzero.max() / nonzero.min())
    is_imbalanced = imbalance_ratio >= 3.0 or (nonzero.min() / len(y_train)) < 0.05
    return {
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "sample_weight": compute_sample_weight(class_weight="balanced", y=y_train),
    }


def fit_all_candidates(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: pd.Series | None,
    cv,
    n_classes: int,
    config: TrainingConfig,
    imbalance: dict[str, Any],
    results_dir: str,
) -> list[dict[str, Any]]:
    candidates = []
    for base_name, model in get_classifiers(n_classes=n_classes, random_state=config.random_state).items():
        variants = [("weighted", None)]
        if base_name in {"Random Forest", "Extra Trees"} and config.enable_smote and IMBLEARN_AVAILABLE:
            variants.append(("smote", SMOTE(random_state=config.random_state, k_neighbors=3)))

        for variant_name, sampler in variants:
            name = base_name if variant_name == "weighted" else f"{base_name} + SMOTE"
            estimator = build_pipeline(model, config, sampler=sampler)
            if base_name in {"Random Forest", "Extra Trees", "XGBoost"}:
                estimator, cv_metrics = randomized_search(
                    name,
                    estimator,
                    base_name,
                    X_train,
                    y_train,
                    groups,
                    cv,
                    config,
                    imbalance,
                    n_classes,
                    results_dir,
                )
            else:
                cv_metrics = cross_validate_estimator(name, estimator, X_train, y_train, groups, cv, config)
                estimator.fit(X_train, y_train)

            candidates.append(
                {
                    "name": name,
                    "base_name": base_name,
                    "estimator": estimator,
                    "cv_metrics": cv_metrics,
                }
            )
    return candidates


def build_pipeline(model: BaseEstimator, config: TrainingConfig, sampler=None):
    steps = [
        ("preprocess", TrafficFeaturePreprocessor(config.correlation_threshold)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", clone(model)))
    if sampler is not None:
        return ImbPipeline(steps)
    return Pipeline(steps)


def randomized_search(
    name: str,
    estimator,
    base_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series | None,
    cv,
    config: TrainingConfig,
    imbalance: dict[str, Any],
    n_classes: int,
    results_dir: str,
):
    LOGGER.info("RandomizedSearchCV: %s", name)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions(base_name, n_classes, imbalance),
        n_iter=config.search_iter,
        scoring=scoring_dict(),
        refit="f1_weighted",
        cv=cv,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        verbose=1,
        return_train_score=True,
        error_score=np.nan,
    )
    fit_params = {}
    if base_name == "XGBoost" and n_classes > 2 and "sampler" not in estimator.named_steps:
        fit_params["model__sample_weight"] = imbalance["sample_weight"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitFailedWarning)
        if groups is not None:
            search.fit(X, y, groups=groups, **fit_params)
        else:
            search.fit(X, y, **fit_params)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(
        os.path.join(results_dir, f"{safe_name(name)}_randomized_search.csv"),
        index=False,
    )
    return search.best_estimator_, extract_cv_metrics(search.best_index_, cv_results)


def param_distributions(base_name: str, n_classes: int, imbalance: dict[str, Any]) -> dict[str, list[Any]]:
    if base_name == "Random Forest":
        return {
            "model__n_estimators": [160, 220, 320],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_samples_split": [200, 500, 1000, 1500],
            "model__min_samples_leaf": [100, 200, 500, 800],
            "model__max_features": ["sqrt", "log2", 0.25, 0.4],
            "model__class_weight": ["balanced", "balanced_subsample"],
            "model__max_samples": [0.35, 0.45, 0.6],
            "model__ccp_alpha": [0.001, 0.005, 0.01, 0.02],
        }
    if base_name == "Extra Trees":
        return {
            "model__n_estimators": [160, 220, 320],
            "model__max_depth": [5, 7, 9, 12],
            "model__min_samples_split": [80, 120, 200, 500],
            "model__min_samples_leaf": [30, 40, 80, 120],
            "model__max_features": ["sqrt", "log2", 0.25, 0.4],
            "model__class_weight": ["balanced"],
            "model__bootstrap": [True],
            "model__max_samples": [0.45, 0.6, 0.75],
            "model__ccp_alpha": [0.0, 0.0001, 0.001],
        }
    if base_name == "XGBoost":
        params = {
            "model__n_estimators": [160, 220, 320],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.01, 0.03, 0.05],
            "model__min_child_weight": [10, 20, 25, 40],
            "model__subsample": [0.55, 0.65, 0.75],
            "model__colsample_bytree": [0.55, 0.65, 0.75],
            "model__gamma": [0.5, 1.0, 2.0],
            "model__reg_alpha": [0.1, 0.5, 1.0],
            "model__reg_lambda": [8, 12, 20],
            "model__max_delta_step": [1, 3, 5],
        }
        if n_classes == 2 and imbalance["is_imbalanced"]:
            params["model__scale_pos_weight"] = [1, imbalance["imbalance_ratio"]]
        return params
    return {}


def scoring_dict() -> dict[str, str]:
    return {
        "accuracy": "accuracy",
        "precision_weighted": make_scorer(
            precision_score,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": make_scorer(
            recall_score,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": make_scorer(
            f1_score,
            average="weighted",
            zero_division=0,
        ),
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }


def extract_cv_metrics(best_index: int, cv_results: pd.DataFrame) -> dict[str, float]:
    return {
        "cv_accuracy_mean": float(cv_results.loc[best_index, "mean_test_accuracy"]),
        "cv_accuracy_std": float(cv_results.loc[best_index, "std_test_accuracy"]),
        "cv_precision_weighted_mean": float(cv_results.loc[best_index, "mean_test_precision_weighted"]),
        "cv_recall_weighted_mean": float(cv_results.loc[best_index, "mean_test_recall_weighted"]),
        "cv_f1_weighted_mean": float(cv_results.loc[best_index, "mean_test_f1_weighted"]),
        "cv_f1_weighted_std": float(cv_results.loc[best_index, "std_test_f1_weighted"]),
        "cv_f1_macro_mean": float(cv_results.loc[best_index, "mean_test_f1_macro"]),
        "cv_roc_auc_ovr_weighted_mean": float(cv_results.loc[best_index, "mean_test_roc_auc_ovr_weighted"]),
        "cv_train_f1_weighted_mean": float(cv_results.loc[best_index, "mean_train_f1_weighted"]),
    }


def cross_validate_estimator(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series | None,
    cv,
    config: TrainingConfig,
) -> dict[str, float]:
    LOGGER.info("Cross-validating baseline: %s", name)
    kwargs = {"groups": groups} if groups is not None else {}
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring_dict(),
        n_jobs=config.n_jobs,
        return_train_score=True,
        error_score=np.nan,
        **kwargs,
    )
    return {
        "cv_accuracy_mean": float(np.nanmean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.nanstd(scores["test_accuracy"])),
        "cv_precision_weighted_mean": float(np.nanmean(scores["test_precision_weighted"])),
        "cv_recall_weighted_mean": float(np.nanmean(scores["test_recall_weighted"])),
        "cv_f1_weighted_mean": float(np.nanmean(scores["test_f1_weighted"])),
        "cv_f1_weighted_std": float(np.nanstd(scores["test_f1_weighted"])),
        "cv_f1_macro_mean": float(np.nanmean(scores["test_f1_macro"])),
        "cv_roc_auc_ovr_weighted_mean": float(np.nanmean(scores["test_roc_auc_ovr_weighted"])),
        "cv_train_f1_weighted_mean": float(np.nanmean(scores["train_f1_weighted"])),
    }


def fit_final_model(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    config: TrainingConfig,
    use_xgb_early_stopping: bool,
):
    if not use_xgb_early_stopping:
        estimator.fit(X, y)
        return estimator
    return fit_xgb_with_early_stopping(estimator, X, y, config)


def fit_xgb_with_early_stopping(estimator, X: pd.DataFrame, y: np.ndarray, config: TrainingConfig):
    train_idx, eval_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.15,
        random_state=config.random_state,
        stratify=y,
    )
    X_fit, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
    y_fit, y_eval = y[train_idx], y[eval_idx]

    preprocess = clone(estimator.named_steps["preprocess"])
    X_fit_tx = preprocess.fit_transform(X_fit, y_fit)
    X_eval_tx = preprocess.transform(X_eval)

    sampler = estimator.named_steps.get("sampler")
    if sampler is not None:
        sampler = clone(sampler)
        X_model_fit, y_model_fit = sampler.fit_resample(X_fit_tx, y_fit)
    else:
        X_model_fit, y_model_fit = X_fit_tx, y_fit

    model = clone(estimator.named_steps["model"])
    model.set_params(early_stopping_rounds=30)
    fit_kwargs = {"eval_set": [(X_eval_tx, y_eval)], "verbose": False}
    if len(np.unique(y)) > 2 and sampler is None:
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_model_fit)
    model.fit(X_model_fit, y_model_fit, **fit_kwargs)

    steps = [("preprocess", preprocess)]
    if sampler is not None:
        steps.append(("sampler", sampler))
        return ImbPipeline(steps + [("model", model)])
    return Pipeline(steps + [("model", model)])


def evaluate_split(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: list[str],
    model_name: str,
    split_name: str,
    results_dir: str,
    save_artifacts: bool,
) -> dict[str, Any]:
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr_weighted": safe_roc_auc(y, y_proba, len(class_names)),
    }

    report_dict = classification_report(
        y,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    metrics["per_class_rows"] = [
        {
            "model": model_name,
            "split": split_name,
            "class": class_name,
            "precision": report_dict[class_name]["precision"],
            "recall": report_dict[class_name]["recall"],
            "f1_score": report_dict[class_name]["f1-score"],
            "support": report_dict[class_name]["support"],
        }
        for class_name in class_names
    ]

    if save_artifacts:
        write_classification_report(y, y_pred, class_names, model_name, split_name, results_dir)
        plot_confusion_matrix(y, y_pred, class_names, model_name, split_name, results_dir)
        if y_proba is not None:
            plot_roc_curve(y, y_proba, class_names, model_name, split_name, results_dir)
    return metrics


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray | None, n_classes: int) -> float:
    if y_proba is None:
        return float("nan")
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except ValueError:
        return float("nan")


def selection_score(train_metrics: dict[str, float], cv_metrics: dict[str, float]) -> float:
    overfit_penalty = max(0.0, train_metrics["f1_weighted"] - cv_metrics["cv_f1_weighted_mean"])
    stability_penalty = cv_metrics["cv_f1_weighted_std"]
    macro_bonus = 0.35 * cv_metrics["cv_f1_macro_mean"]
    return (
        cv_metrics["cv_f1_weighted_mean"]
        + macro_bonus
        - 2.0 * overfit_penalty
        - 2.0 * stability_penalty
    )


def save_model(model, model_name: str, models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{safe_name(model_name)}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(model_name: str, models_dir: str = MODELS_DIR):
    path = os.path.join(models_dir, f"{safe_name(model_name)}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    split_name: str,
    results_dir: str,
) -> None:
    path = os.path.join(results_dir, f"{safe_name(model_name)}_{split_name}_classification_report.txt")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report: {model_name} - {split_name}\n")
        f.write("=" * 80 + "\n")
        f.write(report)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    split_name: str,
    results_dir: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name} - {split_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name(model_name)}_{split_name}_confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    model_name: str,
    split_name: str,
    results_dir: str,
) -> None:
    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.4f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        for idx, color in enumerate(colors):
            fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
            plt.plot(fpr, tpr, color=color, lw=2, label=f"{class_names[idx]} AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} - {split_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name(model_name)}_{split_name}_roc_curve.png"), dpi=150)
    plt.close()


def plot_feature_importance(estimator, model_name: str, results_dir: str, top_n: int = 25) -> None:
    model = estimator.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    feature_names = estimator.named_steps["preprocess"].get_feature_names_out()
    importances = np.asarray(model.feature_importances_)
    limit = min(top_n, len(importances))
    order = np.argsort(importances)[::-1][:limit]
    rows = pd.DataFrame({"feature": feature_names[order], "importance": importances[order]})
    rows.to_csv(os.path.join(results_dir, f"{safe_name(model_name)}_feature_importance.csv"), index=False)

    plt.figure(figsize=(12, 8))
    plt.barh(rows["feature"][::-1], rows["importance"][::-1], color="#4477AA")
    plt.xlabel("Importance")
    plt.title(f"Top Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{safe_name(model_name)}_feature_importance.png"), dpi=150)
    plt.close()


def maybe_generate_shap(
    estimator,
    X_test: pd.DataFrame,
    model_name: str,
    class_names: list[str],
    results_dir: str,
    config: TrainingConfig,
) -> None:
    model = estimator.named_steps["model"]
    if type(model).__name__ not in {"RandomForestClassifier", "ExtraTreesClassifier", "XGBClassifier"}:
        return
    try:
        import shap

        sample = X_test.sample(
            n=min(config.max_shap_samples, len(X_test)),
            random_state=config.random_state,
        )
        X_tx = estimator.named_steps["preprocess"].transform(sample)
        feature_names = estimator.named_steps["preprocess"].get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_tx)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_tx,
            feature_names=feature_names,
            class_names=class_names,
            plot_type="bar",
            show=False,
            max_display=25,
        )
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{safe_name(model_name)}_shap_summary_bar.png"), dpi=150)
        plt.close()
    except Exception as exc:  # SHAP is helpful but should not break training.
        LOGGER.warning("SHAP generation skipped for %s: %s", model_name, exc)


def plot_train_cv_test(summary: pd.DataFrame, results_dir: str) -> None:
    plot_df = summary[["Model", "Train F1-Score", "CV F1-Score Mean", "Test F1-Score"]].copy()
    plot_df = plot_df.melt(id_vars="Model", var_name="Split", value_name="F1")
    plt.figure(figsize=(13, 7))
    sns.barplot(data=plot_df, x="Model", y="F1", hue="Split")
    plt.title("Train vs Cross-Validation vs Test F1")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "train_cv_test_f1_comparison.png"), dpi=150)
    plt.close()


def plot_learning_curve_safe(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    cv,
    groups: pd.Series | None,
    model_name: str,
    results_dir: str,
    config: TrainingConfig,
) -> None:
    try:
        if len(X) > config.max_learning_curve_samples:
            _, X_small, _, y_small = train_test_split(
                X,
                y,
                test_size=config.max_learning_curve_samples,
                random_state=config.random_state,
                stratify=y,
            )
            X_curve, y_curve = X_small, y_small
            groups_curve = None
        else:
            X_curve, y_curve = X, y
            groups_curve = groups

        kwargs = {"groups": groups_curve} if groups_curve is not None else {}
        train_sizes, train_scores, val_scores = learning_curve(
            clone(estimator),
            X_curve,
            y_curve,
            cv=cv if groups_curve is not None else StratifiedKFold(n_splits=3, shuffle=True, random_state=config.random_state),
            scoring="f1_weighted",
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=config.n_jobs,
            **kwargs,
        )
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.nanmean(train_scores, axis=1), marker="o", label="Train")
        plt.plot(train_sizes, np.nanmean(val_scores, axis=1), marker="o", label="Validation")
        plt.fill_between(
            train_sizes,
            np.nanmean(val_scores, axis=1) - np.nanstd(val_scores, axis=1),
            np.nanmean(val_scores, axis=1) + np.nanstd(val_scores, axis=1),
            alpha=0.2,
        )
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Weighted F1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "selected_model_learning_curve.png"), dpi=150)
        plt.close()
    except Exception as exc:
        LOGGER.warning("Learning curve skipped: %s", exc)


def write_generalization_report(
    summary: pd.DataFrame,
    selected_name: str,
    results_dir: str,
    config: TrainingConfig,
) -> None:
    path = os.path.join(results_dir, "generalization_report.md")
    warnings_rows = summary[
        (summary["Overfit Gap F1"] > config.overfit_gap_warning)
        | (summary["Generalization Gap F1"] > config.overfit_gap_warning)
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Generalization Report\n\n")
        f.write(f"Selected model: **{selected_name}**\n\n")
        f.write("Selection is based on cross-validation weighted F1, macro F1, fold stability, and overfit penalty. ")
        f.write("The held-out test set is reported as an external generalization check, not as the sole selector.\n\n")
        f.write("## Why the Original Pipeline Likely Overfit\n\n")
        f.write("- A single validation split can be too similar to the training data and hide distribution shift.\n")
        f.write("- Precomputing preprocessing outside cross-validation risks fitting medians, scalers, or feature filters on validation data.\n")
        f.write("- Weighted metrics can hide poor recall for minority DDoS classes.\n")
        f.write("- Random row splitting can leak near-duplicate flows across train and validation.\n\n")
        f.write("## Mitigations in This Pipeline\n\n")
        f.write("- All preprocessing is inside sklearn/imblearn Pipelines and is refit per CV fold.\n")
        f.write("- StratifiedKFold is used by default; StratifiedGroupKFold is used when session/IP/source grouping columns exist.\n")
        f.write("- Class imbalance is detected and handled with class weights, balanced sample weights, and optional SMOTE variants.\n")
        f.write("- Model selection penalizes high train-CV gaps and unstable fold performance.\n")
        f.write("- Potential identifier/leakage columns and highly correlated features are removed using training-fold statistics only.\n\n")
        if warnings_rows.empty:
            f.write("## Overfitting Warnings\n\nNo large F1 gaps exceeded the configured threshold.\n\n")
        else:
            f.write("## Overfitting Warnings\n\n")
            for _, row in warnings_rows.iterrows():
                f.write(
                    f"- {row['Model']}: train-CV gap={row['Overfit Gap F1']:.4f}, "
                    f"CV-test gap={row['Generalization Gap F1']:.4f}. "
                    "Likely causes include duplicate/near-duplicate flows, class distribution shift, "
                    "or attack-source/domain shift between train and test.\n"
                )
        f.write("\n## Summary Table\n\n")
        f.write(dataframe_to_markdown(summary))


def generate_markdown_report(val_scores_df: pd.DataFrame, test_scores_df: pd.DataFrame) -> str:
    """Generate a concise Markdown report for compatibility with main.py."""
    lines = ["## Robust Model Comparison", ""]
    lines.append("### Cross-Validation Performance")
    lines.append(dataframe_to_markdown(val_scores_df.reset_index()))
    lines.append("")
    lines.append("### Held-Out Test Performance")
    lines.append(dataframe_to_markdown(test_scores_df.reset_index()))
    lines.append("")
    lines.append("Selection prioritizes stable cross-validation F1, macro F1, and low overfitting risk.")
    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small Markdown table without optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def close_logging_handlers() -> None:
    for handler in LOGGER.handlers[:]:
        handler.close()
        LOGGER.removeHandler(handler)


def train_and_evaluate(preprocessed: dict, results_dir: str = RESULTS_DIR, models_dir: str = MODELS_DIR):
    """
    Backward-compatible adapter.

    The robust pipeline should be called with raw DataFrames through
    ``train_and_evaluate_from_raw``. This adapter keeps older notebooks from
    failing, but it cannot recover raw columns once arrays have already been
    preprocessed.
    """
    raise RuntimeError(
        "train_and_evaluate now requires raw DataFrames. "
        "Use train_and_evaluate_from_raw(train_df, test_df) or run python main.py."
    )
