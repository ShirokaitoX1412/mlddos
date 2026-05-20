"""
models.py - DDoS Detection Classifier Definitions, Training & Evaluation

Defines 5 ML classifiers with optimized hyperparameters, trains them,
and generates comprehensive evaluation reports including:
  - Confusion Matrix
  - ROC Curve (per-class, One-vs-Rest)
  - Feature Importance (for tree-based models)
  - Metrics: Accuracy, Precision, Recall, F1-Score
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from tqdm import tqdm

from .paths import MODELS_DIR as PROJECT_MODELS_DIR
from .paths import RESULTS_DIR as PROJECT_RESULTS_DIR

RESULTS_DIR = str(PROJECT_RESULTS_DIR)
MODELS_DIR = str(PROJECT_MODELS_DIR)


def get_classifiers(n_classes: int = 7, random_state: int = 42) -> dict:
    """Return a dictionary of named classifiers with tuned hyperparameters.

    Parameters
    ----------
    n_classes : int
        Number of target classes.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name -> sklearn-compatible estimator.
    """
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=n_classes,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
    }

    return classifiers


def train_and_evaluate(preprocessed: dict, results_dir: str = RESULTS_DIR,
                       models_dir: str = MODELS_DIR) -> pd.DataFrame:
    """Train all 5 models, evaluate, generate visualizations, and save models.

    Parameters
    ----------
    preprocessed : dict
        Output from preprocessor.preprocess().
    results_dir : str
        Directory to save plots and reports.
    models_dir : str
        Directory to save trained model files.

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for all models.
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X_train = preprocessed["X_train"]
    X_val = preprocessed["X_val"]
    X_test = preprocessed["X_test"]
    y_train = preprocessed["y_train"]
    y_val = preprocessed["y_val"]
    y_test = preprocessed["y_test"]
    label_map = preprocessed["label_map"]
    feature_names = preprocessed["feature_names"]

    n_classes = len(label_map)
    class_names = [label_map[i] for i in range(n_classes)]

    classifiers = get_classifiers(n_classes=n_classes)
    scores_list = []
    trained_models = {}

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in tqdm(classifiers.items(), desc="Training Models"):
        print(f"\n{'─' * 50}")
        print(f"  Training: {name}")
        print(f"{'─' * 50}")

        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict on validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        # Metrics (weighted average for multi-class)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        scores_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        })

        print(f"  Accuracy:  {acc:.6f}")
        print(f"  Precision: {prec:.6f}")
        print(f"  Recall:    {rec:.6f}")
        print(f"  F1-Score:  {f1:.6f}")

        # Classification report
        report = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)
        report_path = os.path.join(results_dir, f"{_safe_name(name)}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification Report: {name}\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        print(f"  Report saved: {report_path}")

        # Confusion Matrix
        _plot_confusion_matrix(y_val, y_pred, class_names, name, results_dir)

        # ROC Curve
        _plot_roc_curve(y_val, y_proba, n_classes, class_names, name, results_dir)

        # Feature Importance (tree-based models only)
        if hasattr(model, "feature_importances_"):
            _plot_feature_importance(model, feature_names, name, results_dir)

        # Save model
        model_path = os.path.join(models_dir, f"{_safe_name(name)}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved: {model_path}")

    # Also evaluate all models on the test set for final reporting
    print("\n" + "=" * 60)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 60)

    test_scores_list = []
    for name, model in trained_models.items():
        y_test_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)

        test_scores_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        })
        print(f"  {name:<20s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    # Build comparison DataFrames
    val_scores_df = pd.DataFrame(scores_list).set_index("Model")
    test_scores_df = pd.DataFrame(test_scores_list).set_index("Model")

    # Save comparison tables
    val_scores_df.to_csv(os.path.join(results_dir, "validation_scores.csv"))
    test_scores_df.to_csv(os.path.join(results_dir, "test_scores.csv"))

    # Plot comparison bar chart
    _plot_model_comparison(val_scores_df, results_dir)

    return val_scores_df, test_scores_df


def _safe_name(name: str) -> str:
    """Convert model name to safe filename."""
    return name.lower().replace(" ", "_")


def _plot_confusion_matrix(y_true, y_pred, class_names, model_name, results_dir):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = os.path.join(results_dir, f"{_safe_name(model_name)}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {path}")


def _plot_roc_curve(y_true, y_proba, n_classes, class_names, model_name, results_dir):
    """Generate and save per-class ROC curves (One-vs-Rest)."""
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{class_names[i]} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve (One-vs-Rest) — {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, f"{_safe_name(model_name)}_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved: {path}")


def _plot_feature_importance(model, feature_names, model_name, results_dir,
                             top_n: int = 20):
    """Generate and save feature importance bar chart for tree-based models."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    plt.barh(range(len(top_features)), top_importances[::-1],
             color=colors[::-1], edgecolor="gray", linewidth=0.5)
    plt.yticks(range(len(top_features)), top_features[::-1], fontsize=10)
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances — {model_name}",
              fontsize=14, fontweight="bold")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, f"{_safe_name(model_name)}_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance saved: {path}")


def _plot_model_comparison(scores_df: pd.DataFrame, results_dir: str):
    """Generate a grouped bar chart comparing all models across metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = scores_df.index.tolist()
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, metric in enumerate(metrics):
        values = scores_df[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i],
                      edgecolor="gray", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, 1.005)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Model comparison chart saved: {path}")


def generate_markdown_report(val_scores_df: pd.DataFrame,
                             test_scores_df: pd.DataFrame) -> str:
    """Generate a Markdown-formatted comparison report.

    Returns
    -------
    str
        Markdown text for the README.
    """
    lines = []
    lines.append("## Model Comparison Results")
    lines.append("")
    lines.append("### Validation Set Performance")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1-Score |")
    lines.append("|-------|----------|-----------|--------|----------|")

    best_model = val_scores_df["F1-Score"].idxmax()

    for model, row in val_scores_df.iterrows():
        marker = " **[BEST]**" if model == best_model else ""
        lines.append(
            f"| {model}{marker} | {row['Accuracy']:.6f} | {row['Precision']:.6f} "
            f"| {row['Recall']:.6f} | {row['F1-Score']:.6f} |"
        )

    lines.append("")
    lines.append("### Test Set Performance")
    lines.append("")
    lines.append("| Model | Accuracy | Precision | Recall | F1-Score |")
    lines.append("|-------|----------|-----------|--------|----------|")

    best_test = test_scores_df["F1-Score"].idxmax()

    for model, row in test_scores_df.iterrows():
        marker = " **[BEST]**" if model == best_test else ""
        lines.append(
            f"| {model}{marker} | {row['Accuracy']:.6f} | {row['Precision']:.6f} "
            f"| {row['Recall']:.6f} | {row['F1-Score']:.6f} |"
        )

    lines.append("")
    lines.append(f"**Best Model (Validation F1):** {best_model}")
    lines.append(f"**Best Model (Test F1):** {best_test}")

    return "\n".join(lines)
