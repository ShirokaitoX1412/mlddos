"""
shap_explainer.py - SHAP-based Explainability for DDoS Detection Models

Generates SHAP summary plots to explain WHY a traffic flow is classified
as a specific attack type. Supports tree-based models (RF, Extra Trees, XGBoost).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")


def load_model(model_name: str, models_dir: str = MODELS_DIR):
    """Load a saved model from pickle file."""
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(models_dir, f"{safe_name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_shap_plots(model, model_name: str, X_data: np.ndarray,
                         feature_names: list, label_map: dict,
                         results_dir: str = RESULTS_DIR,
                         max_samples: int = 500):
    """Generate SHAP summary plots for a trained model.

    Parameters
    ----------
    model : estimator
        Trained sklearn/xgboost model.
    model_name : str
        Display name of the model.
    X_data : np.ndarray
        Feature matrix (validation or test set).
    feature_names : list
        Names of features.
    label_map : dict
        Mapping from class index to class name.
    results_dir : str
        Directory to save SHAP plots.
    max_samples : int
        Max samples to use for SHAP (for speed).
    """
    os.makedirs(results_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    # Subsample for speed
    if len(X_data) > max_samples:
        idx = np.random.RandomState(42).choice(len(X_data), max_samples, replace=False)
        X_sample = X_data[idx]
    else:
        X_sample = X_data

    print(f"[SHAP] Computing SHAP values for {model_name} ({len(X_sample)} samples) ...")

    # Use TreeExplainer for tree-based models, KernelExplainer for others
    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "ExtraTreesClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # For MLP/KNN use KernelExplainer with a small background
        bg = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

    class_names = [label_map[i] for i in range(len(label_map))]

    # --- SHAP Summary Plot (bar) ---
    print(f"[SHAP] Generating bar summary plot ...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(results_dir, f"{safe_name}_shap_summary_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Bar plot saved: {bar_path}")

    # --- SHAP Summary Plot (dot/beeswarm) for each class ---
    for cls_idx, cls_name in label_map.items():
        print(f"[SHAP] Generating beeswarm plot for class: {cls_name} ...")
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            sv = shap_values[cls_idx]
        else:
            sv = shap_values[:, :, cls_idx] if shap_values.ndim == 3 else shap_values
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=15,
        )
        safe_cls = cls_name.lower().replace(" ", "_").replace("-", "_")
        plt.title(f"SHAP Values — {model_name} — {cls_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        dot_path = os.path.join(results_dir, f"{safe_name}_shap_{safe_cls}.png")
        plt.savefig(dot_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[SHAP] All SHAP plots saved for {model_name}.")
    return shap_values


def explain_single_prediction(model, model_name: str, sample: np.ndarray,
                               feature_names: list, label_map: dict,
                               results_dir: str = RESULTS_DIR):
    """Explain a single prediction with a SHAP force plot.

    Parameters
    ----------
    sample : np.ndarray
        Single feature vector (1D array).
    """
    os.makedirs(results_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")

    sample_2d = sample.reshape(1, -1)

    model_type = type(model).__name__
    if model_type in ("RandomForestClassifier", "ExtraTreesClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, sample_2d)

    shap_values = explainer.shap_values(sample_2d)

    pred_class = model.predict(sample_2d)[0]
    pred_name = label_map.get(pred_class, str(pred_class))

    if isinstance(shap_values, list):
        sv = shap_values[pred_class]
    else:
        sv = shap_values[:, :, pred_class] if shap_values.ndim == 3 else shap_values

    plt.figure(figsize=(16, 4))
    shap.force_plot(
        explainer.expected_value[pred_class] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value,
        sv[0],
        sample,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Force Plot — Predicted: {pred_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, f"{safe_name}_shap_force_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Force plot saved: {path}")


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessor import preprocess

    train_df, test_df = load_dataset()
    result = preprocess(train_df, test_df)

    model = load_model("Random Forest")
    generate_shap_plots(
        model, "Random Forest",
        result["X_val"], result["feature_names"], result["label_map"]
    )
    explain_single_prediction(
        model, "Random Forest",
        result["X_val"][0], result["feature_names"], result["label_map"]
    )
