"""Smoke tests for the trained IDS/IPS model.

Usage:
    python -m pytest tests/test_smoke_predict.py -v

The tests verify that a trained scikit-learn pipeline can be loaded and can
score one flow-like row without depending on any external network controller.
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))

MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_CANDIDATES = [
    MODELS_DIR / "selected_model.pkl",
    MODELS_DIR / "report_ready_best_model.pkl",
    MODELS_DIR / "notebook_best_binary_model.pkl",
]


def _find_model() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def _load_model(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def _get_feature_names(model) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    feature_names = getattr(preprocessor, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    pytest.skip("The saved model does not expose feature_names_in_. Re-train the model first.")


def _build_flow_like_sample(feature_names: list[str]) -> pd.DataFrame:
    row = {feature: 0.0 for feature in feature_names}
    defaults = {
        "Protocol": 6,
        "Flow Duration": 1_000_000,
        "Total Fwd Packets": 20,
        "Total Backward Packets": 10,
        "Fwd Packets Length Total": 1200,
        "Bwd Packets Length Total": 800,
        "Flow Bytes/s": 2000,
        "Flow Packets/s": 30,
        "SYN Flag Count": 1,
        "ACK Flag Count": 10,
    }
    for feature, value in defaults.items():
        if feature in row:
            row[feature] = value

    sample = pd.DataFrame([row], columns=feature_names)
    assert not sample.isna().any().any()
    assert np.isfinite(sample.select_dtypes(include=[np.number]).to_numpy()).all()
    return sample


@pytest.fixture(scope="module")
def model():
    path = _find_model()
    if path is None:
        pytest.skip("No trained model found. Run: python -m ml_ddos.main")
    return _load_model(path)


def test_model_predict_proba_smoke(model):
    feature_names = _get_feature_names(model)
    sample = _build_flow_like_sample(feature_names)

    proba = model.predict_proba(sample)
    pred = model.predict(sample)

    assert proba.shape[0] == 1
    assert len(pred) == 1
    assert np.isfinite(proba).all()
    assert abs(float(proba.sum()) - 1.0) < 1e-6


def test_traffic_generator_exists():
    generator = PROJECT_ROOT / "tools" / "ddos_traffic_generator.py"
    assert generator.exists()
