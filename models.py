"""
models.py - DDoS Detection Classifier Definitions

Defines the 5 machine learning classifiers used for multi-class
DDoS attack detection:
  1. Random Forest
  2. K-Nearest Neighbors (KNN)
  3. Extra Trees
  4. MLP Classifier (Neural Network)
  5. XGBoost

(Model training and evaluation will be implemented in Step 3.)
"""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_classifiers(n_classes: int = 7, random_state: int = 42) -> dict:
    """Return a dictionary of named classifiers.

    Parameters
    ----------
    n_classes : int
        Number of target classes (used for XGBoost configuration).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, estimator]
        Mapping of model name → sklearn-compatible estimator.
    """
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=10,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            objective="multi:softprob",
            num_class=n_classes,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
    }

    return classifiers
