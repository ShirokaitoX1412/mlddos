"""Build a clean report-ready DDoS detection notebook.

The generated notebook keeps the core pipeline needed for a graduation report:
load data, clean data, EDA, leakage checks, stratified split, imbalance-aware
training, evaluation, model comparison, and saving artifacts.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "ddos_detection_report_ready.ipynb"
MIRROR_NOTEBOOK_PATH = (
    PROJECT_ROOT.parents[1] / "mlddos" / "notebooks" / "ddos_detection_report_ready.ipynb"
)


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        """
        # DDoS Detection using Machine Learning - Two-VM IDS/IPS Ready

        Notebook này là bản rút gọn, sạch và phù hợp để đưa vào báo cáo đồ án,
        đồng thời xuất mô hình theo đúng định dạng để demo IDS/IPS hai m?y ?o sử dụng.
        Mục tiêu là xây dựng mô hình học máy phát hiện lưu lượng `Benign` và các
        nhóm tấn công DDoS trong bộ dữ liệu CICDDoS2019.

        Nguyên tắc chính:

        - Không đưa nhãn mục tiêu vào tập đặc trưng.
        - Không fit scaler, imputer hoặc bộ lọc tương quan trên toàn bộ dữ liệu trước khi chia train/test.
        - Chia dữ liệu bằng `stratify` để giữ phân bố lớp.
        - Đánh giá bằng các chỉ số phù hợp với dữ liệu mất cân bằng: `Balanced Accuracy`, `Macro F1`, `Weighted F1`, báo cáo từng lớp và ma trận nhầm lẫn.
        - Lưu best pipeline trực tiếp vào `saved_models/selected_model.pkl` để controller có thể gọi `predict_proba`.
        """
    ),
    md("## 1. Project Overview"),
    code(
        """
        from pathlib import Path
        import pickle
        import re
        import sys
        import time
        import warnings

        candidate_roots = [
            Path.cwd(),
            Path.cwd().parent,
            Path(r"D:\\source\\CyRadar\\VBoxShare\\mlddos"),
            Path("/mnt/d/source/CyRadar/VBoxShare/mlddos"),
        ]
        PROJECT_ROOT = next(
            (path for path in candidate_roots if (path / "backend" / "src").exists()),
            Path.cwd(),
        )
        BACKEND_SRC = PROJECT_ROOT / "backend" / "src"
        if str(BACKEND_SRC) not in sys.path:
            sys.path.insert(0, str(BACKEND_SRC))

        DATA_DIR = PROJECT_ROOT / "data"
        RESULTS_DIR = PROJECT_ROOT / "results"
        MODELS_DIR = PROJECT_ROOT / "saved_models"
        for directory in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
            directory.mkdir(parents=True, exist_ok=True)

        RANDOM_STATE = 42
        TEST_SIZE = 0.2
        CV_FOLDS = 3
        FAST_CV_MODE = True
        CV_MAX_PER_CLASS = 1000 if FAST_CV_MODE else 3000
        TRAIN_GAP_MAX_SAMPLES = 20000

        warnings.filterwarnings("ignore", category=FutureWarning)
        print("Project root:", PROJECT_ROOT)
        print("Dataset directory:", DATA_DIR)
        print("Results directory:", RESULTS_DIR)
        """
    ),
    md("## 2. Import Libraries"),
    code(
        """
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from IPython import get_ipython
        from IPython.display import display
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn.base import clone
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.metrics import (
            accuracy_score,
            auc,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            make_scorer,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
        from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import LabelEncoder, label_binarize
        from xgboost import XGBClassifier

        from ml_ddos.data_loader import KAGGLE_DATASET_SLUG, collect_file_paths, load_dataset
        from ml_ddos.models import (
            CappedUnderSamplingStrategy,
            LEAKAGE_COLUMN_PATTERNS,
            TrafficFeaturePreprocessor,
        )
        from ml_ddos.preprocessor import TARGET_COL, harmonize_labels, remove_duplicates

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid")
        print("Dataset:", KAGGLE_DATASET_SLUG)
        """
    ),
    md("## 3. Load Dataset"),
    code(
        """
        train_paths, test_paths = collect_file_paths(str(DATA_DIR))
        file_summary = pd.DataFrame(
            {
                "training_files": pd.Series([Path(path).name for path in train_paths]),
                "testing_files": pd.Series([Path(path).name for path in test_paths]),
            }
        )
        display(file_summary.head(20))

        train_raw, test_raw = load_dataset(str(DATA_DIR))
        print("Raw train shape:", train_raw.shape)
        print("Raw test shape:", test_raw.shape)
        display(train_raw.head())
        print("Columns:")
        print(list(train_raw.columns))

        if TARGET_COL not in train_raw.columns or TARGET_COL not in test_raw.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in train/test data.")
        """
    ),
    md("## 4. Data Cleaning"),
    code(
        """
        train_harmonized, test_harmonized = harmonize_labels(train_raw, test_raw)
        raw_combined_df = pd.concat([train_harmonized, test_harmonized], ignore_index=True)
        raw_class_distribution = (
            raw_combined_df[TARGET_COL]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="count")
        )
        raw_class_distribution["percent"] = (
            raw_class_distribution["count"] / raw_class_distribution["count"].sum() * 100
        )

        cleaning_summary = pd.DataFrame(
            [
                {
                    "split": "train",
                    "rows_before": len(train_harmonized),
                    "duplicates_before": int(train_harmonized.duplicated().sum()),
                },
                {
                    "split": "test",
                    "rows_before": len(test_harmonized),
                    "duplicates_before": int(test_harmonized.duplicated().sum()),
                },
            ]
        )

        train_clean = remove_duplicates(train_harmonized, "training source")
        test_clean = remove_duplicates(test_harmonized, "testing source")
        combined_df = pd.concat([train_clean, test_clean], ignore_index=True)
        combined_duplicates = int(combined_df.duplicated().sum())
        combined_df = combined_df.drop_duplicates().reset_index(drop=True)

        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        inf_count = int(np.isinf(combined_df[numeric_cols]).sum().sum()) if len(numeric_cols) else 0
        missing_count = int(combined_df.isna().sum().sum())

        cleaning_summary["rows_after_source_dedup"] = [len(train_clean), len(test_clean)]
        display(cleaning_summary)
        print("Combined shape after source dedup:", combined_df.shape)
        print("Duplicate rows removed after combining sources:", combined_duplicates)
        print("Total missing values:", missing_count)
        print("Total inf/-inf values in numeric columns:", inf_count)
        """
    ),
    md("## 5. Exploratory Data Analysis"),
    code(
        """
        class_distribution = (
            combined_df[TARGET_COL]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="count")
        )
        class_distribution["percent"] = (
            class_distribution["count"] / class_distribution["count"].sum() * 100
        )
        print("Class distribution before cleaning:")
        display(raw_class_distribution)
        print("Class distribution after cleaning:")
        display(class_distribution)

        binary_distribution = (
            combined_df[TARGET_COL]
            .map(lambda value: "Benign" if value == "Benign" else "Attack")
            .value_counts()
            .rename_axis("binary_class")
            .reset_index(name="count")
        )
        binary_distribution["percent"] = (
            binary_distribution["count"] / binary_distribution["count"].sum() * 100
        )
        display(binary_distribution)

        plt.figure(figsize=(10, 4))
        sns.barplot(data=class_distribution, x="class", y="count", color="#4c72b0")
        plt.title("Class distribution after cleaning")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

        missing_summary = (
            combined_df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False)
        )
        if missing_summary.empty:
            print("No missing values detected.")
        else:
            display(missing_summary.rename("missing_count").to_frame().head(20))
        """
    ),
    code(
        """
        numeric_features = combined_df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL], errors="ignore")
        if not numeric_features.empty:
            variance_top = numeric_features.var(numeric_only=True).sort_values(ascending=False).head(10).index.tolist()
            display(pd.DataFrame({"selected_numeric_feature": variance_top}))

            plot_cols = variance_top[:4]
            fig, axes = plt.subplots(1, len(plot_cols), figsize=(4 * len(plot_cols), 3))
            if len(plot_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, plot_cols):
                sns.histplot(combined_df[col].replace([np.inf, -np.inf], np.nan).dropna(), bins=40, ax=ax)
                ax.set_title(col)
            plt.tight_layout()
            plt.show()

            corr_cols = variance_top[:20]
            corr_matrix = (
                combined_df[corr_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(combined_df[corr_cols].median(numeric_only=True))
                .corr()
            )
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
            plt.title("Correlation heatmap of selected numeric features")
            plt.tight_layout()
            plt.show()
        else:
            print("No numeric feature columns found for EDA plots.")
        """
    ),
    md("## 6. Data Quality and Leakage Check"),
    code(
        """
        def find_leakage_like_columns(columns):
            found = []
            for col in columns:
                normalized = re.sub(r"[_\\-]+", " ", str(col).strip().lower())
                if any(re.search(pattern, normalized) for pattern in LEAKAGE_COLUMN_PATTERNS):
                    found.append(col)
            return found


        leakage_like_columns = [
            col for col in find_leakage_like_columns(combined_df.columns)
            if col != TARGET_COL
        ]
        constant_columns = [
            col for col in combined_df.columns
            if col != TARGET_COL and combined_df[col].nunique(dropna=False) <= 1
        ]
        high_unique_columns = [
            {
                "feature": col,
                "unique_count": int(combined_df[col].nunique(dropna=False)),
                "unique_ratio": float(combined_df[col].nunique(dropna=False) / len(combined_df)),
            }
            for col in combined_df.columns
            if col != TARGET_COL and combined_df[col].nunique(dropna=False) / len(combined_df) > 0.5
        ]

        leakage_check = pd.DataFrame(
            [
                {"check": "Target is not in feature matrix", "status": "checked before split"},
                {"check": "Duplicate rows removed before split", "status": f"{combined_duplicates} removed after combine"},
                {"check": "Leakage-like columns detected", "status": ", ".join(map(str, leakage_like_columns)) or "none"},
                {"check": "Constant columns detected", "status": str(len(constant_columns))},
                {"check": "Preprocessing fit scope", "status": "inside Pipeline, fit on train/CV fold only"},
            ]
        )
        display(leakage_check)
        display(pd.DataFrame(high_unique_columns).head(20))
        """
    ),
    md("## 7. Preprocessing and Stratified Split"),
    code(
        """
        X_all = combined_df.drop(columns=[TARGET_COL])
        y_text = combined_df[TARGET_COL].astype(str)

        X_train, X_test, y_train_text, y_test_text = train_test_split(
            X_all,
            y_text,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_text,
        )

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_text)
        y_test = label_encoder.transform(y_test_text)
        class_names = label_encoder.classes_.tolist()

        label_mapping = pd.DataFrame(
            {"encoded_label": range(len(class_names)), "original_label": class_names}
        )
        display(label_mapping)

        split_distribution = pd.DataFrame(
            {
                "original": y_text.value_counts().sort_index(),
                "train": y_train_text.value_counts().sort_index(),
                "test": y_test_text.value_counts().sort_index(),
            }
        )
        split_distribution["train_plus_test"] = split_distribution["train"] + split_distribution["test"]
        split_distribution["integrity_ok"] = split_distribution["original"] == split_distribution["train_plus_test"]
        display(split_distribution)

        if not split_distribution["integrity_ok"].all():
            raise ValueError("Split integrity check failed: train + test counts do not match original counts.")

        print("X_train:", X_train.shape, "X_test:", X_test.shape)
        print("Target included in X:", TARGET_COL in X_train.columns)
        """
    ),
    md("## 8. Imbalance Handling"),
    code(
        """
        train_class_counts = pd.Series(y_train).value_counts().sort_index()
        train_class_counts.index = [class_names[idx] for idx in train_class_counts.index]
        test_class_counts = pd.Series(y_test).value_counts().sort_index()
        test_class_counts.index = [class_names[idx] for idx in test_class_counts.index]

        imbalance_table = pd.DataFrame(
            {
                "train_count": train_class_counts,
                "test_count": test_class_counts,
            }
        )
        imbalance_table["train_percent"] = imbalance_table["train_count"] / imbalance_table["train_count"].sum() * 100
        imbalance_table["test_percent"] = imbalance_table["test_count"] / imbalance_table["test_count"].sum() * 100
        display(imbalance_table)

        minority_class_indices = [
            int(label)
            for label, count in pd.Series(y_train).value_counts().items()
            if count < 1000
        ]
        if not minority_class_indices:
            minority_class_indices = [int(pd.Series(y_train).value_counts().idxmin())]
        minority_class_names = [class_names[idx] for idx in minority_class_indices]
        print("Minority classes used for recall tracking:", minority_class_names)

        rare_warning = imbalance_table[imbalance_table["train_count"] < 100]
        if not rare_warning.empty:
            print("Warning: Some classes have very few training samples and may be unstable:")
            display(rare_warning)

        max_oversample_target = 2000
        oversample_strategy = {
            int(label): int(min(max_oversample_target, max(count, count * 4)))
            for label, count in pd.Series(y_train).value_counts().sort_index().items()
            if count < max_oversample_target
        }
        print("RandomOverSampler strategy:", oversample_strategy)
        print("RandomUnderSampler cap per majority class: 8000")
        """
    ),
    md("## 9. Model Training"),
    code(
        """
        def make_preprocessor():
            return TrafficFeaturePreprocessor(correlation_threshold=0.90)


        def make_pipeline(estimator, sampler=None):
            steps = [("preprocess", make_preprocessor())]
            if sampler is not None:
                steps.append(("sampler", sampler))
            steps.append(("model", estimator))
            return ImbPipeline(steps)


        def random_forest(class_weight=None):
            return RandomForestClassifier(
                n_estimators=60,
                max_depth=6,
                min_samples_split=400,
                min_samples_leaf=180,
                max_features="sqrt",
                max_samples=0.50,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )


        def extra_trees(class_weight=None):
            return ExtraTreesClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_split=150,
                min_samples_leaf=40,
                max_features="sqrt",
                max_samples=0.85,
                bootstrap=True,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )


        def knn_model():
            return KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                metric="minkowski",
                n_jobs=-1,
            )


        def xgboost_model():
            return XGBClassifier(
                objective="multi:softprob",
                num_class=len(class_names),
                eval_metric="mlogloss",
                n_estimators=120,
                learning_rate=0.06,
                max_depth=2,
                min_child_weight=25,
                subsample=0.65,
                colsample_bytree=0.65,
                reg_alpha=0.5,
                reg_lambda=12.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )


        def mlp_model():
            return MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=0.01,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.15,
                max_iter=300,
                random_state=RANDOM_STATE,
            )


        candidate_specs = [
            {
                "model": "Random Forest",
                "method": "baseline",
                "pipeline": make_pipeline(random_forest()),
            },
            {
                "model": "Random Forest",
                "method": "class_weight",
                "pipeline": make_pipeline(random_forest(class_weight="balanced_subsample")),
            },
            {
                "model": "Extra Trees",
                "method": "class_weight",
                "pipeline": make_pipeline(extra_trees(class_weight="balanced")),
            },
            {
                "model": "KNN",
                "method": "RandomUnderSampler",
                "pipeline": make_pipeline(
                    knn_model(),
                    RandomUnderSampler(
                        sampling_strategy=CappedUnderSamplingStrategy(cap=8000),
                        random_state=RANDOM_STATE,
                    ),
                ),
            },
            {
                "model": "XGBoost",
                "method": "baseline",
                "pipeline": make_pipeline(xgboost_model()),
            },
            {
                "model": "XGBoost",
                "method": "RandomUnderSampler",
                "pipeline": make_pipeline(
                    xgboost_model(),
                    RandomUnderSampler(
                        sampling_strategy=CappedUnderSamplingStrategy(cap=8000),
                        random_state=RANDOM_STATE,
                    ),
                ),
            },
            {
                "model": "MLP Classifier",
                "method": "RandomOverSampler",
                "pipeline": make_pipeline(
                    mlp_model(),
                    RandomOverSampler(
                        sampling_strategy=oversample_strategy,
                        random_state=RANDOM_STATE,
                    ),
                ),
            },
        ]

        print("Candidate models:", len(candidate_specs))
        display(pd.DataFrame([{k: v for k, v in spec.items() if k != "pipeline"} for spec in candidate_specs]))
        """
    ),
    code(
        """
        def make_cv_sample(X, y, max_per_class=CV_MAX_PER_CLASS):
            y_series = pd.Series(y, index=X.index)
            sampled_indices = []
            for label, label_indices in y_series.groupby(y_series).groups.items():
                label_indices = list(label_indices)
                if len(label_indices) > max_per_class:
                    sampled = pd.Series(label_indices).sample(
                        n=max_per_class,
                        random_state=RANDOM_STATE,
                    ).tolist()
                else:
                    sampled = label_indices
                sampled_indices.extend(sampled)
            sampled_indices = pd.Index(sampled_indices)
            return X.loc[sampled_indices], y_series.loc[sampled_indices].to_numpy()


        def minority_recall_score(y_true, y_pred):
            return recall_score(
                y_true,
                y_pred,
                labels=minority_class_indices,
                average="macro",
                zero_division=0,
            )


        def evaluate_candidate(spec, X_train, y_train, X_test, y_test):
            pipeline = clone(spec["pipeline"])
            start = time.time()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.time() - start

            train_eval_size = min(TRAIN_GAP_MAX_SAMPLES, len(X_train))
            train_eval_positions = np.random.RandomState(RANDOM_STATE).choice(
                len(X_train),
                size=train_eval_size,
                replace=False,
            )
            X_train_eval = X_train.iloc[train_eval_positions]
            y_train_eval = y_train[train_eval_positions]
            y_train_pred = pipeline.predict(X_train_eval)
            y_test_pred = pipeline.predict(X_test)
            y_test_proba = None
            if hasattr(pipeline, "predict_proba"):
                y_test_proba = pipeline.predict_proba(X_test)
            report = classification_report(
                y_test,
                y_test_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            benign_index = int(np.where(label_encoder.classes_ == "Benign")[0][0])
            benign_mask = y_test == benign_index
            attack_mask = y_test != benign_index
            false_alarm_rate = (
                float(np.mean(y_test_pred[benign_mask] != benign_index))
                if benign_mask.any()
                else np.nan
            )
            attack_recall = (
                float(np.mean(y_test_pred[attack_mask] != benign_index))
                if attack_mask.any()
                else np.nan
            )
            roc_auc_macro = np.nan
            roc_auc_weighted = np.nan
            if y_test_proba is not None:
                try:
                    roc_auc_macro = roc_auc_score(
                        y_test,
                        y_test_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                    roc_auc_weighted = roc_auc_score(
                        y_test,
                        y_test_proba,
                        multi_class="ovr",
                        average="weighted",
                    )
                except ValueError:
                    pass

            row = {
                "Model": spec["model"],
                "Imbalance Method": spec["method"],
                "Accuracy": accuracy_score(y_test, y_test_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_test_pred),
                "Macro F1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
                "Weighted F1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "Minority Class Recall": minority_recall_score(y_test, y_test_pred),
                "False Alarm Rate": false_alarm_rate,
                "Attack Recall": attack_recall,
                "ROC AUC Macro OvR": roc_auc_macro,
                "ROC AUC Weighted OvR": roc_auc_weighted,
                "Train Macro F1": f1_score(y_train_eval, y_train_pred, average="macro", zero_division=0),
                "Test Macro F1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
                "Train/Test Gap": f1_score(y_train_eval, y_train_pred, average="macro", zero_division=0)
                - f1_score(y_test, y_test_pred, average="macro", zero_division=0),
                "Fit Seconds": fit_seconds,
            }

            for class_name in class_names:
                row[f"{class_name} Recall"] = report.get(class_name, {}).get("recall", 0.0)
                row[f"{class_name} F1"] = report.get(class_name, {}).get("f1-score", 0.0)

            tcp_syn_f1 = row.get("TCP SYN Flood F1", np.nan)
            udp_lag_f1 = row.get("UDP-Lag Flood F1", np.nan)
            notes = []
            if not np.isnan(tcp_syn_f1) and tcp_syn_f1 < 0.50:
                notes.append("TCP SYN F1 low")
            if not np.isnan(udp_lag_f1) and udp_lag_f1 < 0.20:
                notes.append("UDP-Lag F1 low")
            row["Notes"] = "; ".join(notes) if notes else "OK"
            return pipeline, row, y_test_pred, y_test_proba, report


        X_cv, y_cv = make_cv_sample(X_train, y_train)
        print("CV sample shape:", X_cv.shape)
        print("FAST_CV_MODE:", FAST_CV_MODE, "| CV_MAX_PER_CLASS:", CV_MAX_PER_CLASS)
        print("Train gap evaluation max samples:", TRAIN_GAP_MAX_SAMPLES)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scoring = {
            "macro_f1": "f1_macro",
            "weighted_f1": "f1_weighted",
            "balanced_accuracy": "balanced_accuracy",
            "minority_recall": make_scorer(minority_recall_score),
        }

        fitted_models = {}
        predictions = {}
        probabilities = {}
        reports = {}
        comparison_rows = []

        for spec in candidate_specs:
            print(f"Training {spec['model']} | {spec['method']} ...")
            cv_result = cross_validate(
                clone(spec["pipeline"]),
                X_cv,
                y_cv,
                cv=cv,
                scoring=cv_scoring,
                n_jobs=1,
                error_score="raise",
            )
            fitted_pipeline, row, y_pred, y_proba, report = evaluate_candidate(
                spec,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            row["CV Macro F1 Mean"] = cv_result["test_macro_f1"].mean()
            row["CV Macro F1 Std"] = cv_result["test_macro_f1"].std()
            row["CV Weighted F1 Mean"] = cv_result["test_weighted_f1"].mean()
            row["CV Balanced Accuracy Mean"] = cv_result["test_balanced_accuracy"].mean()
            row["CV Minority Recall Mean"] = cv_result["test_minority_recall"].mean()

            model_key = f"{spec['model']} | {spec['method']}"
            fitted_models[model_key] = fitted_pipeline
            predictions[model_key] = y_pred
            probabilities[model_key] = y_proba
            reports[model_key] = report
            comparison_rows.append(row)

        comparison_df = pd.DataFrame(comparison_rows)

        guardrail_penalty = (
            comparison_df["Notes"].str.contains("TCP SYN F1 low", regex=False).astype(float) * 0.50
            + comparison_df["Notes"].str.contains("UDP-Lag F1 low", regex=False).astype(float) * 0.40
        )
        comparison_df["Selection Score"] = (
            comparison_df["Macro F1"]
            + 0.50 * comparison_df["Minority Class Recall"]
            + 0.25 * comparison_df["Balanced Accuracy"]
            - 0.25 * comparison_df["Train/Test Gap"].clip(lower=0)
            - guardrail_penalty
        )

        main_columns = [
            "Model",
            "Imbalance Method",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
            "Weighted F1",
            "Minority Class Recall",
            "False Alarm Rate",
            "Attack Recall",
            "ROC AUC Macro OvR",
            "Train Macro F1",
            "Test Macro F1",
            "Train/Test Gap",
            "CV Macro F1 Mean",
            "CV Macro F1 Std",
            "TCP SYN Flood F1",
            "UDP-Lag Flood F1",
            "Notes",
            "Selection Score",
            "Fit Seconds",
        ]
        display(comparison_df[main_columns].sort_values("Selection Score", ascending=False))
        """
    ),
    md("## 10. Evaluation of the Best Model"),
    code(
        """
        best_row = comparison_df.sort_values("Selection Score", ascending=False).iloc[0]
        best_model_key = f"{best_row['Model']} | {best_row['Imbalance Method']}"
        best_model = fitted_models[best_model_key]
        y_pred_best = predictions[best_model_key]
        y_proba_best = probabilities[best_model_key]
        best_report = reports[best_model_key]

        print("Best model:", best_model_key)
        print(best_row[main_columns].to_string())

        report_df = pd.DataFrame(best_report).transpose()
        display(report_df)

        cm = confusion_matrix(y_test, y_pred_best, labels=list(range(len(class_names))))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        display(cm_df)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion matrix - {best_model_key}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.show()

        y_pred_counts = pd.Series(y_pred_best).value_counts().sort_index()
        y_pred_counts.index = [class_names[idx] for idx in y_pred_counts.index]
        display(y_pred_counts.rename("predicted_count").to_frame())

        if best_row["Test Macro F1"] > 0.98:
            print("Warning: Test Macro F1 is very high. The dataset may be easy or hidden leakage may still exist.")
        if best_row["Train/Test Gap"] > 0.05:
            print("Warning: Train/Test Macro F1 gap is large, indicating overfitting risk.")
        if best_row["Notes"] != "OK":
            print("Warning:", best_row["Notes"])
        """
    ),
    md("## 11. Model Comparison"),
    code(
        """
        comparison_plot_df = comparison_df.sort_values("Selection Score", ascending=False).copy()
        display(comparison_plot_df[main_columns])

        plt.figure(figsize=(10, 4))
        sns.barplot(
            data=comparison_plot_df,
            x="Model",
            y="Macro F1",
            hue="Imbalance Method",
        )
        plt.title("Macro F1 comparison")
        plt.ylim(0, 1)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        sns.barplot(
            data=comparison_plot_df,
            x="Model",
            y="Balanced Accuracy",
            hue="Imbalance Method",
        )
        plt.title("Balanced Accuracy comparison")
        plt.ylim(0, 1)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## 12. Per-model Classification Reports

        Cell này tạo báo cáo phân loại chi tiết cho từng mô hình theo định dạng log,
        thuận tiện để sao chép vào báo cáo đồ án.
        """
    ),
    code(
        """
        from datetime import datetime

        per_model_report_lines = []
        per_model_report_dir = RESULTS_DIR / "per_model_reports"
        per_model_report_dir.mkdir(parents=True, exist_ok=True)

        for _, model_row in comparison_df.sort_values("Selection Score", ascending=False).iterrows():
            model_key = f"{model_row['Model']} | {model_row['Imbalance Method']}"
            y_pred_model = predictions[model_key]
            model_accuracy = accuracy_score(y_test, y_pred_model)
            false_alarm = float(model_row["False Alarm Rate"])
            attack_recall_value = float(model_row["Attack Recall"])
            roc_auc_value = float(model_row["ROC AUC Macro OvR"])
            report_text = classification_report(
                y_test,
                y_pred_model,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
            timestamp_accuracy = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            timestamp_report = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            block = (
                f"{timestamp_accuracy} - INFO - Độ chính xác (Accuracy) trên tập test: {model_accuracy:.4f}\\n"
                f"{timestamp_accuracy} - INFO - Tỉ lệ báo động giả (False Alarm Rate): {false_alarm:.4f}\\n"
                f"{timestamp_accuracy} - INFO - Recall phát hiện tấn công tổng quát: {attack_recall_value:.4f}\\n"
                f"{timestamp_accuracy} - INFO - ROC-AUC đa lớp macro OvR: {roc_auc_value:.4f}\\n"
                f"{timestamp_report} - INFO - Báo cáo phân loại cho {model_key}:\\n"
                f"{report_text}\\n"
                + "=" * 90
                + "\\n"
            )
            print(block)
            per_model_report_lines.append(block)

            safe_model_name = (
                model_key.lower()
                .replace(" | ", "_")
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
            )
            (per_model_report_dir / f"{safe_model_name}_classification_report.txt").write_text(
                block,
                encoding="utf-8",
            )

        all_reports_path = RESULTS_DIR / "report_ready_per_model_classification_reports.txt"
        all_reports_path.write_text("\\n".join(per_model_report_lines), encoding="utf-8")
        print("Saved all per-model reports:", all_reports_path)
        print("Saved individual reports dir:", per_model_report_dir)

        exact_one_rows = []
        for model_key, report in reports.items():
            for label_name in class_names:
                label_report = report.get(label_name, {})
                for metric_name in ["precision", "recall", "f1-score"]:
                    metric_value = float(label_report.get(metric_name, np.nan))
                    if metric_value == 1.0:
                        exact_one_rows.append(
                            {
                                "Model": model_key,
                                "Class": label_name,
                                "Metric": metric_name,
                                "Value": metric_value,
                                "Note": "Metric is exactly 1.0 on the current test split",
                            }
                        )
        exact_one_metric_audit = pd.DataFrame(exact_one_rows)
        if exact_one_metric_audit.empty:
            print("No exact 1.0 precision/recall/f1-score values found in per-class reports.")
        else:
            display(exact_one_metric_audit)
        exact_one_metric_audit.to_csv(RESULTS_DIR / "report_ready_exact_one_metric_audit.csv", index=False)
        """
    ),
    md(
        """
        ## 13. Expected Results Coverage Check

        Cell này kiểm tra các kết quả dự kiến trong báo cáo so với phạm vi dữ liệu
        và code thực tế. Những lớp tấn công không có trong dataset hiện tại sẽ
        không được tuyên bố là đã phân loại trong kết quả thực nghiệm.
        """
    ),
    code(
        """
        expected_attack_labels = [
            "UDP Flood",
            "ICMP Flood",
            "TCP SYN Flood",
            "Ping of Death",
            "HTTP Flood",
            "LDAP Flood",
            "MSSQL Flood",
            "NetBIOS Flood",
            "UDP-Lag Flood",
        ]
        observed_labels = set(class_names)
        expected_coverage_df = pd.DataFrame(
            [
                {
                    "Kết quả dự kiến": "Xây dựng mô hình phát hiện DDoS bằng học máy",
                    "Trạng thái trong code": "Đã có",
                    "Bằng chứng": "Có pipeline huấn luyện, so sánh model và selected_model.pkl",
                },
                {
                    "Kết quả dự kiến": "Kiểm soát tỉ lệ báo động giả",
                    "Trạng thái trong code": "Đã bổ sung",
                    "Bằng chứng": "False Alarm Rate được tính cho từng model",
                },
                {
                    "Kết quả dự kiến": "Phân loại các dạng DDoS có trong dataset",
                    "Trạng thái trong code": "Đã có trong phạm vi dataset",
                    "Bằng chứng": ", ".join(class_names),
                },
                {
                    "Kết quả dự kiến": "Trích xuất đặc trưng từ dữ liệu thô",
                    "Trạng thái trong code": "Một phần",
                    "B?ng ch?ng": "Dataset hi?n d?ng flow-level features ?? tr?ch xu?t s?n; demo hai m?y ?o sinh l?u l??ng ki?m th? v? h? th?ng IDS/IPS ph?n lo?i theo ??c tr?ng lu?ng",
                },
                {
                    "Kết quả dự kiến": "Trực quan hóa phân bố, confusion matrix, ROC",
                    "Trạng thái trong code": "Đã bổ sung",
                    "Bằng chứng": "report_figures chứa biểu đồ phân bố, so sánh model, confusion matrix và ROC",
                },
            ]
        )
        display(expected_coverage_df)

        attack_label_coverage_df = pd.DataFrame(
            [
                {
                    "Dạng tấn công trong mô tả dự kiến": label,
                    "Có trong dataset/code hiện tại": label in observed_labels,
                    "Ghi chú": (
                        "Được đánh giá trong classification report"
                        if label in observed_labels
                        else "Không có nhãn trong dataset hiện tại, không nên ghi là đã phân loại"
                    ),
                }
                for label in expected_attack_labels
            ]
        )
        display(attack_label_coverage_df)

        expected_coverage_df.to_csv(RESULTS_DIR / "report_ready_expected_results_coverage.csv", index=False)
        attack_label_coverage_df.to_csv(RESULTS_DIR / "report_ready_attack_label_coverage.csv", index=False)
        """
    ),
    md("## 14. Save Results"),
    code(
        """
        comparison_path = RESULTS_DIR / "report_ready_model_comparison.csv"
        report_path = RESULTS_DIR / "report_ready_classification_report.csv"
        cm_path = RESULTS_DIR / "report_ready_confusion_matrix.csv"
        summary_path = RESULTS_DIR / "report_ready_final_summary.csv"
        model_path = MODELS_DIR / "report_ready_best_model.pkl"
        selected_model_path = MODELS_DIR / "selected_model.pkl"
        metadata_path = RESULTS_DIR / "report_ready_selected_model_metadata.csv"

        comparison_df.to_csv(comparison_path, index=False)
        report_df.to_csv(report_path)
        cm_df.to_csv(cm_path)

        final_summary = pd.DataFrame(
            [
                {
                    "total_rows": len(combined_df),
                    "total_features_before_preprocessing": X_all.shape[1],
                    "duplicates_removed_after_combine": combined_duplicates,
                    "missing_values": missing_count,
                    "inf_values": inf_count,
                    "best_model": best_model_key,
                    "accuracy": best_row["Accuracy"],
                    "balanced_accuracy": best_row["Balanced Accuracy"],
                    "macro_f1": best_row["Macro F1"],
                    "weighted_f1": best_row["Weighted F1"],
                    "minority_recall": best_row["Minority Class Recall"],
                    "false_alarm_rate": best_row["False Alarm Rate"],
                    "attack_recall": best_row["Attack Recall"],
                    "roc_auc_macro_ovr": best_row["ROC AUC Macro OvR"],
                    "train_test_gap": best_row["Train/Test Gap"],
                    "cv_macro_f1_mean": best_row["CV Macro F1 Mean"],
                    "cv_macro_f1_std": best_row["CV Macro F1 Std"],
                    "notes": best_row["Notes"],
                }
            ]
        )
        final_summary.to_csv(summary_path, index=False)
        display(final_summary)

        # demo IDS/IPS hai m?y ?o expects the pickle object itself to expose
        # predict_proba(features). Therefore selected_model.pkl stores the
        # fitted Pipeline directly, not a metadata dictionary.
        with open(model_path, "wb") as file:
            pickle.dump(best_model, file)
        with open(selected_model_path, "wb") as file:
            pickle.dump(best_model, file)

        selected_metadata = pd.DataFrame(
            [
                {
                    "selected_model_file": str(selected_model_path),
                    "best_model": best_model_key,
                    "class_names": "|".join(class_names),
                    "accuracy": best_row["Accuracy"],
                    "balanced_accuracy": best_row["Balanced Accuracy"],
                    "macro_f1": best_row["Macro F1"],
                    "weighted_f1": best_row["Weighted F1"],
                    "minority_recall": best_row["Minority Class Recall"],
                    "false_alarm_rate": best_row["False Alarm Rate"],
                    "attack_recall": best_row["Attack Recall"],
                    "roc_auc_macro_ovr": best_row["ROC AUC Macro OvR"],
                    "cv_macro_f1_mean": best_row["CV Macro F1 Mean"],
                    "cv_macro_f1_std": best_row["CV Macro F1 Std"],
                }
            ]
        )
        selected_metadata.to_csv(metadata_path, index=False)

        print("Saved comparison:", comparison_path)
        print("Saved classification report:", report_path)
        print("Saved confusion matrix:", cm_path)
        print("Saved summary:", summary_path)
        print("Saved best model:", model_path)
        print("Saved selected IDS/IPS model:", selected_model_path)
        print("Saved selected model metadata:", metadata_path)
        """
    ),
    md("## 15. IDS/IPS Runtime Compatibility Check"),
    code(
        """
        with open(selected_model_path, "rb") as file:
            runtime_model = pickle.load(file)

        runtime_sample = X_test.head(1).copy()
        proba = runtime_model.predict_proba(runtime_sample)[0]
        pred_index = int(np.argmax(proba))
        runtime_check = pd.DataFrame(
            [
                {
                    "selected_model_path": str(selected_model_path),
                    "predict_proba_ok": True,
                    "predicted_index": pred_index,
                    "predicted_label": class_names[pred_index],
                    "confidence": float(proba[pred_index]),
                    "input_feature_count": runtime_sample.shape[1],
                }
            ]
        )
        display(runtime_check)
        """
    ),
    md(
        """
        ## 16. Report Figures

        Cell này sinh các biểu đồ chính dùng để chèn vào báo cáo tốt nghiệp.
        Tất cả biểu đồ được tạo từ dữ liệu và kết quả thực nghiệm thật trong notebook,
        đồng thời được lưu thành ảnh PNG để có thể chèn trực tiếp vào Word.
        """
    ),
    code(
        """
        REPORT_FIGURES_DIR = RESULTS_DIR / "report_figures"
        REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


        def save_report_figure(filename: str) -> Path:
            path = REPORT_FIGURES_DIR / filename
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.show()
            print("Saved:", path)
            return path


        saved_figures = []

        # Hình 4.1: Phân bố lớp trong bộ dữ liệu trước xử lý.
        plt.figure(figsize=(10, 5))
        raw_class_plot_df = raw_class_distribution.copy()
        sns.barplot(data=raw_class_plot_df, x="class", y="count", color="#9A6FB0")
        plt.title("Hình 4.1. Phân bố lớp dữ liệu trước xử lý")
        plt.xlabel("Lớp lưu lượng")
        plt.ylabel("Số lượng mẫu")
        plt.xticks(rotation=35, ha="right")
        for index, row in raw_class_plot_df.iterrows():
            plt.text(index, row["count"], f"{int(row['count']):,}", ha="center", va="bottom", fontsize=8)
        saved_figures.append(save_report_figure("hinh_4_1_phan_bo_lop_truoc_xu_ly.png"))

        # Hình 4.2: Phân bố lớp trong bộ dữ liệu sau xử lý.
        plt.figure(figsize=(10, 5))
        class_plot_df = class_distribution.copy()
        sns.barplot(data=class_plot_df, x="class", y="count", color="#4C72B0")
        plt.title("Hình 4.2. Phân bố lớp dữ liệu sau xử lý")
        plt.xlabel("Lớp lưu lượng")
        plt.ylabel("Số lượng mẫu")
        plt.xticks(rotation=35, ha="right")
        for index, row in class_plot_df.iterrows():
            plt.text(index, row["count"], f"{int(row['count']):,}", ha="center", va="bottom", fontsize=8)
        saved_figures.append(save_report_figure("hinh_4_2_phan_bo_lop_sau_xu_ly.png"))

        # Hình 4.3: Phân bố nhị phân Benign/Attack.
        plt.figure(figsize=(6, 4))
        binary_plot_df = binary_distribution.copy()
        sns.barplot(data=binary_plot_df, x="binary_class", y="count", palette=["#55A868", "#C44E52"])
        plt.title("Hình 4.3. Phân bố nhị phân giữa Benign và Attack")
        plt.xlabel("Nhóm lưu lượng")
        plt.ylabel("Số lượng mẫu")
        for index, row in binary_plot_df.iterrows():
            plt.text(index, row["count"], f"{int(row['count']):,}", ha="center", va="bottom", fontsize=9)
        saved_figures.append(save_report_figure("hinh_4_3_phan_bo_benign_attack.png"))

        # Hình 4.4: Phân bố lớp giữa tập huấn luyện và tập kiểm thử.
        split_reset_df = split_distribution[["train", "test"]].reset_index()
        split_reset_df = split_reset_df.rename(columns={split_reset_df.columns[0]: "class"})
        split_plot_df = split_reset_df.melt(
            id_vars="class",
            var_name="Tập dữ liệu",
            value_name="Số lượng mẫu",
        )
        split_plot_df["Tập dữ liệu"] = split_plot_df["Tập dữ liệu"].map(
            {"train": "Tập huấn luyện", "test": "Tập kiểm thử"}
        )
        plt.figure(figsize=(11, 5))
        sns.barplot(data=split_plot_df, x="class", y="Số lượng mẫu", hue="Tập dữ liệu")
        plt.title("Hình 4.4. Phân bố lớp giữa tập huấn luyện và tập kiểm thử")
        plt.xlabel("Lớp lưu lượng")
        plt.ylabel("Số lượng mẫu")
        plt.xticks(rotation=35, ha="right")
        saved_figures.append(save_report_figure("hinh_4_4_phan_bo_train_test.png"))

        # Hình 4.5: So sánh các metric chính giữa các mô hình.
        comparison_for_plot = comparison_df.copy()
        comparison_for_plot["Mô hình"] = (
            comparison_for_plot["Model"] + " - " + comparison_for_plot["Imbalance Method"]
        )
        metrics_to_plot = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"]
        metrics_plot_df = comparison_for_plot.melt(
            id_vars="Mô hình",
            value_vars=metrics_to_plot,
            var_name="Chỉ số",
            value_name="Giá trị",
        )
        plt.figure(figsize=(12, 5))
        sns.barplot(data=metrics_plot_df, x="Mô hình", y="Giá trị", hue="Chỉ số")
        plt.title("Hình 4.5. So sánh các chỉ số đánh giá giữa các mô hình")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("Giá trị")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        saved_figures.append(save_report_figure("hinh_4_5_so_sanh_chi_so_mo_hinh.png"))

        # So sánh Macro F1 giữa các mô hình.
        plt.figure(figsize=(10, 4))
        macro_plot_df = comparison_for_plot.sort_values("Macro F1", ascending=False)
        sns.barplot(data=macro_plot_df, x="Mô hình", y="Macro F1", color="#8172B2")
        plt.title("Hình 4.6. So sánh Macro F1 giữa các mô hình")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("Macro F1")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        saved_figures.append(save_report_figure("hinh_4_6_so_sanh_macro_f1.png"))

        # So sánh Balanced Accuracy giữa các mô hình.
        plt.figure(figsize=(10, 4))
        balanced_plot_df = comparison_for_plot.sort_values("Balanced Accuracy", ascending=False)
        sns.barplot(data=balanced_plot_df, x="Mô hình", y="Balanced Accuracy", color="#64B5CD")
        plt.title("Hình 4.7. So sánh Balanced Accuracy giữa các mô hình")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("Balanced Accuracy")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        saved_figures.append(save_report_figure("hinh_4_7_so_sanh_balanced_accuracy.png"))

        # Confusion matrix của mô hình tốt nhất.
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
        plt.title(f"Hình 4.8. Ma trận nhầm lẫn của mô hình tốt nhất: {best_model_key}")
        plt.xlabel("Nhãn dự đoán")
        plt.ylabel("Nhãn thực tế")
        saved_figures.append(save_report_figure("hinh_4_8_ma_tran_nham_lan_best_model.png"))

        # Precision/Recall/F1 theo từng lớp của mô hình tốt nhất.
        per_class_report = report_df.loc[
            [label for label in class_names if label in report_df.index],
            ["precision", "recall", "f1-score"],
        ].reset_index().rename(columns={"index": "Lớp"})
        per_class_plot_df = per_class_report.melt(
            id_vars="Lớp",
            var_name="Chỉ số",
            value_name="Giá trị",
        )
        plt.figure(figsize=(11, 5))
        sns.barplot(data=per_class_plot_df, x="Lớp", y="Giá trị", hue="Chỉ số")
        plt.title("Hình 4.9. Precision, Recall và F1-score theo từng lớp")
        plt.xlabel("Lớp lưu lượng")
        plt.ylabel("Giá trị")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=35, ha="right")
        saved_figures.append(save_report_figure("hinh_4_9_classification_report_theo_lop.png"))

        # Train/Test Gap để phân tích overfitting.
        plt.figure(figsize=(10, 4))
        gap_plot_df = comparison_for_plot.sort_values("Train/Test Gap", ascending=False)
        sns.barplot(data=gap_plot_df, x="Mô hình", y="Train/Test Gap", color="#DD8452")
        plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="Ngưỡng cảnh báo 0.05")
        plt.axhline(0.00, color="black", linestyle="-", linewidth=0.8)
        plt.title("Hình 4.10. So sánh Train/Test Gap giữa các mô hình")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("Train/Test Macro F1 Gap")
        plt.xticks(rotation=25, ha="right")
        plt.legend()
        saved_figures.append(save_report_figure("hinh_4_10_train_test_gap.png"))

        # Cross-validation Macro F1 mean/std.
        plt.figure(figsize=(10, 4))
        cv_plot_df = comparison_for_plot.sort_values("CV Macro F1 Mean", ascending=False)
        plt.bar(
            cv_plot_df["Mô hình"],
            cv_plot_df["CV Macro F1 Mean"],
            yerr=cv_plot_df["CV Macro F1 Std"],
            capsize=5,
            color="#55A868",
        )
        plt.title("Hình 4.11. Kết quả cross-validation Macro F1")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("CV Macro F1 Mean")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        saved_figures.append(save_report_figure("hinh_4_11_cross_validation_macro_f1.png"))

        # Minority Recall giữa các mô hình.
        plt.figure(figsize=(10, 4))
        minority_plot_df = comparison_for_plot.sort_values("Minority Class Recall", ascending=False)
        sns.barplot(data=minority_plot_df, x="Mô hình", y="Minority Class Recall", color="#C44E52")
        plt.title("Hình 4.12. So sánh Recall của nhóm lớp thiểu số")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("Minority Class Recall")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        saved_figures.append(save_report_figure("hinh_4_12_minority_recall.png"))

        # False Alarm Rate giữa các mô hình.
        plt.figure(figsize=(10, 4))
        false_alarm_plot_df = comparison_for_plot.sort_values("False Alarm Rate", ascending=True)
        sns.barplot(data=false_alarm_plot_df, x="Mô hình", y="False Alarm Rate", color="#E17C05")
        plt.title("Hình 4.13. So sánh tỉ lệ báo động giả giữa các mô hình")
        plt.xlabel("Mô hình và phương án xử lý mất cân bằng")
        plt.ylabel("False Alarm Rate")
        plt.ylim(0, max(0.05, false_alarm_plot_df["False Alarm Rate"].max() * 1.2))
        plt.xticks(rotation=25, ha="right")
        saved_figures.append(save_report_figure("hinh_4_13_false_alarm_rate.png"))

        # ROC nhị phân: Benign và Attack.
        benign_index = int(np.where(label_encoder.classes_ == "Benign")[0][0])
        y_binary_attack = (y_test != benign_index).astype(int)
        attack_score = 1.0 - y_proba_best[:, benign_index]
        fpr_binary, tpr_binary, _ = roc_curve(y_binary_attack, attack_score)
        binary_auc = auc(fpr_binary, tpr_binary)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_binary, tpr_binary, label=f"AUC = {binary_auc:.4f}", color="#4C72B0")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.title("Hình 4.14. Đường cong ROC nhị phân Benign/Attack")
        plt.xlabel("Tỉ lệ báo động giả")
        plt.ylabel("Tỉ lệ phát hiện đúng")
        plt.legend(loc="lower right")
        saved_figures.append(save_report_figure("hinh_4_14_roc_binary_benign_attack.png"))

        # ROC đa lớp One-vs-Rest cho mô hình tốt nhất.
        y_test_binarized = label_binarize(y_test, classes=list(range(len(class_names))))
        plt.figure(figsize=(8, 6))
        for class_idx, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, class_idx], y_proba_best[:, class_idx])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=1.5, label=f"{class_name} (AUC={class_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.title(f"Hình 4.15. Đường cong ROC đa lớp của mô hình tốt nhất: {best_model_key}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right", fontsize=8)
        saved_figures.append(save_report_figure("hinh_4_15_roc_multiclass_best_model.png"))

        # ROC da lop One-vs-Rest cho tat ca mo hinh.
        plt.figure(figsize=(12, 9))
        model_palette = sns.color_palette("tab10", n_colors=max(1, len(probabilities)))
        class_linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
        auc_by_model_key = {
            f"{row['Model']} | {row['Imbalance Method']}": row["ROC AUC Macro OvR"]
            for _, row in comparison_df.iterrows()
        }
        for model_idx, (model_key, model_proba) in enumerate(probabilities.items()):
            if model_proba is None:
                continue
            color = model_palette[model_idx % len(model_palette)]
            for class_idx, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, class_idx], model_proba[:, class_idx])
                class_auc = auc(fpr, tpr)
                plt.plot(
                    fpr,
                    tpr,
                    linewidth=1.2,
                    alpha=0.78,
                    color=color,
                    linestyle=class_linestyles[class_idx % len(class_linestyles)],
                    label=f"{model_key} - {class_name} (AUC={class_auc:.3f})",
                )
            model_auc = auc_by_model_key.get(model_key, np.nan)
            if not np.isnan(model_auc):
                plt.plot([], [], color=color, linewidth=3, label=f"{model_key} macro AUC={model_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Du doan ngau nhien")
        plt.title("Hình 4.16. Đường cong ROC đa lớp cho tất cả mô hình")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.2)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
        saved_figures.append(save_report_figure("hinh_4_16_roc_all_models_multiclass.png"))

        # Hình 4.17: So sánh phân phối nhãn thực tế và nhãn dự đoán.
        actual_distribution = pd.Series(
            label_encoder.inverse_transform(y_test),
            name="Nhãn thực tế",
        ).value_counts().reindex(class_names, fill_value=0)
        predicted_distribution = pd.Series(
            label_encoder.inverse_transform(y_pred_best),
            name="Nhãn dự đoán",
        ).value_counts().reindex(class_names, fill_value=0)
        actual_pred_plot_df = pd.DataFrame(
            {
                "Lớp lưu lượng": class_names,
                "Thực tế": actual_distribution.values,
                "Dự đoán": predicted_distribution.values,
            }
        ).melt(
            id_vars="Lớp lưu lượng",
            var_name="Loại nhãn",
            value_name="Số lượng mẫu",
        )
        plt.figure(figsize=(11, 5))
        sns.barplot(
            data=actual_pred_plot_df,
            x="Lớp lưu lượng",
            y="Số lượng mẫu",
            hue="Loại nhãn",
            palette=["#4C72B0", "#DD8452"],
        )
        plt.title(f"Hình 4.17. So sánh phân phối nhãn thực tế và nhãn dự đoán của {best_model_key}")
        plt.xlabel("Lớp lưu lượng")
        plt.ylabel("Số lượng mẫu")
        plt.xticks(rotation=35, ha="right")
        saved_figures.append(save_report_figure("hinh_4_17_phan_phoi_thuc_te_du_doan.png"))

        figures_table = pd.DataFrame(
            {
                "Tên hình": [
                    "Hình 4.1. Phân bố lớp dữ liệu trước xử lý",
                    "Hình 4.2. Phân bố lớp dữ liệu sau xử lý",
                    "Hình 4.3. Phân bố nhị phân giữa Benign và Attack",
                    "Hình 4.4. Phân bố lớp giữa tập huấn luyện và tập kiểm thử",
                    "Hình 4.5. So sánh các chỉ số đánh giá giữa các mô hình",
                    "Hình 4.6. So sánh Macro F1 giữa các mô hình",
                    "Hình 4.7. So sánh Balanced Accuracy giữa các mô hình",
                    "Hình 4.8. Ma trận nhầm lẫn của mô hình tốt nhất",
                    "Hình 4.9. Precision, Recall và F1-score theo từng lớp",
                    "Hình 4.10. So sánh Train/Test Gap giữa các mô hình",
                    "Hình 4.11. Kết quả cross-validation Macro F1",
                    "Hình 4.12. So sánh Recall của nhóm lớp thiểu số",
                    "Hình 4.13. So sánh tỉ lệ báo động giả giữa các mô hình",
                    "Hình 4.14. Đường cong ROC nhị phân Benign/Attack",
                    "Hình 4.15. Đường cong ROC đa lớp của mô hình tốt nhất",
                    "Hình 4.16. Đường cong ROC đa lớp cho tất cả mô hình",
                    "Hình 4.17. So sánh phân phối nhãn thực tế và nhãn dự đoán",
                ],
                "File ảnh": [str(path) for path in saved_figures],
            }
        )
        display(figures_table)
        figures_table.to_csv(REPORT_FIGURES_DIR / "danh_muc_hinh_sinh_tu_notebook.csv", index=False)
        """
    ),
    md("## 17. Final Conclusion"),
    code(
        """
        conclusion = f'''
        Best model: {best_model_key}
        Accuracy: {best_row['Accuracy']:.4f}
        Balanced Accuracy: {best_row['Balanced Accuracy']:.4f}
        Macro F1: {best_row['Macro F1']:.4f}
        Weighted F1: {best_row['Weighted F1']:.4f}
        Minority Recall: {best_row['Minority Class Recall']:.4f}
        Train/Test Macro F1 Gap: {best_row['Train/Test Gap']:.4f}
        CV Macro F1: {best_row['CV Macro F1 Mean']:.4f} +/- {best_row['CV Macro F1 Std']:.4f}

        Nhận xét: mô hình được chọn không chỉ dựa trên Accuracy mà còn xét Macro F1,
        Balanced Accuracy, recall của nhóm lớp ít mẫu và cảnh báo nếu một lớp quan trọng
        bị dự đoán kém. Với các lớp rất ít mẫu như UDP-Lag Flood, kết quả có thể chưa
        ổn định và cần thêm dữ liệu thực tế để cải thiện khả năng tổng quát hóa.
        '''
        print(conclusion)
        """
    ),
]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")

    try:
        MIRROR_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, MIRROR_NOTEBOOK_PATH)
        print(f"Wrote mirror {MIRROR_NOTEBOOK_PATH}")
    except OSError as exc:
        print(f"Mirror write skipped: {exc}")


if __name__ == "__main__":
    main()
