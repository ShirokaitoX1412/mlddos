"""
preprocessor.py - Data Preprocessing Pipeline for DDoS Detection

Handles all data cleaning, feature engineering, encoding, and scaling
following the methodology from the CICDDoS2019 reference notebook.

Pipeline steps:
  1. Harmonize labels between train/test sets
  2. Remove classes not present in training data
  3. Map attack types to standard DDoS categories
  4. Remove duplicate rows
  5. Handle missing / infinite values
  6. Drop single-unique-value columns
  7. Drop highly correlated columns (threshold = 0.8)
  8. Encode target labels
  9. Scale features (MinMaxScaler)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Mapping from test-set label names → training-set label names
LABEL_HARMONIZATION_MAP = {
    "DrDoS_UDP": "UDP",
    "UDP-lag": "UDPLag",
    "DrDoS_MSSQL": "MSSQL",
    "DrDoS_LDAP": "LDAP",
    "DrDoS_NetBIOS": "NetBIOS",
    "Syn": "Syn",
    "Benign": "Benign",
}

# Semantic mapping: raw labels → standard DDoS category names
# (for multi-class classification with descriptive names)
DDOS_CATEGORY_MAP = {
    "Syn": "TCP SYN Flood",
    "UDP": "UDP Flood",
    "UDPLag": "UDP-Lag Flood",
    "LDAP": "LDAP Flood",
    "MSSQL": "MSSQL Flood",
    "NetBIOS": "NetBIOS Flood",
    "Portmap": "Portmap Flood",
    "Benign": "Benign",
}

TARGET_COL = "Label"


def harmonize_labels(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Align label names between train and test DataFrames.

    - Removes 'WebDDoS' rows from test (not in training data).
    - Maps test labels to match training label names.
    - Maps both sets to descriptive DDoS category names.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Remove classes that don't exist in training data
    train_labels = set(train_df[TARGET_COL].unique())
    test_only_labels = set(test_df[TARGET_COL].unique()) - train_labels
    unmapped_test = test_only_labels - set(LABEL_HARMONIZATION_MAP.keys())

    if unmapped_test:
        print(f"[preprocessor] Removing test-only labels (not in train): {unmapped_test}")
        test_df = test_df[~test_df[TARGET_COL].isin(unmapped_test)]

    # Harmonize test labels to match training label conventions
    test_df[TARGET_COL] = test_df[TARGET_COL].map(LABEL_HARMONIZATION_MAP)
    test_df = test_df.dropna(subset=[TARGET_COL])

    # Apply descriptive DDoS category names
    train_df[TARGET_COL] = train_df[TARGET_COL].map(DDOS_CATEGORY_MAP)
    test_df[TARGET_COL] = test_df[TARGET_COL].map(DDOS_CATEGORY_MAP)

    # Drop any rows that couldn't be mapped
    train_df = train_df.dropna(subset=[TARGET_COL])
    test_df = test_df.dropna(subset=[TARGET_COL])

    print(f"[preprocessor] Labels harmonized. Train classes: {sorted(train_df[TARGET_COL].unique())}")
    return train_df, test_df


def remove_duplicates(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """Remove duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"[preprocessor] Removed {n_removed:,} duplicate rows from {name}.")
    return df


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf values with NaN, then fill NaN with column medians."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    if n_inf > 0:
        print(f"[preprocessor] Replacing {n_inf:,} infinite values with NaN.")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_null = df[numeric_cols].isnull().sum().sum()
    if n_null > 0:
        print(f"[preprocessor] Filling {n_null:,} NaN values with column medians.")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def drop_single_value_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Drop columns that have only one unique value in the training set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        (train_df, test_df, dropped_columns)
    """
    single_val_cols = [
        col for col in train_df.columns
        if col != TARGET_COL and train_df[col].nunique() == 1
    ]

    if single_val_cols:
        print(f"[preprocessor] Dropping {len(single_val_cols)} single-value columns: {single_val_cols}")
        train_df = train_df.drop(columns=single_val_cols)
        test_df = test_df.drop(columns=[c for c in single_val_cols if c in test_df.columns])

    return train_df, test_df, single_val_cols


def drop_highly_correlated(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           threshold: float = 0.8):
    """Drop features with absolute correlation > threshold.

    Uses only training data to determine which columns to drop.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        (train_df, test_df, dropped_columns)
    """
    numeric_df = train_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # Upper triangle mask
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(mask)

    high_corr_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    if high_corr_cols:
        print(f"[preprocessor] Dropping {len(high_corr_cols)} highly correlated columns (>{threshold}).")
        train_df = train_df.drop(columns=high_corr_cols)
        test_df = test_df.drop(columns=[c for c in high_corr_cols if c in test_df.columns])

    return train_df, test_df, high_corr_cols


def encode_labels(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
    """Encode string labels to integers using LabelEncoder.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder, dict]
        (y_train_enc, y_val_enc, y_test_enc, encoder, label_map)
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    label_map = {idx: label for idx, label in enumerate(le.classes_)}
    print(f"[preprocessor] Label encoding: {label_map}")
    return y_train_enc, y_val_enc, y_test_enc, le, label_map


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """Apply MinMaxScaler fitted on training data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"[preprocessor] Features scaled with MinMaxScaler. Shape: {X_train_scaled.shape}")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def analyze_columns(df: pd.DataFrame, cat_threshold: int = 10):
    """Categorize columns into categorical and numerical groups.

    Returns
    -------
    tuple[list[str], list[str]]
        (categorical_cols, numerical_cols)
    """
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [
        col for col in df.columns
        if df[col].nunique() < cat_threshold and df[col].dtypes != "O"
    ]
    cat_cols = cat_cols + num_but_cat

    num_cols = [
        col for col in df.columns
        if df[col].dtypes != "O" and col not in num_but_cat
    ]

    return cat_cols, num_cols


def generate_data_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Generate a comprehensive text summary of the dataset.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("          CICDDoS2019 DATASET SUMMARY")
    lines.append("=" * 70)

    lines.append(f"\n--- Shape ---")
    lines.append(f"Training set:  {train_df.shape[0]:>10,} samples x {train_df.shape[1]:>3} features")
    lines.append(f"Testing set:   {test_df.shape[0]:>10,} samples x {test_df.shape[1]:>3} features")

    lines.append(f"\n--- Training Label Distribution ---")
    train_counts = train_df[TARGET_COL].value_counts()
    for label, count in train_counts.items():
        pct = count / len(train_df) * 100
        lines.append(f"  {label:<20s} {count:>10,}  ({pct:5.2f}%)")

    lines.append(f"\n--- Testing Label Distribution ---")
    test_counts = test_df[TARGET_COL].value_counts()
    for label, count in test_counts.items():
        pct = count / len(test_df) * 100
        lines.append(f"  {label:<20s} {count:>10,}  ({pct:5.2f}%)")

    lines.append(f"\n--- Data Types ---")
    dtypes = train_df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        lines.append(f"  {str(dtype):<15s} {count:>3} columns")

    cat_cols, num_cols = analyze_columns(train_df)
    lines.append(f"\n--- Column Analysis ---")
    lines.append(f"  Categorical columns:  {len(cat_cols)}")
    lines.append(f"  Numerical columns:    {len(num_cols)}")

    lines.append(f"\n--- Missing Values ---")
    n_missing_train = train_df.isnull().sum().sum()
    n_missing_test = test_df.isnull().sum().sum()
    lines.append(f"  Training: {n_missing_train:,}")
    lines.append(f"  Testing:  {n_missing_test:,}")

    lines.append(f"\n--- Duplicate Rows ---")
    lines.append(f"  Training: {train_df.duplicated().sum():,}")
    lines.append(f"  Testing:  {test_df.duplicated().sum():,}")

    n_inf_train = np.isinf(train_df.select_dtypes(include=[np.number])).sum().sum()
    n_inf_test = np.isinf(test_df.select_dtypes(include=[np.number])).sum().sum()
    lines.append(f"\n--- Infinite Values ---")
    lines.append(f"  Training: {n_inf_train:,}")
    lines.append(f"  Testing:  {n_inf_test:,}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame,
               correlation_threshold: float = 0.8,
               test_size: float = 0.2,
               random_state: int = 42):
    """Full preprocessing pipeline.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training data.
    test_df : pd.DataFrame
        Raw testing data.
    correlation_threshold : float
        Threshold for dropping correlated features.
    test_size : float
        Fraction of training data used for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - X_train, X_val, X_test: scaled feature arrays
        - y_train, y_val, y_test: encoded label arrays
        - label_encoder: fitted LabelEncoder
        - label_map: dict mapping int → label string
        - scaler: fitted MinMaxScaler
        - feature_names: list of feature column names
        - summary_before: data summary before preprocessing
        - summary_after: data summary after preprocessing
        - dropped_single_val: columns dropped for single value
        - dropped_high_corr: columns dropped for high correlation
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 50)
    print("  PREPROCESSING PIPELINE")
    print("=" * 50)

    # Step 0: Raw data summary
    summary_before = generate_data_summary(train_df, test_df)
    print("\n[Step 0] Raw data summary generated.")

    # Step 1: Harmonize labels
    print("\n[Step 1] Harmonizing labels ...")
    train_df, test_df = harmonize_labels(train_df, test_df)

    # Step 2: Remove duplicates
    print("\n[Step 2] Removing duplicates ...")
    train_df = remove_duplicates(train_df, "training set")
    test_df = remove_duplicates(test_df, "testing set")

    # Step 3: Handle invalid values
    print("\n[Step 3] Handling invalid/infinite values ...")
    train_df = handle_invalid_values(train_df)
    test_df = handle_invalid_values(test_df)

    # Step 4: Drop single-value columns
    print("\n[Step 4] Dropping single-value columns ...")
    train_df, test_df, dropped_sv = drop_single_value_columns(train_df, test_df)

    # Step 5: Drop highly correlated columns
    print("\n[Step 5] Dropping highly correlated columns ...")
    train_df, test_df, dropped_hc = drop_highly_correlated(
        train_df, test_df, threshold=correlation_threshold
    )

    # Summary after cleaning
    summary_after = generate_data_summary(train_df, test_df)

    # Step 6: Train-validation split
    print("\n[Step 6] Splitting training data into train/validation ...")
    X_train_full = train_df.drop(columns=[TARGET_COL])
    y_train_full = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=test_size, random_state=random_state, stratify=y_train_full
    )
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    feature_names = list(X_train.columns)

    # Step 7: Encode labels
    print("\n[Step 7] Encoding labels ...")
    y_train_enc, y_val_enc, y_test_enc, le, label_map = encode_labels(y_train, y_val, y_test)

    # Step 8: Scale features
    print("\n[Step 8] Scaling features ...")
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    print("\n" + "=" * 50)
    print("  PREPROCESSING COMPLETE")
    print("=" * 50)

    return {
        "X_train": X_train_sc,
        "X_val": X_val_sc,
        "X_test": X_test_sc,
        "y_train": y_train_enc,
        "y_val": y_val_enc,
        "y_test": y_test_enc,
        "label_encoder": le,
        "label_map": label_map,
        "scaler": scaler,
        "feature_names": feature_names,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "dropped_single_val": dropped_sv,
        "dropped_high_corr": dropped_hc,
    }
