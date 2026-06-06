"""
data_loader.py - CICDDoS2019 Dataset Downloader & Loader

Downloads the CICDDoS2019 dataset (parquet format) from Kaggle
and loads it into pandas DataFrames for training and testing.

Dataset: CICDDoS2019 by Canadian Institute for Cybersecurity
Source:  https://www.kaggle.com/datasets/dhoogla/cicddos2019
"""

import os
import glob
import pandas as pd

from .paths import DATA_DIR as PROJECT_DATA_DIR

DATA_DIR = str(PROJECT_DATA_DIR)

KAGGLE_DATASET_SLUG = "dhoogla/cicddos2019"


def download_dataset(dest_dir: str = DATA_DIR) -> str:
    """Download the CICDDoS2019 dataset from Kaggle using kagglehub.

    Parameters
    ----------
    dest_dir : str
        Local directory where parquet files will be stored.

    Returns
    -------
    str
        Path to the directory containing the downloaded parquet files.
    """
    os.makedirs(dest_dir, exist_ok=True)

    existing = glob.glob(os.path.join(dest_dir, "*.parquet"))
    if len(existing) >= 5:
        print(f"[data_loader] Dataset already present ({len(existing)} files). Skipping download.")
        return dest_dir

    try:
        import kagglehub
        print("[data_loader] Downloading CICDDoS2019 from Kaggle via kagglehub ...")
        downloaded_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        print(f"[data_loader] Downloaded to: {downloaded_path}")

        parquet_files = glob.glob(os.path.join(downloaded_path, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(downloaded_path, "**", "*.csv"), recursive=True)

        for f in parquet_files:
            dest = os.path.join(dest_dir, os.path.basename(f))
            if not os.path.exists(dest):
                os.link(f, dest) if _same_fs(f, dest_dir) else _copy_file(f, dest)

        print(f"[data_loader] {len(parquet_files)} files ready in {dest_dir}")
        return dest_dir

    except ImportError:
        print("[data_loader] kagglehub not installed. Install it: pip install kagglehub")
        print("[data_loader] Or download manually from: https://www.kaggle.com/datasets/dhoogla/cicddos2019")
        print(f"[data_loader] Place parquet files in: {dest_dir}")
        raise
    except Exception as e:
        print(f"[data_loader] Download failed: {e}")
        print("[data_loader] Trying alternative: download from reference repository ...")
        return _download_from_github(dest_dir)


def _download_from_github(dest_dir: str) -> str:
    """Fallback: clone parquet files from the reference GitHub repository."""
    import subprocess

    repo_url = "https://github.com/rakibnsajib/DDoS-Defense-A-Multiclass-and-Multidimensional-Detection-System-with-Diverse-Machine-Learning-Models.git"
    clone_dir = os.path.join(os.path.dirname(dest_dir), ".ref_repo_clone")

    if not os.path.exists(clone_dir):
        print("[data_loader] Cloning reference repo for data files ...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, clone_dir],
                       check=True, capture_output=True)

    src_data = os.path.join(clone_dir, "data")
    parquet_files = glob.glob(os.path.join(src_data, "*.parquet"))

    os.makedirs(dest_dir, exist_ok=True)
    for f in parquet_files:
        dest = os.path.join(dest_dir, os.path.basename(f))
        if not os.path.exists(dest):
            _copy_file(f, dest)

    print(f"[data_loader] {len(parquet_files)} parquet files copied to {dest_dir}")
    return dest_dir


def _same_fs(path1: str, path2: str) -> bool:
    """Check if two paths are on the same filesystem."""
    try:
        return os.stat(path1).st_dev == os.stat(path2).st_dev
    except OSError:
        return False


def _copy_file(src: str, dst: str) -> None:
    """Copy a file from src to dst."""
    import shutil
    shutil.copy2(src, dst)


def collect_file_paths(data_dir: str = DATA_DIR):
    """Scan data_dir for training and testing parquet files.

    Returns
    -------
    tuple[list[str], list[str]]
        (training_paths, testing_paths)
    """
    train_paths = sorted(glob.glob(os.path.join(data_dir, "*-training.parquet")))
    test_paths = sorted(glob.glob(os.path.join(data_dir, "*-testing.parquet")))

    print(f"[data_loader] Found {len(train_paths)} training files, {len(test_paths)} testing files.")
    return train_paths, test_paths


def filter_common_attack_types(train_paths: list, test_paths: list):
    """Keep only attack types present in BOTH training and testing sets.

    Returns
    -------
    tuple[list[str], list[str]]
        Filtered (training_paths, testing_paths)
    """
    def _prefix(path):
        return os.path.basename(path).split("-")[0]

    train_prefixes = {_prefix(p) for p in train_paths}
    test_prefixes = {_prefix(p) for p in test_paths}
    common = train_prefixes & test_prefixes

    train_filtered = [p for p in train_paths if _prefix(p) in common]
    test_filtered = [p for p in test_paths if _prefix(p) in common]

    print(f"[data_loader] Common attack types: {sorted(common)}")
    return train_filtered, test_filtered


def load_dataframes(train_paths: list, test_paths: list):
    """Read parquet files and concatenate into train/test DataFrames.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    from tqdm import tqdm

    print("[data_loader] Loading training data ...")
    train_dfs = [pd.read_parquet(p) for p in tqdm(train_paths, desc="Train files")]
    train_df = pd.concat(train_dfs, ignore_index=True)

    print("[data_loader] Loading testing data ...")
    test_dfs = [pd.read_parquet(p) for p in tqdm(test_paths, desc="Test files")]
    test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"[data_loader] Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def load_dataset(data_dir: str = DATA_DIR):
    """Full pipeline: download → collect → filter → load.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    download_dataset(data_dir)
    train_paths, test_paths = collect_file_paths(data_dir)

    if not train_paths or not test_paths:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}. "
            "Ensure the dataset is downloaded correctly."
        )

    train_paths, test_paths = filter_common_attack_types(train_paths, test_paths)
    train_df, test_df = load_dataframes(train_paths, test_paths)
    return train_df, test_df


def combine_and_resplit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine train+test, deduplicate, and stratified-resplit.

    This eliminates the CICDDoS2019 source-split distribution shift
    (e.g. UDP-Lag: 0.05% train vs 22.79% test) by creating a new
    split where every class has proportional representation in both
    train and test sets.

    Accepts DataFrames in either raw or harmonized label format.
    Test labels are mapped to match training labels before combining.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        DataFrames with a ``Label`` column (raw or harmonized).
    test_size : float
        Fraction of the combined data to reserve for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (new_train_df, new_test_df) with unified label names.
    """
    from sklearn.model_selection import train_test_split

    from .preprocessor import LABEL_HARMONIZATION_MAP

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Map test labels to train convention so they can be combined
    train_labels = set(train_df["Label"].unique())
    test_labels = set(test_df["Label"].unique())
    unmapped = test_labels - train_labels - set(LABEL_HARMONIZATION_MAP.keys())
    if unmapped:
        print(f"[data_loader] Removing unmappable test labels: {unmapped}")
        test_df = test_df[~test_df["Label"].isin(unmapped)]
    test_df["Label"] = test_df["Label"].map(
        lambda x: LABEL_HARMONIZATION_MAP.get(x, x)
    )
    test_df = test_df.dropna(subset=["Label"])

    combined = pd.concat([train_df, test_df], ignore_index=True)

    n_before = len(combined)
    feature_cols = [c for c in combined.columns if c != "Label"]
    combined = combined.drop_duplicates(subset=feature_cols, keep="first")
    n_after = len(combined)
    n_dropped = n_before - n_after
    print(f"[data_loader] Combined {n_before:,} rows, dropped {n_dropped:,} "
          f"duplicates → {n_after:,} unique rows")

    new_train, new_test = train_test_split(
        combined,
        test_size=test_size,
        random_state=random_state,
        stratify=combined["Label"],
    )
    new_train = new_train.reset_index(drop=True)
    new_test = new_test.reset_index(drop=True)

    print(f"[data_loader] Re-split: train={len(new_train):,}, test={len(new_test):,}")
    print(f"[data_loader] Train label distribution:\n"
          f"{new_train['Label'].value_counts().to_string()}")
    print(f"[data_loader] Test label distribution:\n"
          f"{new_test['Label'].value_counts().to_string()}")

    return new_train, new_test


if __name__ == "__main__":
    train_df, test_df = load_dataset()
    print("\n=== Dataset Summary ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Testing samples:  {len(test_df):,}")
    print(f"Features:         {train_df.shape[1] - 1}")
    print(f"\nTraining label distribution:\n{train_df['Label'].value_counts()}")
    print(f"\nTesting label distribution:\n{test_df['Label'].value_counts()}")
