"""
main.py - DDoS Detection Pipeline Orchestrator

Runs the end-to-end pipeline:
  Step 1: Download and load the CICDDoS2019 dataset
  Step 2: Preprocess the data (clean, engineer, encode, scale)
  Step 3: Train and evaluate models (to be implemented)

Usage:
    python main.py              # Run Steps 1 & 2 only
    python main.py --full       # Run all steps (when Step 3 is ready)
"""

import argparse
import os
import sys

from data_loader import load_dataset
from preprocessor import preprocess


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def step1_load_data():
    """Step 1: Download and load the dataset."""
    print("\n" + "#" * 60)
    print("#  STEP 1: DATA LOADING")
    print("#" * 60)

    train_df, test_df = load_dataset()
    return train_df, test_df


def step2_preprocess(train_df, test_df):
    """Step 2: Run the full preprocessing pipeline."""
    print("\n" + "#" * 60)
    print("#  STEP 2: PREPROCESSING")
    print("#" * 60)

    result = preprocess(train_df, test_df)

    # Print summaries
    print("\n\n" + "=" * 70)
    print("  DATA SUMMARY (BEFORE PREPROCESSING)")
    print("=" * 70)
    print(result["summary_before"])

    print("\n\n" + "=" * 70)
    print("  DATA SUMMARY (AFTER PREPROCESSING)")
    print("=" * 70)
    print(result["summary_after"])

    # Final stats
    print("\n\n" + "=" * 70)
    print("  FINAL PROCESSED DATA SHAPES")
    print("=" * 70)
    print(f"  X_train: {result['X_train'].shape}")
    print(f"  X_val:   {result['X_val'].shape}")
    print(f"  X_test:  {result['X_test'].shape}")
    print(f"  y_train: {result['y_train'].shape}")
    print(f"  y_val:   {result['y_val'].shape}")
    print(f"  y_test:  {result['y_test'].shape}")
    print(f"\n  Features retained:   {len(result['feature_names'])}")
    print(f"  Columns dropped (single-value): {len(result['dropped_single_val'])}")
    print(f"  Columns dropped (high corr):    {len(result['dropped_high_corr'])}")
    print(f"\n  Label mapping: {result['label_map']}")

    return result


def step3_train_evaluate(result):
    """Step 3: Train models and evaluate. (To be implemented.)"""
    print("\n" + "#" * 60)
    print("#  STEP 3: MODEL TRAINING & EVALUATION")
    print("#  (Not yet implemented — will be added next)")
    print("#" * 60)


def main():
    parser = argparse.ArgumentParser(description="DDoS Detection Pipeline")
    parser.add_argument("--full", action="store_true",
                        help="Run all steps including model training")
    args = parser.parse_args()

    # Step 1
    train_df, test_df = step1_load_data()

    # Step 2
    result = step2_preprocess(train_df, test_df)

    # Step 3 (only if --full flag is passed)
    if args.full:
        step3_train_evaluate(result)
    else:
        print("\n" + "=" * 70)
        print("  Steps 1 & 2 complete.")
        print("  Run with --full to proceed to model training (Step 3).")
        print("=" * 70)


if __name__ == "__main__":
    main()
