"""
main.py - DDoS Detection Pipeline Orchestrator

Runs the end-to-end pipeline:
  Step 1: Download and load the CICDDoS2019 dataset
  Step 2: Preprocess the data (clean, engineer, encode, scale)
  Step 3: Train all 5 models, evaluate, and generate reports
  Step 4: Generate comparison report and update README

Usage:
    python main.py              # Run full pipeline (all steps)
    python main.py --steps-1-2  # Run only Steps 1 & 2
"""

import argparse
import os

from .data_loader import load_dataset
from .models import generate_markdown_report, train_and_evaluate
from .paths import RESULTS_DIR as PROJECT_RESULTS_DIR
from .preprocessor import preprocess


RESULTS_DIR = str(PROJECT_RESULTS_DIR)


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

    print("\n\n" + "=" * 70)
    print("  DATA SUMMARY (BEFORE PREPROCESSING)")
    print("=" * 70)
    print(result["summary_before"])

    print("\n\n" + "=" * 70)
    print("  DATA SUMMARY (AFTER PREPROCESSING)")
    print("=" * 70)
    print(result["summary_after"])

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
    """Step 3: Train all 5 models, evaluate, and generate visualizations."""
    print("\n" + "#" * 60)
    print("#  STEP 3: MODEL TRAINING & EVALUATION")
    print("#" * 60)

    val_scores, test_scores = train_and_evaluate(result)

    print("\n" + "=" * 60)
    print("  VALIDATION SET RESULTS")
    print("=" * 60)
    print(val_scores.to_string())

    print("\n" + "=" * 60)
    print("  TEST SET RESULTS")
    print("=" * 60)
    print(test_scores.to_string())

    return val_scores, test_scores


def step4_generate_report(val_scores, test_scores):
    """Step 4: Generate comparison report and save."""
    print("\n" + "#" * 60)
    print("#  STEP 4: GENERATING COMPARISON REPORT")
    print("#" * 60)

    report_md = generate_markdown_report(val_scores, test_scores)

    report_path = os.path.join(RESULTS_DIR, "comparison_report.md")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"  Report saved: {report_path}")

    # Determine best model
    best_val = val_scores["F1-Score"].idxmax()
    best_test = test_scores["F1-Score"].idxmax()

    print(f"\n  Best Model (Validation F1): {best_val} "
          f"(F1 = {val_scores.loc[best_val, 'F1-Score']:.6f})")
    print(f"  Best Model (Test F1):       {best_test} "
          f"(F1 = {test_scores.loc[best_test, 'F1-Score']:.6f})")

    return report_md


def main():
    parser = argparse.ArgumentParser(description="DDoS Detection Pipeline")
    parser.add_argument("--steps-1-2", action="store_true",
                        help="Run only Steps 1 & 2 (data loading + preprocessing)")
    args = parser.parse_args()

    # Step 1
    train_df, test_df = step1_load_data()

    # Step 2
    result = step2_preprocess(train_df, test_df)

    if args.steps_1_2:
        print("\n" + "=" * 70)
        print("  Steps 1 & 2 complete.")
        print("  Run without --steps-1-2 to proceed to model training.")
        print("=" * 70)
        return

    # Step 3
    val_scores, test_scores = step3_train_evaluate(result)

    # Step 4
    step4_generate_report(val_scores, test_scores)

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE!")
    print("  Check results/ for plots and reports.")
    print("  Check saved_models/ for trained model files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
