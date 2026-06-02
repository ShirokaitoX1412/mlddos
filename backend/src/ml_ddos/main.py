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
from .models import TrainingConfig, generate_markdown_report, train_and_evaluate_from_raw
from .paths import RESULTS_DIR as PROJECT_RESULTS_DIR
from .preprocessor import generate_data_summary, harmonize_labels


RESULTS_DIR = str(PROJECT_RESULTS_DIR)


def step1_load_data():
    """Step 1: Download and load the dataset."""
    print("\n" + "#" * 60)
    print("#  STEP 1: DATA LOADING")
    print("#" * 60)

    train_df, test_df = load_dataset()
    return train_df, test_df


def step2_profile_data(train_df, test_df):
    """Step 2: Profile raw data without fitting feature transforms."""
    print("\n" + "#" * 60)
    print("#  STEP 2: RAW DATA PROFILING")
    print("#" * 60)

    summary_before = generate_data_summary(train_df, test_df)
    harmonized_train, harmonized_test = harmonize_labels(train_df, test_df)
    summary_after = generate_data_summary(harmonized_train, harmonized_test)

    print("\n\n" + "=" * 70)
    print("  DATA SUMMARY (RAW)")
    print("=" * 70)
    print(summary_before)

    print("\n\n" + "=" * 70)
    print("  DATA SUMMARY (AFTER LABEL HARMONIZATION ONLY)")
    print("=" * 70)
    print(summary_after)

    print("\n  No scalers, imputers, encoders, or feature filters were fitted in this step.")
    print("  Those transforms are fitted inside each CV training fold.")
    return summary_before, summary_after


def step3_train_evaluate(train_df, test_df, args):
    """Step 3: Train, tune, evaluate, and generate visualizations."""
    print("\n" + "#" * 60)
    print("#  STEP 3: LEAKAGE-SAFE MODEL TRAINING & EVALUATION")
    print("#" * 60)

    config = TrainingConfig(
        cv_folds=args.cv_folds,
        search_iter=args.search_iter,
        correlation_threshold=args.correlation_threshold,
        enable_smote=not args.no_smote,
        n_jobs=args.n_jobs,
    )
    val_scores, test_scores = train_and_evaluate_from_raw(train_df, test_df, config=config)

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION RESULTS")
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
                        help="Run only Steps 1 & 2 (data loading + raw profiling)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of stratified/group-aware CV folds")
    parser.add_argument("--search-iter", type=int, default=20,
                        help="RandomizedSearchCV iterations for RF, Extra Trees, and XGBoost")
    parser.add_argument("--correlation-threshold", type=float, default=0.9,
                        help="Training-fold correlation threshold for dropping redundant features")
    parser.add_argument("--no-smote", action="store_true",
                        help="Disable SMOTE candidate pipelines")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for CV/search")
    args = parser.parse_args()

    # Step 1
    train_df, test_df = step1_load_data()

    # Step 2
    step2_profile_data(train_df, test_df)

    if args.steps_1_2:
        print("\n" + "=" * 70)
        print("  Steps 1 & 2 complete.")
        print("  Run without --steps-1-2 to proceed to model training.")
        print("=" * 70)
        return

    # Step 3
    val_scores, test_scores = step3_train_evaluate(train_df, test_df, args)

    # Step 4
    step4_generate_report(val_scores, test_scores)

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE!")
    print("  Check results/ for plots and reports.")
    print("  Check saved_models/ for trained model files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
