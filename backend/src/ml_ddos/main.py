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

from .data_loader import combine_and_resplit, load_dataset
from .models import TrainingConfig, train_and_evaluate_from_raw
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
        balance_training=not args.no_balance,
        attack_to_benign_ratio=args.attack_to_benign_ratio,
        max_binary_group_samples=args.max_binary_group_samples,
        min_attack_class_samples=args.min_attack_class_samples,
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


def step4_print_summary(val_scores, test_scores):
    """Step 4: Print best-model summary without generating report files."""
    print("\n" + "#" * 60)
    print("#  STEP 4: MODEL SUMMARY")
    print("#" * 60)

    val_f1_col = "CV F1-Score Mean" if "CV F1-Score Mean" in val_scores.columns else "F1-Score"
    test_f1_col = "Test F1-Score" if "Test F1-Score" in test_scores.columns else "F1-Score"

    best_val = val_scores[val_f1_col].idxmax()
    best_test = test_scores[test_f1_col].idxmax()

    print(f"\n  Best Model (Validation F1): {best_val} "
          f"(F1 = {val_scores.loc[best_val, val_f1_col]:.6f})")
    print(f"  Best Model (Test F1):       {best_test} "
          f"(F1 = {test_scores.loc[best_test, test_f1_col]:.6f})")



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
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable train-only benign/attack balancing")
    parser.add_argument("--attack-to-benign-ratio", type=float, default=1.0,
                        help="Target attack/benign ratio in the balanced training split")
    parser.add_argument("--max-binary-group-samples", type=int, default=40000,
                        help="Maximum samples kept for each binary group in the training split")
    parser.add_argument("--min-attack-class-samples", type=int, default=500,
                        help="Minimum samples per attack class after oversampling rare classes")
    parser.add_argument("--combine-resplit", action="store_true",
                        help="Combine train+test, deduplicate, and stratified-resplit "
                             "to eliminate CICDDoS2019 source-split distribution shift")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for CV/search")
    args = parser.parse_args()

    # Step 1
    train_df, test_df = step1_load_data()

    # Step 2
    step2_profile_data(train_df, test_df)

    # Optional: combine and re-split to fix distribution shift
    if args.combine_resplit:
        print("\n" + "#" * 60)
        print("#  COMBINE & RE-SPLIT (fixing distribution shift)")
        print("#" * 60)
        train_df, test_df = combine_and_resplit(train_df, test_df)
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
    step4_print_summary(val_scores, test_scores)

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE!")
    print("  Check results/ for metric CSVs and live logs.")
    print("  Check saved_models/ for trained model files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
