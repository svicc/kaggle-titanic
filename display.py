from pathlib import Path


def print_results(fold_df, summary):
    """Print fold-level results and the model comparison table."""
    print("\n" + "=" * 75)
    print("INDIVIDUAL FOLD RESULTS")
    print("=" * 75)
    print(fold_df.round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("FINAL THREE-MODEL COMPARISON")
    print("=" * 75)
    print(summary.round(4).to_string())


def save_results(
    fold_df,
    summary,
    output_dir=".",
):
    """Save fold-level and summary results as CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_path = output_dir / "three_model_fold_results.csv"
    summary_path = output_dir / "three_model_summary.csv"

    fold_df.to_csv(fold_path, index=False)
    summary.to_csv(summary_path)

    print(f"\nResults saved:\n- {fold_path}\n- {summary_path}")
