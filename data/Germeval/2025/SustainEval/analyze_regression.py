#!/usr/bin/env python3
"""
Script to analyze regression performance for SustainEval task.
Compares label vs pred columns and calculates various metrics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(file_path):
    """Load the TSV file and extract label and prediction columns."""
    df = pd.read_csv(file_path, sep="\t")
    print(f"Loaded {len(df)} samples from {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def load_train_data(script_dir):
    """Load train data to get train label distribution."""
    train_path = script_dir / "train.txt"
    if train_path.exists():
        try:
            train_df = pd.read_csv(train_path, sep="\t")
            if "task_b_label" in train_df.columns:
                train_labels = train_df["task_b_label"].values
                print(f"Loaded {len(train_labels)} train samples for distribution comparison")
                return train_labels
            else:
                print("Warning: 'task_b_label' column not found in train.txt")
        except Exception as e:
            print(f"Warning: Could not load train data: {e}")
    else:
        print("Warning: train.txt not found, skipping train distribution")
    return None


def calculate_metrics(labels, predictions):
    """Calculate various regression metrics."""
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)

    # Correlation metrics
    pearson_r, pearson_p = stats.pearsonr(labels, predictions)
    spearman_r, spearman_p = stats.spearmanr(labels, predictions)
    kendall_tau, kendall_p = stats.kendalltau(labels, predictions)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "Pearson r": pearson_r,
        "Pearson p-value": pearson_p,
        "Spearman ρ": spearman_r,
        "Spearman p-value": spearman_p,
        "Kendall τ": kendall_tau,
        "Kendall p-value": kendall_p,
    }

    return metrics


def analyze_errors(labels, predictions):
    """Analyze prediction errors and identify patterns."""
    errors = predictions - labels
    abs_errors = np.abs(errors)

    error_stats = {
        "Mean Error": np.mean(errors),
        "Std Error": np.std(errors),
        "Mean Absolute Error": np.mean(abs_errors),
        "Max Error": np.max(abs_errors),
        "Min Error": np.min(abs_errors),
        "95th Percentile Error": np.percentile(abs_errors, 95),
        "Error Skewness": stats.skew(errors),
        "Error Kurtosis": stats.kurtosis(errors),
    }

    return errors, error_stats


def identify_problems(labels, predictions, df=None):
    """Identify potential problems with the model."""
    problems = []

    # Check for systematic bias
    errors = predictions - labels
    mean_error = np.mean(errors)
    if abs(mean_error) > 0.1:
        if mean_error > 0:
            problems.append(f"Systematic overestimation: mean error = {mean_error:.3f}")
        else:
            problems.append(f"Systematic underestimation: mean error = {mean_error:.3f}")

    # Check for range issues
    pred_range = predictions.max() - predictions.min()
    label_range = labels.max() - labels.min()
    if pred_range < 0.7 * label_range:
        problems.append(
            f"Underdispersion: prediction range ({pred_range:.3f}) much smaller than label range ({label_range:.3f})"
        )
    elif pred_range > 1.3 * label_range:
        problems.append(
            f"Overdispersion: prediction range ({pred_range:.3f}) much larger than label range ({label_range:.3f})"
        )

    # Check for ceiling/floor effects
    if (predictions > 0.9).sum() > 0.1 * len(predictions):
        problems.append(
            f"Ceiling effect: {(predictions > 0.9).sum()} predictions ({100 * (predictions > 0.9).sum() / len(predictions):.1f}%) > 0.9"
        )

    if (predictions < 0.1).sum() > 0.1 * len(predictions):
        problems.append(
            f"Floor effect: {(predictions < 0.1).sum()} predictions ({100 * (predictions < 0.1).sum() / len(predictions):.1f}%) < 0.1"
        )

    # Check for prediction concentration
    pred_std = np.std(predictions)
    if pred_std < 0.1:
        problems.append(f"Low prediction variance: std = {pred_std:.3f} (model may be too conservative)")

    # Check correlation strength
    pearson_r, _ = stats.pearsonr(labels, predictions)
    if pearson_r < 0.3:
        problems.append(f"Weak correlation: Pearson r = {pearson_r:.3f} (poor predictive performance)")
    elif pearson_r < 0.5:
        problems.append(f"Moderate correlation: Pearson r = {pearson_r:.3f} (could be improved)")

    return problems


def create_visualizations(labels, predictions, errors, train_labels=None, save_dir=None):
    """Create various plots to visualize the regression performance."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Regression Analysis: SustainEval", fontsize=16, fontweight="bold")

    # 1. Scatter plot: Predictions vs Labels
    ax1 = axes[0, 0]
    ax1.scatter(labels, predictions, alpha=0.6, s=30)
    ax1.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect Prediction")
    ax1.set_xlabel("True Labels")
    ax1.set_ylabel("Predictions")
    ax1.set_title("Predictions vs True Labels")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add correlation info
    pearson_r, _ = stats.pearsonr(labels, predictions)
    ax1.text(
        0.05,
        0.95,
        f"Pearson r = {pearson_r:.3f}",
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 2. Residual plot
    ax2 = axes[0, 1]
    ax2.scatter(predictions, errors, alpha=0.6, s=30)
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax2.set_xlabel("Predictions")
    ax2.set_ylabel("Residuals (Pred - True)")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3 = axes[0, 2]
    ax3.hist(errors, bins=30, alpha=0.7, edgecolor="black")
    ax3.axvline(x=0, color="r", linestyle="--", linewidth=2)
    ax3.set_xlabel("Prediction Errors")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Error Distribution")
    ax3.grid(True, alpha=0.3)

    # Add stats
    ax3.text(
        0.05,
        0.95,
        f"Mean: {np.mean(errors):.3f}\nStd: {np.std(errors):.3f}",
        transform=ax3.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 4. Distribution comparison
    ax4 = axes[1, 0]
    ax4.hist(labels, bins=20, alpha=0.5, label="Dev True Labels", density=True)
    ax4.hist(predictions, bins=20, alpha=0.5, label="Dev Predictions", density=True)
    if train_labels is not None:
        ax4.hist(train_labels, bins=20, alpha=0.5, label="Train Labels", density=True)
    ax4.set_xlabel("Values")
    ax4.set_ylabel("Density")
    ax4.set_title("Distribution Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Q-Q plot for error normality
    ax5 = axes[1, 1]
    stats.probplot(errors, dist="norm", plot=ax5)
    ax5.set_title("Q-Q Plot (Error Normality)")
    ax5.grid(True, alpha=0.3)

    # 6. Absolute error vs true labels
    ax6 = axes[1, 2]
    abs_errors = np.abs(errors)
    ax6.scatter(labels, abs_errors, alpha=0.6, s=30)
    ax6.set_xlabel("True Labels")
    ax6.set_ylabel("Absolute Error")
    ax6.set_title("Absolute Error vs True Labels")
    ax6.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(labels, abs_errors, 1)
    p = np.poly1d(z)
    ax6.plot(labels, p(labels), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "regression_analysis.png", dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_dir / 'regression_analysis.png'}")

    plt.show()


def print_metrics_report(metrics, error_stats, problems):
    """Print a comprehensive metrics report."""
    print("\n" + "=" * 80)
    print("REGRESSION PERFORMANCE ANALYSIS")
    print("=" * 80)

    print("\n📊 CORE METRICS:")
    print("-" * 40)
    print(f"MAE (Mean Absolute Error):    {metrics['MAE']:.4f}")
    print(f"MSE (Mean Squared Error):     {metrics['MSE']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['RMSE']:.4f}")
    print(f"R² (Coefficient of Determination): {metrics['R²']:.4f}")

    print("\n📈 CORRELATION METRICS:")
    print("-" * 40)
    print(f"Pearson r:    {metrics['Pearson r']:.4f} (p = {metrics['Pearson p-value']:.4e})")
    print(f"Spearman ρ:   {metrics['Spearman ρ']:.4f} (p = {metrics['Spearman p-value']:.4e})")
    print(f"Kendall τ:    {metrics['Kendall τ']:.4f} (p = {metrics['Kendall p-value']:.4e})")

    print("\n📉 ERROR ANALYSIS:")
    print("-" * 40)
    for key, value in error_stats.items():
        print(f"{key:<25}: {value:.4f}")

    print("\n⚠️  POTENTIAL PROBLEMS:")
    print("-" * 40)
    if problems:
        for i, problem in enumerate(problems, 1):
            print(f"{i}. {problem}")
    else:
        print("No major problems detected! ✅")

    print("\n🎯 INTERPRETATION:")
    print("-" * 40)

    # Interpret Pearson correlation
    pearson_r = metrics["Pearson r"]
    if pearson_r < 0.3:
        corr_strength = "weak"
    elif pearson_r < 0.5:
        corr_strength = "moderate"
    elif pearson_r < 0.7:
        corr_strength = "strong"
    else:
        corr_strength = "very strong"

    print(f"• Correlation strength: {corr_strength} (r = {pearson_r:.3f})")

    # Interpret R²
    r2 = metrics["R²"]
    if r2 < 0.1:
        r2_desc = "very poor"
    elif r2 < 0.25:
        r2_desc = "poor"
    elif r2 < 0.5:
        r2_desc = "moderate"
    elif r2 < 0.75:
        r2_desc = "good"
    else:
        r2_desc = "excellent"

    print(f"• Model explains {r2 * 100:.1f}% of variance ({r2_desc} fit)")

    # Interpret MAE relative to label range
    print(f"• Average prediction error: {metrics['MAE']:.3f} units")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze regression performance")
    parser.add_argument(
        "--file", "-f", default="dev_regr_mgb_no_decoder.tsv", help="Path to TSV file (default: dev_regr.tsv)"
    )
    parser.add_argument("--save-plots", "-s", action="store_true", help="Save plots to file")

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    file_path = script_dir / args.file

    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        return

    # Load and analyze data
    df = load_data(file_path)

    if "label" not in df.columns or "pred" not in df.columns:
        print("Error: File must contain 'label' and 'pred' columns!")
        return

    labels = df["label"].values
    predictions = df["pred"].values

    print(f"\nLabel range: [{labels.min():.3f}, {labels.max():.3f}]")
    print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")

    # Load train data for distribution comparison
    train_labels = load_train_data(script_dir)
    if train_labels is not None:
        print(f"Train label range: [{train_labels.min():.3f}, {train_labels.max():.3f}]")

    # Calculate metrics
    metrics = calculate_metrics(labels, predictions)
    errors, error_stats = analyze_errors(labels, predictions)
    problems = identify_problems(labels, predictions, df)

    # Print report
    print_metrics_report(metrics, error_stats, problems)

    # Create visualizations
    save_dir = script_dir if args.save_plots else None
    create_visualizations(labels, predictions, errors, train_labels, save_dir)


if __name__ == "__main__":
    main()
