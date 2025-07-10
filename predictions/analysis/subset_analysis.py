import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, spearmanr


def load_data():
    """Load existing and new task results"""
    with open("predictions/analysis/results_existing_tasks.json", "r") as f:
        existing_results = json.load(f)

    with open("predictions/analysis/results_new_tasks.json", "r") as f:
        new_results = json.load(f)

    return existing_results, new_results


def create_unified_dataframe(existing_results, new_results):
    """Create a unified dataframe with all models and tasks"""

    # Process existing tasks
    existing_data = []
    for entry in existing_results:
        model = entry["model"]
        row = {"model": model}

        # Add all task scores, excluding metadata
        for key, value in entry.items():
            if key not in ["model", "team", "team_url", "setting", "model_type", "model_arch", "num_params"]:
                try:
                    row[f"existing_{key}"] = float(value)
                except (ValueError, TypeError):
                    pass
        existing_data.append(row)

    existing_df = pd.DataFrame(existing_data)

    # Process new tasks
    new_data = []
    for entry in new_results:
        model = entry["model"]
        row = {"model": model}

        # Add all task scores
        for task, data in entry.items():
            if task != "model" and isinstance(data, dict) and "metric" in data:
                try:
                    row[f"new_{task}"] = float(data["metric"])
                except (ValueError, TypeError):
                    pass
        new_data.append(row)

    new_df = pd.DataFrame(new_data)

    # Merge on model name
    unified_df = pd.merge(existing_df, new_df, on="model", how="outer")

    return unified_df


def calculate_rankings_and_averages(df):
    """Calculate average scores and rankings for different task groups"""

    # Separate task columns
    existing_tasks = [col for col in df.columns if col.startswith("existing_")]
    new_tasks = [col for col in df.columns if col.startswith("new_")]
    all_tasks = existing_tasks + new_tasks

    # Filter to only include models with complete data for all tasks
    initial_model_count = len(df)
    df_complete = df.dropna(subset=all_tasks).copy()
    final_model_count = len(df_complete)

    print(f"Filtered from {initial_model_count} to {final_model_count} models with complete data")
    print(f"Removed {initial_model_count - final_model_count} models with incomplete data")
    print()

    # Calculate averages (now all models have complete data)
    df_complete["avg_existing"] = df_complete[existing_tasks].mean(axis=1)
    df_complete["avg_new"] = df_complete[new_tasks].mean(axis=1)
    df_complete["avg_all"] = df_complete[all_tasks].mean(axis=1)

    # Calculate rankings (higher score = better rank, use min method for consistent tie handling)
    df_complete["rank_existing"] = df_complete["avg_existing"].rank(ascending=False, method="min")
    df_complete["rank_new"] = df_complete["avg_new"].rank(ascending=False, method="min")
    df_complete["rank_all"] = df_complete["avg_all"].rank(ascending=False, method="min")

    return df_complete, existing_tasks, new_tasks, all_tasks


def evaluate_subset_ranking(df, task_subset, reference_ranking):
    """Evaluate how well a task subset reproduces the reference ranking"""

    # Calculate average score for the subset
    subset_avg = df[task_subset].mean(axis=1, skipna=True)
    subset_ranking = subset_avg.rank(ascending=False, method="min", na_option="bottom")

    # Remove NaN values for correlation calculation
    valid_mask = ~(subset_ranking.isna() | reference_ranking.isna())

    if valid_mask.sum() < 2:
        return 0.0, 0.0

    # Calculate Spearman and Kendall correlations
    spearman_corr, _ = spearmanr(subset_ranking[valid_mask], reference_ranking[valid_mask])
    kendall_corr, _ = kendalltau(subset_ranking[valid_mask], reference_ranking[valid_mask])

    return spearman_corr, kendall_corr


def greedy_subset_selection(df, all_tasks, reference_ranking, max_tasks=10):
    """Greedy algorithm to find minimal subset that best reproduces ranking"""

    selected_tasks = []
    remaining_tasks = all_tasks.copy()
    correlations = []

    print("Greedy subset selection:")
    print("=" * 50)

    for i in range(min(max_tasks, len(all_tasks))):
        best_task = None
        best_corr = -1

        # Try adding each remaining task
        for task in remaining_tasks:
            candidate_subset = selected_tasks + [task]
            spearman_corr, _ = evaluate_subset_ranking(df, candidate_subset, reference_ranking)

            # Handle NaN correlations
            if not np.isnan(spearman_corr) and spearman_corr > best_corr:
                best_corr = spearman_corr
                best_task = task

        if best_task is not None:
            selected_tasks.append(best_task)
            remaining_tasks.remove(best_task)
            correlations.append(best_corr)

            print(f"Step {i + 1}: Added '{best_task}' -> Correlation: {best_corr:.4f}")

            # Stop if we achieve very high correlation
            if not np.isnan(best_corr) and best_corr > 0.95:
                print(f"High correlation achieved ({best_corr:.4f}), stopping early.")
                break
        else:
            break

    return selected_tasks, correlations


def exhaustive_small_subset_search(df, all_tasks, reference_ranking, max_size=5):
    """Exhaustive search for small subsets"""

    print(f"\nExhaustive search for subsets of size 1-{max_size}:")
    print("=" * 50)

    best_subsets = {}

    for size in range(1, min(max_size + 1, len(all_tasks) + 1)):
        best_corr = -1
        best_subset = None

        # Try all combinations of this size
        for subset in combinations(all_tasks, size):
            spearman_corr, _ = evaluate_subset_ranking(df, list(subset), reference_ranking)

            # Handle NaN correlations
            if not np.isnan(spearman_corr) and spearman_corr > best_corr:
                best_corr = spearman_corr
                best_subset = subset

        best_subsets[size] = (best_subset, best_corr)
        print(f"Size {size}: Best correlation = {best_corr:.4f}")
        if best_subset is not None:
            print(f"  Tasks: {list(best_subset)}")
        else:
            print(f"  Tasks: None found")
        print()

    return best_subsets


def analyze_task_importance(df, all_tasks, reference_ranking):
    """Analyze individual task importance for ranking prediction"""

    task_correlations = {}

    for task in all_tasks:
        task_ranking = df[task].rank(ascending=False, method="min", na_option="bottom")
        spearman_corr, _ = evaluate_subset_ranking(df, [task], reference_ranking)
        task_correlations[task] = spearman_corr

    # Sort by correlation
    sorted_tasks = sorted(task_correlations.items(), key=lambda x: x[1], reverse=True)

    print("Individual task correlations with overall ranking:")
    print("=" * 60)
    for task, corr in sorted_tasks[:15]:  # Top 15
        print(f"{task:<40}: {corr:.4f}")

    return task_correlations


def display_ranking_comparison_tables(df, best_subsets):
    """Display ranking comparison tables for each subset"""

    print("\n" + "=" * 80)
    print("RANKING COMPARISON TABLES")
    print("=" * 80)

    for size, (subset, corr) in best_subsets.items():
        print(f"\nSubset Size {size} (Correlation: {corr:.4f})")
        print(f"Tasks: {list(subset)}")
        print("-" * 80)

        # Calculate subset ranking with method='min' to handle ties consistently
        subset_avg = df[list(subset)].mean(axis=1, skipna=True)
        subset_ranking = subset_avg.rank(ascending=False, method="min", na_option="bottom")

        # Create comparison dataframe
        comparison_df = df[["model", "avg_all", "rank_all"]].copy()
        comparison_df[f"avg_subset_{size}"] = subset_avg
        comparison_df[f"rank_subset_{size}"] = subset_ranking
        comparison_df["rank_diff"] = comparison_df[f"rank_subset_{size}"] - comparison_df["rank_all"]

        # Sort by full ranking and display top 15
        comparison_df_sorted = comparison_df.sort_values("rank_all").head(15)

        print(f"{'Rank':<4} {'Model':<35} {'Full Avg':<8} {'Sub Avg':<8} {'Sub Rank':<8} {'Diff':<6}")
        print("-" * 80)

        for _, row in comparison_df_sorted.iterrows():
            model_short = row["model"][:34] if len(row["model"]) > 34 else row["model"]
            full_rank = int(row["rank_all"])
            subset_rank = int(row[f"rank_subset_{size}"]) if not pd.isna(row[f"rank_subset_{size}"]) else "N/A"
            rank_diff = int(row["rank_diff"]) if not pd.isna(row["rank_diff"]) else "N/A"
            full_avg = f"{row['avg_all']:.3f}"
            subset_avg = f"{row[f'avg_subset_{size}']:.3f}" if not pd.isna(row[f"avg_subset_{size}"]) else "N/A"

            # Add tie indicator for subset ranking
            tie_indicator = ""
            if isinstance(subset_rank, int):
                # Check if there are other models with the same subset rank
                same_rank_count = (comparison_df[f"rank_subset_{size}"] == subset_rank).sum()
                if same_rank_count > 1:
                    tie_indicator = "T"

            print(
                f"{full_rank:<4} {model_short:<35} {full_avg:<8} {subset_avg:<8} {subset_rank}{tie_indicator:<7} {rank_diff:<6}"
            )

        # Show some statistics about ranking differences
        valid_diffs = comparison_df["rank_diff"].dropna()
        if len(valid_diffs) > 0:
            print(f"\nRanking difference statistics:")
            print(f"  Mean absolute difference: {abs(valid_diffs).mean():.2f}")
            print(f"  Max absolute difference: {abs(valid_diffs).max():.0f}")
            print(f"  Models with exact rank match: {(valid_diffs == 0).sum()}")

            # Show tie information
            subset_ranks = comparison_df[f"rank_subset_{size}"].dropna()
            tied_ranks = subset_ranks[subset_ranks.duplicated(keep=False)].unique()
            if len(tied_ranks) > 0:
                print(f"  Tied ranks in subset: {sorted(tied_ranks.astype(int))}")
                print(f"  Number of models in ties: {len(subset_ranks[subset_ranks.isin(tied_ranks)])}")


def create_visualizations(df, all_tasks, reference_ranking, output_dir="plots"):
    """Create visualizations of the analysis"""

    Path(output_dir).mkdir(exist_ok=True)

    # 1. Correlation matrix between different rankings
    ranking_cols = ["rank_existing", "rank_new", "rank_all"]
    ranking_data = df[ranking_cols].corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(ranking_data, annot=True, cmap="RdYlBu_r", center=0, square=True, fmt=".3f")
    plt.title("Ranking Correlations Between Task Groups")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ranking_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Distribution of average scores
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    df["avg_existing"].hist(bins=20, alpha=0.7, color="blue")
    plt.title("Distribution of Existing Task Averages")
    plt.xlabel("Average Score")
    plt.ylabel("Count")

    plt.subplot(2, 2, 2)
    df["avg_new"].hist(bins=20, alpha=0.7, color="red")
    plt.title("Distribution of New Task Averages")
    plt.xlabel("Average Score")
    plt.ylabel("Count")

    plt.subplot(2, 2, 3)
    df["avg_all"].hist(bins=20, alpha=0.7, color="green")
    plt.title("Distribution of All Task Averages")
    plt.xlabel("Average Score")
    plt.ylabel("Count")

    plt.subplot(2, 2, 4)
    plt.scatter(df["avg_existing"], df["avg_new"], alpha=0.6)
    plt.xlabel("Existing Task Average")
    plt.ylabel("New Task Average")
    plt.title("Existing vs New Task Performance")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("SuperGLEBer Task Subset Analysis")
    print("=" * 50)

    # Load and process data
    existing_results, new_results = load_data()
    df = create_unified_dataframe(existing_results, new_results)
    df, existing_tasks, new_tasks, all_tasks = calculate_rankings_and_averages(df)

    print(f"Loaded data for {len(df)} models")
    print(f"Existing tasks: {len(existing_tasks)}")
    print(f"New tasks: {len(new_tasks)}")
    print(f"Total tasks: {len(all_tasks)}")
    print()

    # Display top models by overall ranking
    print("Top 10 models by overall average:")
    top_models = df.nlargest(10, "avg_all")[["model", "avg_all", "rank_all"]]
    for _, row in top_models.iterrows():
        print(f"{row['model']:<40}: {row['avg_all']:.4f} (rank {row['rank_all']:.0f})")
    print()

    # Analyze task importance
    task_correlations = analyze_task_importance(df, all_tasks, df["rank_all"])
    print()

    # Greedy subset selection
    selected_tasks, correlations = greedy_subset_selection(df, all_tasks, df["rank_all"])
    print()

    # Exhaustive search for small subsets
    best_subsets = exhaustive_small_subset_search(df, all_tasks, df["rank_all"], max_size=5)

    # Display ranking comparison tables
    display_ranking_comparison_tables(df, best_subsets)

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(
        f"Full correlation between existing and new task rankings: {spearmanr(df['rank_existing'], df['rank_new'], nan_policy='omit')[0]:.4f}"
    )
    print()
    print("Minimal subsets with high correlation (>0.9):")

    for size, (subset, corr) in best_subsets.items():
        if corr > 0.9:
            print(f"  Size {size} (correlation {corr:.4f}): {list(subset)}")

    print()
    print(f"Greedy selection achieved {correlations[-1]:.4f} correlation with {len(selected_tasks)} tasks:")
    print(f"  {selected_tasks}")

    # Create visualizations
    create_visualizations(df, all_tasks, df["rank_all"])

    # Save results
    results_summary = {
        "task_correlations": task_correlations,
        "greedy_selection": {"tasks": selected_tasks, "correlations": correlations},
        "best_subsets": {str(k): {"tasks": list(v[0]), "correlation": v[1]} for k, v in best_subsets.items()},
        "model_rankings": df[["model", "avg_all", "rank_all"]].to_dict("records"),
    }

    with open("subset_analysis_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nAnalysis complete! Results saved to 'subset_analysis_results.json'")
    print("Visualizations saved to 'plots/' directory")


if __name__ == "__main__":
    main()
