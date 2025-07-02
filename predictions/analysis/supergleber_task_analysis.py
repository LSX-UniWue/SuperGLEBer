#!/usr/bin/env python3
"""
SuperGLEBer Task Analysis: Comparing New and Existing Tasks

This script analyzes the relationship between new and existing tasks in the SuperGLEBer benchmark,
including task discrimination power, correlations, and performance patterns.
"""

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SuperGLEBerAnalyzer:
    """Main analyzer class for SuperGLEBer benchmark results."""

    def __init__(self, existing_results_path, new_results_path):
        """Initialize with paths to results files."""
        self.existing_results_path = existing_results_path
        self.new_results_path = new_results_path
        self.existing_df = None
        self.new_df = None
        self.merged_df = None
        self.existing_tasks = []
        self.new_tasks = []

    def load_and_preprocess_data(self):
        """Load and preprocess the benchmark results."""
        print("Loading and preprocessing data...")

        # Load existing tasks results
        with open(self.existing_results_path, "r") as f:
            existing_data = json.load(f)

        # Load new tasks results
        with open(self.new_results_path, "r") as f:
            new_data = json.load(f)

        # Convert to DataFrames
        self.existing_df = pd.DataFrame(existing_data)

        # Process new tasks data (extract metrics from nested structure)
        new_processed = []
        for entry in new_data:
            row = {"model": entry["model"]}
            for task, info in entry.items():
                if task != "model" and isinstance(info, dict) and "metric" in info:
                    row[task] = float(info["metric"])
            new_processed.append(row)

        self.new_df = pd.DataFrame(new_processed)

        # Get task lists (excluding metadata columns)
        existing_meta_cols = ["model", "team", "team_url", "setting", "model_type", "model_arch", "num_params"]
        self.existing_tasks = [col for col in self.existing_df.columns if col not in existing_meta_cols]
        self.new_tasks = [col for col in self.new_df.columns if col != "model"]

        # Clean model names for better matching
        self.existing_df["model_clean"] = self.existing_df["model"].str.strip()
        self.new_df["model_clean"] = self.new_df["model"].apply(self._clean_model_name)

        # Merge datasets
        self.merged_df = pd.merge(self.existing_df, self.new_df, on="model_clean", how="inner", suffixes=("", "_new"))

        print(f"Loaded {len(self.existing_df)} existing results and {len(self.new_df)} new results")
        print(f"Successfully merged {len(self.merged_df)} models")
        print(f"Existing tasks: {len(self.existing_tasks)}")
        print(f"New tasks: {len(self.new_tasks)}")

    def _clean_model_name(self, model_name):
        """Clean model names for better matching."""
        # Handle path-like model names
        if "/" in model_name and "models_hf" in model_name:
            # Extract the key part for matching
            parts = model_name.split("/")
            for part in parts:
                if any(key in part.lower() for key in ["llam", "gbert", "bert", "gpt", "bloom"]):
                    return part
        return model_name.strip()

    def calculate_task_discrimination_power(self):
        """Calculate how well each task discriminates between models."""
        print("\nCalculating task discrimination power...")

        discrimination_data = []

        # Calculate for existing tasks
        for task in self.existing_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            if len(scores) > 1:
                discrimination_data.append(
                    {
                        "task": task,
                        "task_type": "existing",
                        "std": scores.std(),
                        "cv": scores.std() / scores.mean() if scores.mean() != 0 else 0,
                        "range": scores.max() - scores.min(),
                        "mean": scores.mean(),
                        "n_models": len(scores),
                    }
                )

        # Calculate for new tasks
        for task in self.new_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            if len(scores) > 1:
                discrimination_data.append(
                    {
                        "task": task,
                        "task_type": "new",
                        "std": scores.std(),
                        "cv": scores.std() / scores.mean() if scores.mean() != 0 else 0,
                        "range": scores.max() - scores.min(),
                        "mean": scores.mean(),
                        "n_models": len(scores),
                    }
                )

        discrimination_df = pd.DataFrame(discrimination_data)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Task Discrimination Power Analysis", fontsize=16)

        # Standard deviation comparison
        ax1 = axes[0, 0]
        discrimination_df.boxplot(column="std", by="task_type", ax=ax1)
        ax1.set_title("Standard Deviation by Task Type")
        ax1.set_xlabel("Task Type")
        ax1.set_ylabel("Standard Deviation")

        # Coefficient of variation
        ax2 = axes[0, 1]
        discrimination_df.boxplot(column="cv", by="task_type", ax=ax2)
        ax2.set_title("Coefficient of Variation by Task Type")
        ax2.set_xlabel("Task Type")
        ax2.set_ylabel("CV (std/mean)")

        # Top discriminating tasks
        ax3 = axes[1, 0]
        top_existing = discrimination_df[discrimination_df["task_type"] == "existing"].nlargest(10, "std")
        top_new = discrimination_df[discrimination_df["task_type"] == "new"].nlargest(10, "std")

        y_pos = np.arange(len(top_existing))
        ax3.barh(y_pos, top_existing["std"], alpha=0.7, label="Existing")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(top_existing["task"], fontsize=8)
        ax3.set_xlabel("Standard Deviation")
        ax3.set_title("Top Discriminating Existing Tasks")

        # Top new tasks
        ax4 = axes[1, 1]
        y_pos = np.arange(len(top_new))
        ax4.barh(y_pos, top_new["std"], alpha=0.7, label="New", color="orange")
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top_new["task"], fontsize=8)
        ax4.set_xlabel("Standard Deviation")
        ax4.set_title("Top Discriminating New Tasks")

        plt.tight_layout()
        plt.savefig("predictions/analysis/task_discrimination_power.png", dpi=300, bbox_inches="tight")
        plt.show()

        return discrimination_df

    def analyze_task_correlations(self):
        """Analyze correlations between new and existing tasks."""
        print("\nAnalyzing task correlations...")

        # Prepare correlation matrix data
        correlation_data = {}

        # Get scores for all tasks
        task_scores = {}
        for task in self.existing_tasks + self.new_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce")
            if not scores.isna().all():
                task_scores[task] = scores

        # Calculate correlation matrix between new and existing tasks
        correlations = []
        for new_task in self.new_tasks:
            if new_task in task_scores:
                for existing_task in self.existing_tasks:
                    if existing_task in task_scores:
                        # Calculate correlation only for models with both scores
                        common_idx = ~(task_scores[new_task].isna() | task_scores[existing_task].isna())
                        if common_idx.sum() > 3:  # Need at least 4 data points
                            corr, p_value = stats.pearsonr(
                                task_scores[new_task][common_idx], task_scores[existing_task][common_idx]
                            )
                            correlations.append(
                                {
                                    "new_task": new_task,
                                    "existing_task": existing_task,
                                    "correlation": corr,
                                    "p_value": p_value,
                                    "n_models": common_idx.sum(),
                                }
                            )

        corr_df = pd.DataFrame(correlations)

        # Create correlation heatmap
        pivot_corr = corr_df.pivot(index="new_task", columns="existing_task", values="correlation")

        plt.figure(figsize=(20, 8))
        mask = pivot_corr.isna()
        sns.heatmap(
            pivot_corr,
            annot=True,
            cmap="RdBu_r",
            center=0,
            mask=mask,
            fmt=".2f",
            cbar_kws={"label": "Pearson Correlation"},
        )
        plt.title("Correlations between New and Existing Tasks")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("predictions/analysis/task_correlations_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Find best correlations for each new task
        print("\nBest correlations for each new task:")
        for new_task in self.new_tasks:
            task_corrs = corr_df[corr_df["new_task"] == new_task].copy()
            if not task_corrs.empty:
                task_corrs = task_corrs[task_corrs["p_value"] < 0.05]  # Only significant
                if not task_corrs.empty:
                    best_corr = task_corrs.loc[task_corrs["correlation"].abs().idxmax()]
                    print(
                        f"  {new_task}: best corr with {best_corr['existing_task']} (r={best_corr['correlation']:.3f}, p={best_corr['p_value']:.3f})"
                    )

        return corr_df

    def analyze_performance_distributions(self):
        """Analyze performance distributions across task types."""
        print("\nAnalyzing performance distributions...")

        # Collect all scores
        existing_scores = []
        new_scores = []

        for task in self.existing_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            existing_scores.extend(scores.tolist())

        for task in self.new_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            new_scores.extend(scores.tolist())

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Performance Distribution Analysis", fontsize=16)

        # Distribution comparison
        ax1 = axes[0, 0]
        ax1.hist(existing_scores, alpha=0.7, label="Existing Tasks", bins=30)
        ax1.hist(new_scores, alpha=0.7, label="New Tasks", bins=30)
        ax1.set_xlabel("Performance Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Score Distributions")
        ax1.legend()

        # Box plots
        ax2 = axes[0, 1]
        ax2.boxplot([existing_scores, new_scores], labels=["Existing", "New"])
        ax2.set_ylabel("Performance Score")
        ax2.set_title("Score Distribution Comparison")

        # Task difficulty (mean scores)
        ax3 = axes[1, 0]
        task_means = []
        task_names = []
        task_types = []

        for task in self.existing_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            if len(scores) > 0:
                task_means.append(scores.mean())
                task_names.append(task)
                task_types.append("existing")

        for task in self.new_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            if len(scores) > 0:
                task_means.append(scores.mean())
                task_names.append(task)
                task_types.append("new")

        difficulty_df = pd.DataFrame({"task": task_names, "mean_score": task_means, "task_type": task_types})

        difficulty_df.boxplot(column="mean_score", by="task_type", ax=ax3)
        ax3.set_title("Task Difficulty (Mean Scores)")
        ax3.set_xlabel("Task Type")
        ax3.set_ylabel("Mean Performance")

        # Performance range comparison
        ax4 = axes[1, 1]
        existing_stats = pd.Series(existing_scores).describe()
        new_stats = pd.Series(new_scores).describe()

        stats_comparison = pd.DataFrame({"Existing": existing_stats, "New": new_stats})

        stats_comparison.loc[["min", "25%", "50%", "75%", "max"]].plot(kind="bar", ax=ax4)
        ax4.set_title("Performance Statistics Comparison")
        ax4.set_ylabel("Performance Score")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("predictions/analysis/performance_distributions.png", dpi=300, bbox_inches="tight")
        plt.show()

        return difficulty_df

    def analyze_model_ranking_consistency(self):
        """Analyze how consistent model rankings are across tasks."""
        print("\nAnalyzing model ranking consistency...")

        # Calculate rankings for each task
        rankings = {}

        # Existing tasks rankings
        for task in self.existing_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce")
            valid_scores = scores.dropna()
            if len(valid_scores) > 1:
                rankings[task] = valid_scores.rank(ascending=False, method="average")

        # New tasks rankings
        for task in self.new_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce")
            valid_scores = scores.dropna()
            if len(valid_scores) > 1:
                rankings[task] = valid_scores.rank(ascending=False, method="average")

        # Create rankings DataFrame
        rankings_df = pd.DataFrame(rankings)

        # Calculate rank correlations
        rank_correlations = rankings_df.corr(method="spearman")

        # Focus on new vs existing correlations
        new_vs_existing_corr = rank_correlations.loc[self.new_tasks, self.existing_tasks]

        plt.figure(figsize=(20, 8))
        sns.heatmap(
            new_vs_existing_corr,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".2f",
            cbar_kws={"label": "Spearman Rank Correlation"},
        )
        plt.title("Model Ranking Consistency: New vs Existing Tasks")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("predictions/analysis/ranking_consistency.png", dpi=300, bbox_inches="tight")
        plt.show()

        return rank_correlations

    def perform_task_clustering(self):
        """Perform clustering analysis on tasks based on model performance patterns."""
        print("\nPerforming task clustering analysis...")

        # Prepare data matrix (models x tasks)
        task_data = {}
        all_tasks = self.existing_tasks + self.new_tasks

        for task in all_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce")
            if not scores.isna().all():
                task_data[task] = scores.fillna(scores.mean())  # Fill NaN with mean

        # Create data matrix
        data_matrix = pd.DataFrame(task_data)

        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix.T)  # Transpose to have tasks as rows

        # Perform hierarchical clustering
        linkage_matrix = linkage(data_scaled, method="ward")

        # Create dendrogram
        plt.figure(figsize=(15, 10))
        dendrogram(linkage_matrix, labels=data_matrix.columns, leaf_rotation=90)
        plt.title("Task Clustering Based on Model Performance Patterns")
        plt.xlabel("Tasks")
        plt.ylabel("Distance")

        # Color code new vs existing tasks
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            if lbl.get_text() in self.new_tasks:
                lbl.set_color("red")

        plt.tight_layout()
        plt.savefig("predictions/analysis/task_clustering.png", dpi=300, bbox_inches="tight")
        plt.show()

        # K-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled)

        # Analyze clusters
        cluster_df = pd.DataFrame(
            {
                "task": data_matrix.columns,
                "cluster": cluster_labels,
                "task_type": ["new" if task in self.new_tasks else "existing" for task in data_matrix.columns],
            }
        )

        print("\nTask clusters:")
        for i in range(n_clusters):
            cluster_tasks = cluster_df[cluster_df["cluster"] == i]
            print(f"\nCluster {i}:")
            print(f"  New tasks: {cluster_tasks[cluster_tasks['task_type'] == 'new']['task'].tolist()}")
            print(f"  Existing tasks: {cluster_tasks[cluster_tasks['task_type'] == 'existing']['task'].tolist()}")

        return cluster_df

    def perform_pca_analysis(self):
        """Perform PCA to understand task relationships in lower dimensional space."""
        print("\nPerforming PCA analysis...")

        # Prepare data matrix
        task_data = {}
        all_tasks = self.existing_tasks + self.new_tasks

        for task in all_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce")
            if not scores.isna().all():
                task_data[task] = scores.fillna(scores.mean())

        data_matrix = pd.DataFrame(task_data)

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)

        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)

        # Create PCA plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Explained variance
        ax1 = axes[0]
        ax1.plot(np.cumsum(pca.explained_variance_ratio_))
        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Cumulative Explained Variance")
        ax1.set_title("PCA Explained Variance")
        ax1.grid(True)

        # Task loadings in PC1 vs PC2
        ax2 = axes[1]
        loadings = pca.components_[:2].T
        for i, task in enumerate(data_matrix.columns):
            color = "red" if task in self.new_tasks else "blue"
            ax2.scatter(loadings[i, 0], loadings[i, 1], c=color, alpha=0.7)
            ax2.annotate(task, (loadings[i, 0], loadings[i, 1]), fontsize=8, alpha=0.7)

        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax2.set_title("Task Loadings in PCA Space")
        ax2.grid(True)

        # Model positions in PC space
        ax3 = axes[2]
        ax3.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        for i, model in enumerate(self.merged_df["model"]):
            ax3.annotate(model.split("/")[-1][:10], (pca_result[i, 0], pca_result[i, 1]), fontsize=6, alpha=0.7)

        ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax3.set_title("Models in PCA Space")
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig("predictions/analysis/pca_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        return pca, pca_result

    def generate_summary_report(self, discrimination_df, corr_df, difficulty_df):
        """Generate a summary report of all analyses."""
        print("\n" + "=" * 60)
        print("SUPERGLEBER TASK ANALYSIS SUMMARY REPORT")
        print("=" * 60)

        print(f"\nDATASET OVERVIEW:")
        print(f"- Models analyzed: {len(self.merged_df)}")
        print(f"- Existing tasks: {len(self.existing_tasks)}")
        print(f"- New tasks: {len(self.new_tasks)}")

        print(f"\nTASK DISCRIMINATION POWER:")
        existing_disc = discrimination_df[discrimination_df["task_type"] == "existing"]["std"]
        new_disc = discrimination_df[discrimination_df["task_type"] == "new"]["std"]
        print(f"- Existing tasks std: {existing_disc.mean():.3f} ± {existing_disc.std():.3f}")
        print(f"- New tasks std: {new_disc.mean():.3f} ± {new_disc.std():.3f}")

        # Most discriminating tasks
        top_existing = discrimination_df[discrimination_df["task_type"] == "existing"].nlargest(3, "std")
        top_new = discrimination_df[discrimination_df["task_type"] == "new"].nlargest(3, "std")

        print(f"\nMOST DISCRIMINATING TASKS:")
        print("Existing:")
        for _, row in top_existing.iterrows():
            print(f"  - {row['task']}: std={row['std']:.3f}")
        print("New:")
        for _, row in top_new.iterrows():
            print(f"  - {row['task']}: std={row['std']:.3f}")

        print(f"\nSTRONGEST CORRELATIONS (NEW vs EXISTING):")
        significant_corrs = corr_df[(corr_df["p_value"] < 0.05) & (corr_df["correlation"].abs() > 0.5)]
        top_corrs = significant_corrs.nlargest(5, "correlation")
        for _, row in top_corrs.iterrows():
            print(f"  - {row['new_task']} ↔ {row['existing_task']}: r={row['correlation']:.3f}")

        print(f"\nTASK DIFFICULTY:")
        existing_diff = difficulty_df[difficulty_df["task_type"] == "existing"]["mean_score"]
        new_diff = difficulty_df[difficulty_df["task_type"] == "new"]["mean_score"]
        print(f"- Existing tasks mean: {existing_diff.mean():.3f}")
        print(f"- New tasks mean: {new_diff.mean():.3f}")

        # Hardest and easiest tasks
        hardest_new = difficulty_df[difficulty_df["task_type"] == "new"].nsmallest(3, "mean_score")
        easiest_new = difficulty_df[difficulty_df["task_type"] == "new"].nlargest(3, "mean_score")

        print(f"\nHARDEST NEW TASKS:")
        for _, row in hardest_new.iterrows():
            print(f"  - {row['task']}: {row['mean_score']:.3f}")

        print(f"\nEASIEST NEW TASKS:")
        for _, row in easiest_new.iterrows():
            print(f"  - {row['task']}: {row['mean_score']:.3f}")

        print("\n" + "=" * 60)

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting SuperGLEBer Task Analysis...")

        # Load and preprocess data
        self.load_and_preprocess_data()

        # Run all analyses
        discrimination_df = self.calculate_task_discrimination_power()
        corr_df = self.analyze_task_correlations()
        difficulty_df = self.analyze_performance_distributions()
        rank_corr = self.analyze_model_ranking_consistency()
        cluster_df = self.perform_task_clustering()
        pca, pca_result = self.perform_pca_analysis()

        # Generate summary report
        self.generate_summary_report(discrimination_df, corr_df, difficulty_df)

        print(f"\nAnalysis complete! All plots saved to predictions/analysis/")

        return {
            "discrimination": discrimination_df,
            "correlations": corr_df,
            "difficulty": difficulty_df,
            "rank_correlations": rank_corr,
            "clusters": cluster_df,
            "pca": pca,
            "pca_result": pca_result,
        }


def main():
    """Main function to run the analysis."""
    # Initialize analyzer
    analyzer = SuperGLEBerAnalyzer(
        existing_results_path="predictions/analysis/results_existing_tasks.json",
        new_results_path="predictions/analysis/results_new_tasks.json",
    )

    # Run full analysis
    results = analyzer.run_full_analysis()

    return results


if __name__ == "__main__":
    results = main()
