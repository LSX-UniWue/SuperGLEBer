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

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

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

        # Create individual plots

        # 1. Standard deviation comparison
        plt.figure(figsize=(10, 6))
        discrimination_df.boxplot(column="std", by="task_type")
        plt.title("Standard Deviation by Task Type")
        plt.xlabel("Task Type")
        plt.ylabel("Standard Deviation")
        plt.suptitle("")  # Remove default suptitle
        plt.tight_layout()
        plt.savefig("predictions/analysis/task_discrimination_std.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. Coefficient of variation
        plt.figure(figsize=(10, 6))
        discrimination_df.boxplot(column="cv", by="task_type")
        plt.title("Coefficient of Variation by Task Type")
        plt.xlabel("Task Type")
        plt.ylabel("CV (std/mean)")
        plt.suptitle("")  # Remove default suptitle
        plt.tight_layout()
        plt.savefig("predictions/analysis/task_discrimination_cv.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 3. Top discriminating existing tasks
        plt.figure(figsize=(12, 8))
        top_existing = discrimination_df[discrimination_df["task_type"] == "existing"].nlargest(10, "std")
        y_pos = np.arange(len(top_existing))
        plt.barh(y_pos, top_existing["std"], alpha=0.7, color="steelblue")
        plt.yticks(y_pos, top_existing["task"].tolist())
        plt.xlabel("Standard Deviation")
        plt.title("Top Discriminating Existing Tasks")
        plt.tight_layout()
        plt.savefig("predictions/analysis/task_discrimination_existing.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 4. Top discriminating new tasks
        plt.figure(figsize=(12, 8))
        top_new = discrimination_df[discrimination_df["task_type"] == "new"].nlargest(10, "std")
        y_pos = np.arange(len(top_new))
        plt.barh(y_pos, top_new["std"], alpha=0.7, color="orange")
        plt.yticks(y_pos, top_new["task"].tolist())
        plt.xlabel("Standard Deviation")
        plt.title("Top Discriminating New Tasks")
        plt.tight_layout()
        plt.savefig("predictions/analysis/task_discrimination_new.png", dpi=300, bbox_inches="tight")
        plt.show()

        return discrimination_df

    def analyze_task_correlations(self):
        """Analyze correlations between new and existing tasks."""
        print("\nAnalyzing task correlations...")

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

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

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

        # Collect all scores
        existing_scores = []
        new_scores = []

        for task in self.existing_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            existing_scores.extend(scores.tolist())

        for task in self.new_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce").dropna()
            new_scores.extend(scores.tolist())

        # Create individual plots

        # 1. Distribution comparison
        plt.figure(figsize=(12, 6))
        plt.hist(existing_scores, alpha=0.7, label="Existing Tasks", bins=30)
        plt.hist(new_scores, alpha=0.7, label="New Tasks", bins=30)
        plt.xlabel("Performance Score")
        plt.ylabel("Frequency")
        plt.title("Performance Score Distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig("predictions/analysis/performance_distributions_histogram.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. Box plots comparison
        plt.figure(figsize=(10, 6))
        box_data = [existing_scores, new_scores]
        bp = plt.boxplot(box_data)
        plt.xticks([1, 2], ["Existing", "New"])
        plt.ylabel("Performance Score")
        plt.title("Performance Score Distribution Comparison")
        plt.tight_layout()
        plt.savefig("predictions/analysis/performance_distributions_boxplot.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Task difficulty (mean scores) preparation
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

        # 3. Task difficulty comparison
        plt.figure(figsize=(10, 6))
        difficulty_df.boxplot(column="mean_score", by="task_type")
        plt.title("Task Difficulty (Mean Scores)")
        plt.xlabel("Task Type")
        plt.ylabel("Mean Performance")
        plt.suptitle("")  # Remove default suptitle
        plt.tight_layout()
        plt.savefig("predictions/analysis/performance_distributions_difficulty.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 4. Performance statistics comparison
        plt.figure(figsize=(12, 6))
        existing_stats = pd.Series(existing_scores).describe()
        new_stats = pd.Series(new_scores).describe()
        stats_comparison = pd.DataFrame({"Existing": existing_stats, "New": new_stats})
        stats_comparison.loc[["min", "25%", "50%", "75%", "max"]].plot(kind="bar")
        plt.title("Performance Statistics Comparison")
        plt.ylabel("Performance Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("predictions/analysis/performance_distributions_stats.png", dpi=300, bbox_inches="tight")
        plt.show()

        return difficulty_df

    def analyze_model_ranking_consistency(self):
        """Analyze how consistent model rankings are across tasks."""
        print("\nAnalyzing model ranking consistency...")

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

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

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

        # Prepare data matrix (models x tasks)
        task_data = {}
        all_tasks = self.existing_tasks + self.new_tasks

        for task in all_tasks:
            scores = pd.to_numeric(self.merged_df[task], errors="coerce")
            if not scores.isna().all():
                task_data[task] = scores.fillna(scores.mean())  # Fill NaN with mean

        # Create data matrix
        data_matrix = pd.DataFrame(task_data)

        # Standardize the data for task clustering
        scaler = StandardScaler()
        data_scaled_tasks = scaler.fit_transform(data_matrix.T)  # Transpose to have tasks as rows

        # Perform hierarchical clustering for tasks
        linkage_matrix_tasks = linkage(data_scaled_tasks, method="ward")

        # Create task dendrogram
        plt.figure(figsize=(15, 10))
        dendrogram(linkage_matrix_tasks, labels=data_matrix.columns, leaf_rotation=90)
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

        # Standardize the data for model clustering
        data_scaled_models = scaler.fit_transform(data_matrix)  # Models as rows

        # Perform hierarchical clustering for models
        linkage_matrix_models = linkage(data_scaled_models, method="ward")

        # Create model dendrogram
        plt.figure(figsize=(15, 10))
        model_labels = [model.split("/")[-1] if "/" in model else model for model in self.merged_df["model"]]
        dendrogram(linkage_matrix_models, labels=model_labels, leaf_rotation=90)
        plt.title("Model Clustering Based on Task Performance Patterns")
        plt.xlabel("Models")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig("predictions/analysis/model_clustering.png", dpi=300, bbox_inches="tight")
        plt.show()

        # K-means clustering for tasks
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled_tasks)

        # Analyze task clusters
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

        # K-means clustering for models
        kmeans_models = KMeans(n_clusters=min(n_clusters, len(self.merged_df)), random_state=42)
        model_cluster_labels = kmeans_models.fit_predict(data_scaled_models)

        # Analyze model clusters
        model_cluster_df = pd.DataFrame(
            {
                "model": model_labels,
                "cluster": model_cluster_labels,
            }
        )

        print("\nModel clusters:")
        for i in range(min(n_clusters, len(self.merged_df))):
            cluster_models = model_cluster_df[model_cluster_df["cluster"] == i]
            print(f"\nCluster {i}:")
            print(f"  Models: {cluster_models['model'].tolist()}")

        return cluster_df, model_cluster_df

    def perform_pca_analysis(self):
        """Perform PCA to understand task relationships in lower dimensional space."""
        print("\nPerforming PCA analysis...")

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

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

        # Create individual plots

        # 1. Explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("predictions/analysis/pca_explained_variance.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 2. Task loadings in PC1 vs PC2
        plt.figure(figsize=(14, 10))
        loadings = pca.components_[:2].T
        for i, task in enumerate(data_matrix.columns):
            color = "red" if task in self.new_tasks else "blue"
            plt.scatter(loadings[i, 0], loadings[i, 1], c=color, alpha=0.7, s=50)
            plt.annotate(task, (loadings[i, 0], loadings[i, 1]), fontsize=8, alpha=0.8)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title("Task Loadings in PCA Space (Red=New, Blue=Existing)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("predictions/analysis/pca_task_loadings.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 3. Model positions in PC space
        plt.figure(figsize=(12, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, s=50)
        for i, model in enumerate(self.merged_df["model"]):
            plt.annotate(model.split("/")[-1][:15], (pca_result[i, 0], pca_result[i, 1]), fontsize=8, alpha=0.7)

        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.title("Models in PCA Space")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("predictions/analysis/pca_models.png", dpi=300, bbox_inches="tight")
        plt.show()

        return pca, pca_result

    def generate_summary_report(self, discrimination_df, corr_df, difficulty_df):
        """Generate a summary report of all analyses."""
        print("\n" + "=" * 60)
        print("SUPERGLEBER TASK ANALYSIS SUMMARY REPORT")
        print("=" * 60)

        if self.merged_df is None:
            raise ValueError("Data must be loaded first. Call load_and_preprocess_data().")

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
        cluster_df, model_cluster_df = self.perform_task_clustering()
        pca, pca_result = self.perform_pca_analysis()

        # Generate summary report
        self.generate_summary_report(discrimination_df, corr_df, difficulty_df)

        print(f"\nAnalysis complete! All individual plots saved to predictions/analysis/")
        print("Generated plots:")
        print("- task_discrimination_std.png")
        print("- task_discrimination_cv.png")
        print("- task_discrimination_existing.png")
        print("- task_discrimination_new.png")
        print("- task_correlations_heatmap.png")
        print("- performance_distributions_histogram.png")
        print("- performance_distributions_boxplot.png")
        print("- performance_distributions_difficulty.png")
        print("- performance_distributions_stats.png")
        print("- ranking_consistency.png")
        print("- task_clustering.png")
        print("- model_clustering.png")
        print("- pca_explained_variance.png")
        print("- pca_task_loadings.png")
        print("- pca_models.png")

        return {
            "discrimination": discrimination_df,
            "correlations": corr_df,
            "difficulty": difficulty_df,
            "rank_correlations": rank_corr,
            "clusters": cluster_df,
            "model_clusters": model_cluster_df,
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
