"""
Benchmarking Engine for Recommender Systems.
Evaluates multiple models side-by-side on held-out test interactions.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from .metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    mrr_at_k,
    catalog_coverage,
    intra_list_diversity
)


class ModelBenchmark:
    """Runs automated offline comparative benchmarking across recommender models."""

    def __init__(self, dataset, models: Dict[str, Any], k_values: List[int] = [5, 10]):
        self.dataset = dataset
        self.models = models
        self.k_values = k_values
        self.results_df: Optional[pd.DataFrame] = None
        self.results_dict: Dict[str, Dict[str, float]] = {}

    def run(self, max_test_users: int = 100) -> pd.DataFrame:
        """
        Evaluate each model on test users.
        """
        # Collect test users who have at least 1 ground truth item
        test_users = [u for u, items in self.dataset.user_test_pos.items() if len(items) > 0]
        if max_test_users and len(test_users) > max_test_users:
            test_users = test_users[:max_test_users]

        total_catalog = self.dataset.num_products
        summary_records = []

        max_k = max(self.k_values)

        for model_name, model in self.models.items():
            model_metrics = {f"P@{k}": [] for k in self.k_values}
            for k in self.k_values:
                model_metrics[f"R@{k}"] = []
                model_metrics[f"NDCG@{k}"] = []
                model_metrics[f"HR@{k}"] = []
                model_metrics[f"MRR@{k}"] = []

            all_recs_max_k = []
            all_diversity_scores = []

            for u_idx in test_users:
                user_id = self.dataset.idx2user[u_idx]
                gt_indices = set(self.dataset.user_test_pos[u_idx])
                gt_pids = set(self.dataset.idx2prod[p] for p in gt_indices)

                # Get recommendations
                recs = model.recommend(user_id, k=max_k, filter_interacted=True)
                rec_pids = [r["product_id"] for r in recs]
                rec_cats = [r.get("category", "") for r in recs]

                all_recs_max_k.append(rec_pids)
                all_diversity_scores.append(intra_list_diversity(rec_cats))

                for k in self.k_values:
                    model_metrics[f"P@{k}"].append(precision_at_k(rec_pids, gt_pids, k))
                    model_metrics[f"R@{k}"].append(recall_at_k(rec_pids, gt_pids, k))
                    model_metrics[f"NDCG@{k}"].append(ndcg_at_k(rec_pids, gt_pids, k))
                    model_metrics[f"HR@{k}"].append(hit_rate_at_k(rec_pids, gt_pids, k))
                    model_metrics[f"MRR@{k}"].append(mrr_at_k(rec_pids, gt_pids, k))

            # Aggregate averages
            record = {"Model": model_name}
            for metric_name, vals in model_metrics.items():
                record[metric_name] = round(float(np.mean(vals)), 4) if vals else 0.0

            cov = catalog_coverage(all_recs_max_k, total_catalog)
            record["Coverage"] = round(cov, 4)
            record["Diversity"] = round(float(np.mean(all_diversity_scores)), 4) if all_diversity_scores else 0.0

            summary_records.append(record)
            self.results_dict[model_name] = record

        self.results_df = pd.DataFrame(summary_records)
        return self.results_df

    def plot_comparison(self, output_path: str = "benchmark_results.png"):
        """Generate comparative bar plot visualization."""
        if self.results_df is None or self.results_df.empty:
            return

        df = self.results_df.set_index("Model")
        metrics_to_plot = ["NDCG@10", "HR@10", "P@10", "R@10", "Diversity"]
        available_metrics = [m for m in metrics_to_plot if m in df.columns]

        plt.figure(figsize=(10, 6), dpi=150)
        ax = df[available_metrics].plot(kind="bar", figsize=(11, 6), colormap="viridis", width=0.8)
        plt.title("Comparative Performance: Graph Recommenders vs Baselines", fontsize=14, fontweight="bold", pad=12)
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Model Architecture", fontsize=12)
        plt.xticks(rotation=15, ha="right", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Benchmark plot saved to {output_path}")
