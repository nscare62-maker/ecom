"""
Evaluation Metrics Suite for Recommendation Systems.
Implements:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- HitRate@K (HR@K)
- MRR@K (Mean Reciprocal Rank)
- Catalog Coverage
- Intra-List Diversity (ILD)
"""

from typing import List, Set, Dict, Any
import numpy as np


def precision_at_k(recommended: List[Any], ground_truth: Set[Any], k: int) -> float:
    """Precision at rank K."""
    if k <= 0 or not recommended:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return hits / float(k)


def recall_at_k(recommended: List[Any], ground_truth: Set[Any], k: int) -> float:
    """Recall at rank K."""
    if not ground_truth or not recommended:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return hits / float(len(ground_truth))


def hit_rate_at_k(recommended: List[Any], ground_truth: Set[Any], k: int) -> float:
    """Hit Rate at rank K (1 if at least one recommended item is in ground truth, else 0)."""
    if not ground_truth or not recommended:
        return 0.0
    rec_k = recommended[:k]
    return 1.0 if any(item in ground_truth for item in rec_k) else 0.0


def mrr_at_k(recommended: List[Any], ground_truth: Set[Any], k: int) -> float:
    """Mean Reciprocal Rank at rank K."""
    if not ground_truth or not recommended:
        return 0.0
    rec_k = recommended[:k]
    for rank, item in enumerate(rec_k, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommended: List[Any], ground_truth: Set[Any], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    if not ground_truth or not recommended:
        return 0.0
    rec_k = recommended[:k]

    # DCG calculation
    dcg = 0.0
    for rank, item in enumerate(rec_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 1)

    # IDCG calculation (ideal ranking)
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))

    return (dcg / idcg) if idcg > 0.0 else 0.0


def catalog_coverage(all_recommended_lists: List[List[Any]], total_catalog_items: int) -> float:
    """Fraction of unique items from catalog recommended across all users."""
    if total_catalog_items <= 0:
        return 0.0
    unique_items = set()
    for rec_list in all_recommended_lists:
        unique_items.update(rec_list)
    return len(unique_items) / float(total_catalog_items)


def intra_list_diversity(recommended_categories: List[str]) -> float:
    """
    Intra-List Diversity: Average pairwise distance between item categories in recommendation list.
    Distance = 1 if different category, 0 if identical.
    """
    n = len(recommended_categories)
    if n <= 1:
        return 0.0

    diff_pairs = 0
    total_pairs = n * (n - 1) / 2

    for i in range(n):
        for j in range(i + 1, n):
            if recommended_categories[i] != recommended_categories[j]:
                diff_pairs += 1

    return diff_pairs / total_pairs


def evaluate_user_recommendations(recommended_ids: List[str], ground_truth_ids: Set[str],
                                  k_values: List[int] = [5, 10]) -> Dict[str, float]:
    """Calculate all standard top-K metrics for a single user."""
    metrics = {}
    for k in k_values:
        metrics[f"Precision@{k}"] = precision_at_k(recommended_ids, ground_truth_ids, k)
        metrics[f"Recall@{k}"] = recall_at_k(recommended_ids, ground_truth_ids, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(recommended_ids, ground_truth_ids, k)
        metrics[f"HitRate@{k}"] = hit_rate_at_k(recommended_ids, ground_truth_ids, k)
        metrics[f"MRR@{k}"] = mrr_at_k(recommended_ids, ground_truth_ids, k)

    return metrics
