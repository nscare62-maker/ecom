"""
Baseline Recommendation Algorithms for Comparative Benchmarking.
Includes:
- Most Popular Items Recommender (Unpersonalized baseline)
- Item-based Collaborative Filtering (ItemCF / Cosine Co-occurrence)
- Matrix Factorization / Truncated SVD (Latent Factor Model)
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    """Ranks items solely by aggregate interaction volume and weight."""

    def __init__(self):
        self.popular_items: List[Dict[str, Any]] = []
        self.dataset = None

    def fit(self, dataset):
        self.dataset = dataset
        train_df = dataset.train_df

        pop = train_df.groupby("product_id")["weight"].sum().reset_index()
        pop = pop.sort_values(by="weight", ascending=False)

        products_meta = dataset.products_df.set_index("product_id").to_dict("index")

        self.popular_items = []
        for _, row in pop.iterrows():
            pid = row["product_id"]
            meta = products_meta.get(pid, {})
            self.popular_items.append({
                "product_id": pid,
                "name": meta.get("name", pid),
                "category": meta.get("category", "N/A"),
                "price": float(meta.get("price", 0.0)),
                "score": float(row["weight"]),
                "model": "Popularity"
            })

    def recommend(self, user_id: str, k: int = 10, filter_interacted: bool = True) -> List[Dict[str, Any]]:
        if not self.popular_items:
            return []

        seen_items = set()
        if filter_interacted and self.dataset and user_id in self.dataset.user2idx:
            u_idx = self.dataset.user2idx[user_id]
            seen_items = set(self.dataset.idx2prod[p] for p in self.dataset.user_train_pos.get(u_idx, []))

        recs = []
        for item in self.popular_items:
            if item["product_id"] in seen_items:
                continue
            item_copy = dict(item)
            item_copy["rank"] = len(recs) + 1
            recs.append(item_copy)
            if len(recs) >= k:
                break

        return recs


class ItemCFRecommender:
    """Item-Item Collaborative Filtering using cosine similarity on interaction matrix."""

    def __init__(self):
        self.dataset = None
        self.item_sim_matrix: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[np.ndarray] = None

    def fit(self, dataset):
        self.dataset = dataset
        n_users = dataset.num_users
        n_items = dataset.num_products

        # Build user-item dense interaction matrix
        R = np.zeros((n_users, n_items), dtype=np.float32)
        for _, row in dataset.train_df.iterrows():
            u = int(row["user_idx"])
            p = int(row["product_idx"])
            w = float(row["weight"])
            R[u, p] += w

        self.user_item_matrix = R

        # Cosine similarity between items (columns of R)
        item_features = R.T
        self.item_sim_matrix = cosine_similarity(item_features)
        np.fill_diagonal(self.item_sim_matrix, 0.0)

    def recommend(self, user_id: str, k: int = 10, filter_interacted: bool = True) -> List[Dict[str, Any]]:
        if self.dataset is None or user_id not in self.dataset.user2idx:
            return []

        u_idx = self.dataset.user2idx[user_id]
        user_history = self.user_item_matrix[u_idx]  # shape: (n_items,)

        if np.sum(user_history) == 0:
            return []

        # Predict score: dot product of user history with item-item similarity
        scores = np.dot(self.item_sim_matrix, user_history)

        if filter_interacted and u_idx in self.dataset.user_train_pos:
            seen_items = self.dataset.user_train_pos[u_idx]
            scores[seen_items] = -1e9

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, p_idx in enumerate(top_indices, start=1):
            prod_id = self.dataset.idx2prod[p_idx]
            prod_row = self.dataset.products_df[self.dataset.products_df["product_id"] == prod_id].iloc[0]
            results.append({
                "rank": rank,
                "product_id": prod_id,
                "name": prod_row["name"],
                "category": prod_row["category"],
                "price": float(prod_row["price"]),
                "score": float(scores[p_idx]),
                "model": "ItemCF"
            })

        return results


class MatrixFactorizationRecommender:
    """Matrix Factorization using Truncated SVD on user-item interaction matrix."""

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.dataset = None
        self.svd: Optional[TruncatedSVD] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, dataset):
        self.dataset = dataset
        n_users = dataset.num_users
        n_items = dataset.num_products

        R = np.zeros((n_users, n_items), dtype=np.float32)
        for _, row in dataset.train_df.iterrows():
            u = int(row["user_idx"])
            p = int(row["product_idx"])
            w = float(row["weight"])
            R[u, p] += w

        n_comp = min(self.n_components, min(n_users, n_items) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.user_factors = self.svd.fit_transform(R)  # (n_users, n_comp)
        self.item_factors = self.svd.components_       # (n_comp, n_items)

    def recommend(self, user_id: str, k: int = 10, filter_interacted: bool = True) -> List[Dict[str, Any]]:
        if self.dataset is None or user_id not in self.dataset.user2idx:
            return []

        u_idx = self.dataset.user2idx[user_id]
        u_factor = self.user_factors[u_idx]
        scores = np.dot(u_factor, self.item_factors)

        if filter_interacted and u_idx in self.dataset.user_train_pos:
            seen_items = self.dataset.user_train_pos[u_idx]
            scores[seen_items] = -1e9

        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for rank, p_idx in enumerate(top_indices, start=1):
            prod_id = self.dataset.idx2prod[p_idx]
            prod_row = self.dataset.products_df[self.dataset.products_df["product_id"] == prod_id].iloc[0]
            results.append({
                "rank": rank,
                "product_id": prod_id,
                "name": prod_row["name"],
                "category": prod_row["category"],
                "price": float(prod_row["price"]),
                "score": float(scores[p_idx]),
                "model": "MatrixFactorization"
            })

        return results
