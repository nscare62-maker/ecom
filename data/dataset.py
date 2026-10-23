"""
Dataset Loader, ID Indexer, and Train/Test Splitter for Graph Recommendation.
Supports bipartite user-product interactions and heterogeneous graph metadata.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class RecommendationDataset:
    """
    Manages user/product entity mapping, interaction indexing, train/test splitting,
    sparse adjacency matrices for GNN message passing, and BPR training batches.
    """

    def __init__(self, users_df: pd.DataFrame, products_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.users_df = users_df.copy()
        self.products_df = products_df.copy()
        self.interactions_df = interactions_df.copy()

        # Build ID mappings
        self.user_ids = sorted(self.users_df["user_id"].unique().tolist())
        self.product_ids = sorted(self.products_df["product_id"].unique().tolist())
        self.categories = sorted(self.products_df["category"].unique().tolist())

        self.num_users = len(self.user_ids)
        self.num_products = len(self.product_ids)
        self.num_categories = len(self.categories)

        self.user2idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.idx2user = {i: uid for i, uid in enumerate(self.user_ids)}

        self.prod2idx = {pid: i for i, pid in enumerate(self.product_ids)}
        self.idx2prod = {i: pid for i, pid in enumerate(self.product_ids)}

        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx2cat = {i: cat for i, cat in enumerate(self.categories)}

        # Add index columns
        self.interactions_df["user_idx"] = self.interactions_df["user_id"].map(self.user2idx)
        self.interactions_df["product_idx"] = self.interactions_df["product_id"].map(self.prod2idx)
        self.products_df["product_idx"] = self.products_df["product_id"].map(self.prod2idx)
        self.products_df["category_idx"] = self.products_df["category"].map(self.cat2idx)
        self.users_df["user_idx"] = self.users_df["user_id"].map(self.user2idx)

        # Drop any records with unmapped entities
        self.interactions_df = self.interactions_df.dropna(subset=["user_idx", "product_idx"]).reset_index(drop=True)
        self.interactions_df["user_idx"] = self.interactions_df["user_idx"].astype(int)
        self.interactions_df["product_idx"] = self.interactions_df["product_idx"].astype(int)

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.user_train_pos: Dict[int, List[int]] = {}
        self.user_test_pos: Dict[int, List[int]] = {}

    @classmethod
    def from_csv_dir(cls, dir_path: str = "data/raw"):
        """Load dataset from saved CSV files."""
        users_df = pd.read_csv(os.path.join(dir_path, "users.csv"))
        products_df = pd.read_csv(os.path.join(dir_path, "products.csv"))
        interactions_df = pd.read_csv(os.path.join(dir_path, "interactions.csv"))
        return cls(users_df, products_df, interactions_df)

    def train_test_split(self, test_ratio: float = 0.20, strategy: str = "temporal") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split interactions into train and test sets.
        Strategy:
            - 'temporal': sorts interactions by timestamp and reserves the last k interactions per user for testing.
            - 'random': uniform random split.
        """
        if strategy == "temporal":
            sorted_df = self.interactions_df.sort_values(by=["user_idx", "timestamp"])
            train_rows = []
            test_rows = []

            for u_idx, group in sorted_df.groupby("user_idx"):
                n = len(group)
                if n <= 2:
                    train_rows.append(group)
                else:
                    n_test = max(1, int(n * test_ratio))
                    train_rows.append(group.iloc[:-n_test])
                    test_rows.append(group.iloc[-n_test:])

            self.train_df = pd.concat(train_rows).reset_index(drop=True)
            self.test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame(columns=self.interactions_df.columns)
        else:
            shuffled = self.interactions_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            n_train = int(len(shuffled) * (1 - test_ratio))
            self.train_df = shuffled.iloc[:n_train].reset_index(drop=True)
            self.test_df = shuffled.iloc[n_train:].reset_index(drop=True)

        # Build positive interaction sets for fast evaluation and negative sampling
        self.user_train_pos = {}
        for u, p in zip(self.train_df["user_idx"], self.train_df["product_idx"]):
            self.user_train_pos.setdefault(u, set()).add(p)
        self.user_train_pos = {u: list(items) for u, items in self.user_train_pos.items()}

        self.user_test_pos = {}
        for u, p in zip(self.test_df["user_idx"], self.test_df["product_idx"]):
            self.user_test_pos.setdefault(u, set()).add(p)
        self.user_test_pos = {u: list(items) for u, items in self.user_test_pos.items()}

        return self.train_df, self.test_df

    def get_norm_adj_mat(self) -> torch.sparse.FloatTensor:
        """
        Construct the normalized symmetric adjacency matrix for LightGCN message passing.
        A = [[0, R], [R^T, 0]]
        D^-0.5 * A * D^-0.5
        """
        if self.train_df is None:
            raise ValueError("Run train_test_split() before building adjacency matrix.")

        # Aggregated interaction weight between user and product
        grouped = self.train_df.groupby(["user_idx", "product_idx"])["weight"].sum().reset_index()
        u_indices = grouped["user_idx"].values
        p_indices = grouped["product_idx"].values
        weights = grouped["weight"].values

        num_nodes = self.num_users + self.num_products

        # User-item bipartite interaction matrix R
        r_rows = u_indices
        r_cols = p_indices + self.num_users  # Shift item indices

        # Bidirectional graph edges
        rows = np.concatenate([r_rows, r_cols])
        cols = np.concatenate([r_cols, r_rows])
        data = np.concatenate([weights, weights])

        adj_mat = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)

        # Degree normalization: D^-0.5 * A * D^-0.5
        row_sum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
        d_inv_sqrt[row_sum <= 0] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()

        # Convert to PyTorch Sparse Tensor
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def sample_bpr_triplet(self, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples (user, pos_item, neg_item) triplets for BPR (Bayesian Personalized Ranking) loss.
        """
        active_users = [u for u, items in self.user_train_pos.items() if len(items) > 0]
        if not active_users:
            raise ValueError("No active training users found.")

        sampled_users = np.random.choice(active_users, size=batch_size, replace=True)
        pos_items = np.zeros(batch_size, dtype=np.int64)
        neg_items = np.zeros(batch_size, dtype=np.int64)

        all_prod_set = set(range(self.num_products))

        for i, u in enumerate(sampled_users):
            user_pos = self.user_train_pos[u]
            pos_items[i] = np.random.choice(user_pos)

            # Uniform negative item sampling not interacted with
            neg = np.random.randint(0, self.num_products)
            user_pos_set = set(user_pos)
            while neg in user_pos_set:
                neg = np.random.randint(0, self.num_products)
            neg_items[i] = neg

        return (
            torch.tensor(sampled_users, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
        )
