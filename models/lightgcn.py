"""
PyTorch implementation of LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
Reference: He et al., SIGIR 2020.
Features:
- Pure neighborhood aggregation without non-linear feature transformation
- Weighted multi-layer convolution over bipartite user-item graph
- BPR (Bayesian Personalized Ranking) loss with L2 regularization
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN Neural Network Architecture.
    Computes user and item embeddings through multiple graph convolution layers.
    """

    def __init__(self, num_users: int, num_items: int, latent_dim: int = 64, num_layers: int = 3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Initial 0-th layer embeddings (Gaussian initialization)
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs recursive graph convolution:
        E^(k+1) = D^-0.5 * A * D^-0.5 * E^(k)
        Final embedding is the uniform average of embeddings across all layers.
        """
        all_embeddings = [torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        for _ in range(self.num_layers):
            # Sparse matrix multiplication for message passing
            current_emb = torch.sparse.mm(norm_adj, all_embeddings[-1])
            all_embeddings.append(current_emb)

        # Average pooling across all layer representations
        stacked_embeddings = torch.stack(all_embeddings, dim=0)
        final_embeddings = torch.mean(stacked_embeddings, dim=0)

        user_final = final_embeddings[:self.num_users]
        item_final = final_embeddings[self.num_users:]

        return user_final, item_final

    def bpr_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor,
                 norm_adj: torch.Tensor, decay: float = 1e-4) -> torch.Tensor:
        """
        Bayesian Personalized Ranking (BPR) Loss:
        Maximizes margin between positive interaction and negative (unobserved) items.
        """
        u_final, i_final = self.forward(norm_adj)

        u_emb = u_final[users]
        pos_emb = i_final[pos_items]
        neg_emb = i_final[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        bpr = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization on initial embeddings
        u_init = self.user_embedding(users)
        pos_init = self.item_embedding(pos_items)
        neg_init = self.item_embedding(neg_items)
        reg_loss = (1 / 2) * (u_init.norm(2).pow(2) + pos_init.norm(2).pow(2) + neg_init.norm(2).pow(2)) / float(len(users))

        return bpr + decay * reg_loss


class LightGCNRecommender:
    """
    High-level API for training, inference, and top-K recommendation using LightGCN.
    """

    def __init__(self, latent_dim: int = 64, num_layers: int = 3, lr: float = 0.005,
                 decay: float = 1e-4, device: Optional[str] = None):
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.lr = lr
        self.decay = decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[LightGCN] = None
        self.dataset = None
        self.norm_adj: Optional[torch.Tensor] = None
        self.final_user_emb: Optional[np.ndarray] = None
        self.final_item_emb: Optional[np.ndarray] = None

    def fit(self, dataset, epochs: int = 40, batch_size: int = 512, verbose: bool = True):
        """Train LightGCN on interaction dataset."""
        self.dataset = dataset
        self.model = LightGCN(
            num_users=dataset.num_users,
            num_items=dataset.num_products,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.norm_adj = dataset.get_norm_adj_mat().to(self.device)

        steps_per_epoch = max(1, len(dataset.train_df) // batch_size)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for _ in range(steps_per_epoch):
                users, pos_items, neg_items = dataset.sample_bpr_triplet(batch_size)
                users = users.to(self.device)
                pos_items = pos_items.to(self.device)
                neg_items = neg_items.to(self.device)

                optimizer.zero_grad()
                loss = self.model.bpr_loss(users, pos_items, neg_items, self.norm_adj, self.decay)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / steps_per_epoch
            if verbose and (epoch % 10 == 0 or epoch == epochs):
                print(f"[LightGCN] Epoch {epoch:03d}/{epochs:03d} - BPR Loss: {avg_loss:.4f}")

        # Cache final embeddings for rapid vector similarity inference
        self.model.eval()
        with torch.no_grad():
            u_emb, i_emb = self.model(self.norm_adj)
            self.final_user_emb = u_emb.cpu().numpy()
            self.final_item_emb = i_emb.cpu().numpy()

    def get_all_scores_for_user(self, user_idx: int) -> np.ndarray:
        """Compute recommendation score vector across all products for a given user index."""
        if self.final_user_emb is None or self.final_item_emb is None:
            raise RuntimeError("Model must be trained with .fit() before inference.")
        u_vector = self.final_user_emb[user_idx]
        return np.dot(self.final_item_emb, u_vector)

    def recommend(self, user_id: str, k: int = 10, filter_interacted: bool = True) -> List[Dict[str, any]]:
        """
        Generate personalized top-K recommendations for a specific user ID.
        """
        if user_id not in self.dataset.user2idx:
            # Fallback for unknown user: return top products by norm
            return []

        u_idx = self.dataset.user2idx[user_id]
        scores = self.get_all_scores_for_user(u_idx).copy()

        # Mask products already seen in training
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
                "model": "LightGCN"
            })

        return results
