"""
Personalized PageRank (PPR) & Random Walk with Restart (RWR) Recommender.
Operates on the multi-relational heterogeneous customer behavior graph.
Captures both direct interactions and indirect higher-order paths (user-product-category-product).
"""

from typing import Dict, List, Optional, Any
import networkx as nx
import numpy as np
import pandas as pd
from graph.schema import NodeType


class PersonalizedPageRankRecommender:
    """
    Computes graph diffusion / random walk stationary distribution starting from a target user
    or session context on the multi-relational graph.
    """

    def __init__(self, builder, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.builder = builder
        self.graph = builder.graph
        self.undirected_graph = builder.undirected_graph
        self.alpha = alpha  # Restart probability is (1 - alpha)
        self.max_iter = max_iter
        self.tol = tol

    def recommend(self, user_id: str, k: int = 10, filter_interacted: bool = True,
                  context_product_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Runs Personalized PageRank seeded at the user node and any active session context items.
        """
        if not self.undirected_graph.has_node(user_id):
            return []

        # Personalization distribution
        personalization = {user_id: 1.0}

        # If recent context items are provided, boost them in restart vector for session-aware RWR
        if context_product_ids:
            weight_per_item = 0.5
            for pid in context_product_ids:
                if self.undirected_graph.has_node(pid):
                    personalization[pid] = weight_per_item

        # Normalize personalization vector
        total_p = sum(personalization.values())
        personalization = {node: val / total_p for node, val in personalization.items()}

        try:
            pagerank_scores = nx.pagerank(
                self.undirected_graph,
                alpha=self.alpha,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol,
                weight="weight"
            )
        except Exception:
            # Fallback in case of convergence edge-case
            pagerank_scores = nx.pagerank(self.undirected_graph, alpha=self.alpha, max_iter=50)

        # Interacted products by this user
        interacted_products = set()
        if filter_interacted and self.graph.has_node(user_id):
            for _, v, _ in self.graph.out_edges(user_id, data=True):
                interacted_products.add(v)

        # Filter strictly for product nodes
        candidate_scores = []
        for node_id, score in pagerank_scores.items():
            node_data = self.graph.nodes.get(node_id, {})
            if node_data.get("entity_type") == NodeType.PRODUCT.value:
                if filter_interacted and node_id in interacted_products:
                    continue
                candidate_scores.append((node_id, score, node_data))

        # Sort descending by diffusion score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = candidate_scores[:k]

        results = []
        for rank, (prod_id, score, data) in enumerate(top_k, start=1):
            results.append({
                "rank": rank,
                "product_id": prod_id,
                "name": data.get("name", prod_id),
                "category": data.get("category", "N/A"),
                "price": float(data.get("price", 0.0)),
                "score": float(score),
                "model": "PersonalizedPageRank"
            })

        return results

    def get_contextual_affinity(self, user_id: str, product_id: str) -> float:
        """Calculate random walk transition affinity between a user and product."""
        if not (self.undirected_graph.has_node(user_id) and self.undirected_graph.has_node(product_id)):
            return 0.0

        personalization = {user_id: 1.0}
        pr = nx.pagerank(self.undirected_graph, alpha=self.alpha, personalization=personalization, max_iter=40)
        return float(pr.get(product_id, 0.0))
