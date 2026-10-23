"""
Hybrid Graph Recommendation & Re-Ranking Engine with Explainability.
Fuses Graph Neural Network (LightGCN) embeddings with Heterogeneous Personalized PageRank,
applies session-context weighting, category diversity re-ranking, and explanation paths.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import networkx as nx
from graph.schema import NodeType, EdgeType


class HybridGraphRanker:
    """
    Multi-objective re-ranker combining:
    - Deep structural affinity from LightGCN
    - Multi-hop path connectivity from Heterogeneous PPR
    - Session context / real-time behavioral recency
    - Category diversification and explainable rationale generation
    """

    def __init__(self, lightgcn_recommender, ppr_recommender, builder, dataset,
                 weight_gnn: float = 0.60, weight_ppr: float = 0.40):
        self.lightgcn = lightgcn_recommender
        self.ppr = ppr_recommender
        self.builder = builder
        self.dataset = dataset
        self.weight_gnn = weight_gnn
        self.weight_ppr = weight_ppr

    def recommend(self, user_id: str, k: int = 10, filter_interacted: bool = True,
                  recent_session_items: Optional[List[str]] = None,
                  diversity_penalty: float = 0.15) -> List[Dict[str, Any]]:
        """
        Produce hybrid-ranked recommendations with explainable paths.
        """
        if user_id not in self.dataset.user2idx:
            # Fallback to PPR or Popularity
            return self.ppr.recommend(user_id, k=k, filter_interacted=filter_interacted)

        u_idx = self.dataset.user2idx[user_id]

        # 1. Fetch raw scores from LightGCN
        gnn_scores = self.lightgcn.get_all_scores_for_user(u_idx).copy()

        # Min-max normalize GNN scores to [0, 1]
        gnn_min, gnn_max = np.min(gnn_scores), np.max(gnn_scores)
        if gnn_max > gnn_min:
            gnn_norm = (gnn_scores - gnn_min) / (gnn_max - gnn_min)
        else:
            gnn_norm = np.zeros_like(gnn_scores)

        # 2. Fetch scores from Heterogeneous PPR
        ppr_candidates = self.ppr.recommend(
            user_id,
            k=self.dataset.num_products,
            filter_interacted=False,
            context_product_ids=recent_session_items
        )
        ppr_dict = {item["product_id"]: item["score"] for item in ppr_candidates}

        # Normalize PPR scores
        if ppr_dict:
            max_ppr = max(ppr_dict.values())
            min_ppr = min(ppr_dict.values())
            range_ppr = max_ppr - min_ppr if max_ppr > min_ppr else 1.0
            ppr_norm = {pid: (score - min_ppr) / range_ppr for pid, score in ppr_dict.items()}
        else:
            ppr_norm = {}

        # 3. Combine scores with session context boost
        user_seen_indices = set(self.dataset.user_train_pos.get(u_idx, []))
        user_profile = self.dataset.users_df[self.dataset.users_df["user_id"] == user_id].iloc[0]
        user_interests = [i.strip() for i in str(user_profile.get("interests", "")).split(",") if i.strip()]

        scored_items = []
        for p_idx in range(self.dataset.num_products):
            if filter_interacted and p_idx in user_seen_indices:
                continue

            prod_id = self.dataset.idx2prod[p_idx]
            prod_row = self.dataset.products_df[self.dataset.products_df["product_id"] == prod_id].iloc[0]
            cat = prod_row["category"]

            s_gnn = float(gnn_norm[p_idx])
            s_ppr = float(ppr_norm.get(prod_id, 0.0))

            # Base hybrid score
            hybrid_score = (self.weight_gnn * s_gnn) + (self.weight_ppr * s_ppr)

            # Affinity bonus if item matches user primary interests
            if cat in user_interests:
                hybrid_score *= 1.20

            # Session context boost if product matches category of recently viewed items
            if recent_session_items:
                for r_pid in recent_session_items:
                    r_item = self.dataset.products_df[self.dataset.products_df["product_id"] == r_pid]
                    if not r_item.empty and r_item.iloc[0]["category"] == cat:
                        hybrid_score *= 1.25
                        break

            scored_items.append({
                "product_idx": p_idx,
                "product_id": prod_id,
                "name": prod_row["name"],
                "category": cat,
                "price": float(prod_row["price"]),
                "gnn_score": round(s_gnn, 4),
                "ppr_score": round(s_ppr, 4),
                "hybrid_score": float(hybrid_score)
            })

        # Sort by initial hybrid score
        scored_items.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # 4. Diversity re-ranking (Maximal Marginal Relevance / Category Coverage)
        final_recs = []
        category_counts: Dict[str, int] = {}

        for item in scored_items:
            cat = item["category"]
            current_cat_count = category_counts.get(cat, 0)

            # Apply marginal penalty for overrepresented categories in the top-K list
            adjusted_score = item["hybrid_score"] / (1.0 + diversity_penalty * current_cat_count)
            item["adjusted_score"] = round(adjusted_score, 4)

            # Insert keeping sorted order
            final_recs.append(item)
            category_counts[cat] = current_cat_count + 1

        final_recs.sort(key=lambda x: x["adjusted_score"], reverse=True)
        top_k = final_recs[:k]

        # 5. Attach Graph Explanations
        for rank, item in enumerate(top_k, start=1):
            item["rank"] = rank
            item["model"] = "Hybrid-GNN+PPR"
            item["explanation"] = self._generate_explanation(user_id, item["product_id"], item["category"], user_interests)

        return top_k

    def _generate_explanation(self, user_id: str, product_id: str, category: str, user_interests: List[str]) -> Dict[str, Any]:
        """Synthesize intuitive behavioral and graph-path explanations."""
        # Find connection path in graph
        path_desc = f"Direct Graph Neighborhood"
        try:
            if self.builder.undirected_graph.has_node(user_id) and self.builder.undirected_graph.has_node(product_id):
                path = nx.shortest_path(self.builder.undirected_graph, source=user_id, target=product_id)
                if len(path) == 3:
                    # User -> Intermediate -> Product
                    mid_node = path[1]
                    mid_data = self.builder.graph.nodes.get(mid_node, {})
                    mid_label = mid_data.get("name", mid_node)
                    path_desc = f"Connected via your interaction with '{mid_label}'"
                elif len(path) == 4:
                    # User -> Prod1 -> Category -> RecProd
                    path_desc = f"Connected via category affinity '{category}'"
        except Exception:
            pass

        reasons = []
        if category in user_interests:
            reasons.append(f"Matches your high-affinity '{category}' interest cluster")
        reasons.append(f"High graph collaborative embedding alignment")
        if path_desc:
            reasons.append(path_desc)

        return {
            "summary": " & ".join(reasons[:2]),
            "path_description": path_desc,
            "category": category
        }
