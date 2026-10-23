"""
Heterogeneous Graph Builder for Customer Behavior Networks.
Builds, maintains, and updates multi-relational graphs of Users, Products, Categories,
and behavioral interactions.
"""

from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import pandas as pd
from .schema import NodeType, EdgeType, EdgeWeightConfig, NODE_VISUAL_STYLES, EDGE_VISUAL_STYLES


class HeterogeneousGraphBuilder:
    """
    Constructs a multi-relational graph linking Users, Products, and Categories.
    Provides methods for sub-graph querying, live edge insertion, and co-occurrence extraction.
    """

    def __init__(self, weight_config: Optional[EdgeWeightConfig] = None):
        self.graph = nx.MultiDiGraph()
        self.undirected_graph = nx.Graph()
        self.weight_config = weight_config or EdgeWeightConfig()

    def build_from_dataframes(self, users_df: pd.DataFrame, products_df: pd.DataFrame, 
                              interactions_df: pd.DataFrame) -> nx.MultiDiGraph:
        """Populates the multi-relational graph from tabular dataframes."""
        self.graph.clear()
        self.undirected_graph.clear()

        # 1. Add Category Nodes
        categories = products_df["category"].dropna().unique()
        for cat in categories:
            cat_id = f"CAT_{cat.replace(' ', '_').upper()}"
            self.graph.add_node(
                cat_id,
                entity_type=NodeType.CATEGORY.value,
                label=cat,
                name=cat,
                title=f"Category: {cat}"
            )

        # 2. Add Product Nodes & Product -> Category Edges
        for _, row in products_df.iterrows():
            prod_id = row["product_id"]
            cat = row.get("category", "General")
            cat_id = f"CAT_{cat.replace(' ', '_').upper()}"
            
            self.graph.add_node(
                prod_id,
                entity_type=NodeType.PRODUCT.value,
                label=row.get("name", prod_id),
                name=row.get("name", prod_id),
                category=cat,
                price=float(row.get("price", 0.0)),
                brand=row.get("brand", "Unknown"),
                title=f"Product: {row.get('name', prod_id)} (₹{row.get('price', 0):.2f})"
            )

            # Product to Category edge
            w = self.weight_config.get(EdgeType.BELONGS_TO.value)
            self.graph.add_edge(
                prod_id, cat_id,
                key="belongs_to",
                relation=EdgeType.BELONGS_TO.value,
                weight=w,
                label="belongs_to"
            )
            self.undirected_graph.add_edge(prod_id, cat_id, weight=w, relation=EdgeType.BELONGS_TO.value)

        # 3. Add User Nodes
        for _, row in users_df.iterrows():
            user_id = row["user_id"]
            self.graph.add_node(
                user_id,
                entity_type=NodeType.USER.value,
                label=row.get("name", user_id),
                name=row.get("name", user_id),
                age_group=row.get("age_group", "N/A"),
                activity_level=row.get("activity_level", "medium"),
                interests=row.get("interests", ""),
                title=f"User: {row.get('name', user_id)} ({row.get('activity_level', 'medium')})"
            )

        # 4. Add Behavioral Interaction Edges
        action_map = {
            "view": EdgeType.VIEWED.value,
            "search_click": EdgeType.SEARCHED.value,
            "cart": EdgeType.CARTED.value,
            "purchase": EdgeType.PURCHASED.value,
            "rating": EdgeType.RATED.value
        }

        for _, row in interactions_df.iterrows():
            u_id = row["user_id"]
            p_id = row["product_id"]
            raw_action = str(row.get("action_type", "view")).lower()
            rel_type = action_map.get(raw_action, EdgeType.VIEWED.value)
            weight = float(row.get("weight", self.weight_config.get(rel_type)))
            timestamp = str(row.get("timestamp", ""))
            session_id = str(row.get("session_id", ""))

            if self.graph.has_node(u_id) and self.graph.has_node(p_id):
                self.graph.add_edge(
                    u_id, p_id,
                    relation=rel_type,
                    weight=weight,
                    timestamp=timestamp,
                    session_id=session_id,
                    label=rel_type.lower()
                )

                # Maintain weighted undirected graph for pathfinding and random walks
                if self.undirected_graph.has_edge(u_id, p_id):
                    self.undirected_graph[u_id][p_id]["weight"] += weight
                else:
                    self.undirected_graph.add_edge(u_id, p_id, weight=weight, relation="interacted")

        return self.graph

    def add_interaction(self, user_id: str, product_id: str, action_type: str, 
                        timestamp: str = None, session_id: str = "LIVE") -> None:
        """Dynamically add an interaction event at runtime."""
        action_map = {
            "view": EdgeType.VIEWED.value,
            "search_click": EdgeType.SEARCHED.value,
            "cart": EdgeType.CARTED.value,
            "purchase": EdgeType.PURCHASED.value,
            "rating": EdgeType.RATED.value
        }
        rel_type = action_map.get(action_type.lower(), EdgeType.VIEWED.value)
        weight = self.weight_config.get(rel_type)

        if not self.graph.has_node(user_id) or not self.graph.has_node(product_id):
            return

        self.graph.add_edge(
            user_id, product_id,
            relation=rel_type,
            weight=weight,
            timestamp=timestamp or "",
            session_id=session_id,
            label=rel_type.lower()
        )

        if self.undirected_graph.has_edge(user_id, product_id):
            self.undirected_graph[user_id][product_id]["weight"] += weight
        else:
            self.undirected_graph.add_edge(user_id, product_id, weight=weight, relation="interacted")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute structural network metrics."""
        node_counts = {NodeType.USER.value: 0, NodeType.PRODUCT.value: 0, NodeType.CATEGORY.value: 0}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type")
            if etype in node_counts:
                node_counts[etype] += 1

        edge_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "other")
            edge_counts[rel] = edge_counts.get(rel, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "connected_components": nx.number_connected_components(self.undirected_graph) if len(self.undirected_graph) > 0 else 0,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0.0
        }
