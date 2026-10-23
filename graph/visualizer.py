"""
Graph Visualizer & Subgraph Extractor.
Formats NetworkX graph partitions and paths into Vis.js/D3.js-ready JSON payloads.
"""

from typing import Dict, List, Set, Any, Optional
import networkx as nx
from .schema import NodeType, EdgeType, NODE_VISUAL_STYLES, EDGE_VISUAL_STYLES


class GraphVisualizer:
    """Extracts Ego-networks, recommendation explanation paths, and global visual structures."""

    def __init__(self, builder):
        self.builder = builder
        self.graph = builder.graph
        self.undirected = builder.undirected_graph

    def format_node_for_frontend(self, node_id: str) -> Dict[str, Any]:
        """Convert a node and its metadata to frontend Vis.js configuration."""
        data = self.graph.nodes.get(node_id, {})
        etype = data.get("entity_type", NodeType.PRODUCT.value)
        style = NODE_VISUAL_STYLES.get(etype, NODE_VISUAL_STYLES[NodeType.PRODUCT.value])

        label = data.get("label", node_id)
        if len(label) > 16 and etype != NodeType.CATEGORY.value:
            label = label[:14] + "..."

        title = f"<b>{node_id}</b><br/>Type: {etype}<br/>"
        if etype == NodeType.USER.value:
            title += f"Name: {data.get('name')}<br/>Interests: {data.get('interests')}<br/>Activity: {data.get('activity_level')}"
        elif etype == NodeType.PRODUCT.value:
            title += f"Name: {data.get('name')}<br/>Category: {data.get('category')}<br/>Price: ${data.get('price', 0):.2f}"
        elif etype == NodeType.CATEGORY.value:
            title += f"Category: {data.get('name')}"

        return {
            "id": node_id,
            "label": label,
            "title": title,
            "group": etype,
            "shape": style["shape"],
            "size": style["size"],
            "color": style["color"],
            "font": style["font"],
            "raw_data": data
        }

    def format_edge_for_frontend(self, u: str, v: str, key: Any, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an edge and its attributes to frontend Vis.js configuration."""
        rel = data.get("relation", EdgeType.VIEWED.value)
        style = EDGE_VISUAL_STYLES.get(rel, EDGE_VISUAL_STYLES[EdgeType.VIEWED.value])

        edge_label = rel.lower().replace("_", " ")
        return {
            "from": u,
            "to": v,
            "id": f"{u}->{v}:{key}",
            "label": edge_label,
            "title": f"Relation: {rel}<br/>Weight: {data.get('weight', 1.0)}",
            "color": {"color": style["color"], "highlight": style["color"]},
            "width": style["width"],
            "dashes": style.get("dashes", False),
            "arrows": "to",
            "smooth": {"type": "curvedCW", "roundness": 0.15}
        }

    def get_user_ego_network(self, user_id: str, max_items: int = 15, include_category: bool = True) -> Dict[str, Any]:
        """
        Extracts the ego network for a specific user:
        User -> Interacted Products -> Product Categories + top peer users.
        """
        if not self.graph.has_node(user_id):
            return {"nodes": [], "edges": []}

        nodes_set: Set[str] = {user_id}
        edges_list: List[Dict[str, Any]] = []

        # Find products directly interacted with by this user
        out_edges = list(self.graph.out_edges(user_id, data=True, keys=True))
        sorted_out_edges = sorted(out_edges, key=lambda e: e[3].get("weight", 1.0), reverse=True)[:max_items]

        interacted_products = set()
        for u, v, k, data in sorted_out_edges:
            nodes_set.add(v)
            interacted_products.add(v)
            edges_list.append(self.format_edge_for_frontend(u, v, k, data))

        # Add category nodes for these products
        if include_category:
            for p_id in interacted_products:
                for _, target, k, data in self.graph.out_edges(p_id, data=True, keys=True):
                    if data.get("relation") == EdgeType.BELONGS_TO.value:
                        nodes_set.add(target)
                        edges_list.append(self.format_edge_for_frontend(p_id, target, k, data))

        # Optionally add 1 or 2 peer users who also interacted with these items
        peer_users = set()
        for p_id in list(interacted_products)[:5]:
            for in_u, _, k, data in self.graph.in_edges(p_id, data=True, keys=True):
                if in_u != user_id and len(peer_users) < 3:
                    peer_users.add(in_u)
                    nodes_set.add(in_u)
                    edges_list.append(self.format_edge_for_frontend(in_u, p_id, k, data))

        nodes_list = [self.format_node_for_frontend(nid) for nid in nodes_set]
        return {"nodes": nodes_list, "edges": edges_list}

    def get_explanation_subgraph(self, user_id: str, recommended_item_id: str) -> Dict[str, Any]:
        """
        Finds the shortest/most explanatory path connecting User to Recommended Product
        e.g. User -(purchased)-> ProdA -(belongs_to)-> Category <-(belongs_to)- RecommendedProd
        or User -> ProdA <-(rated)- User2 -(viewed)-> RecommendedProd.
        """
        if not (self.undirected.has_node(user_id) and self.undirected.has_node(recommended_item_id)):
            return {"nodes": [], "edges": []}

        try:
            path = nx.shortest_path(self.undirected, source=user_id, target=recommended_item_id)
        except nx.NetworkXNoPath:
            path = [user_id, recommended_item_id]

        nodes_set = set(path)
        edges_list = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # check direction in directed graph
            if self.graph.has_edge(u, v):
                for k, data in self.graph[u][v].items():
                    edges_list.append(self.format_edge_for_frontend(u, v, k, data))
                    break
            elif self.graph.has_edge(v, u):
                for k, data in self.graph[v][u].items():
                    edges_list.append(self.format_edge_for_frontend(v, u, k, data))
                    break
            else:
                edges_list.append({
                    "from": u, "to": v, "id": f"{u}-{v}",
                    "label": "connected", "color": {"color": "#6366F1"},
                    "width": 2, "arrows": "to"
                })

        nodes_list = [self.format_node_for_frontend(nid) for nid in nodes_set]
        return {"nodes": nodes_list, "edges": edges_list, "path": path}

    def get_sample_graph(self, max_users: int = 20, max_products: int = 35) -> Dict[str, Any]:
        """Extracts a crisp, balanced sample graph for macro network overview."""
        users = [n for n, d in self.graph.nodes(data=True) if d.get("entity_type") == NodeType.USER.value][:max_users]
        products = [n for n, d in self.graph.nodes(data=True) if d.get("entity_type") == NodeType.PRODUCT.value][:max_products]
        categories = [n for n, d in self.graph.nodes(data=True) if d.get("entity_type") == NodeType.CATEGORY.value]

        sampled_nodes = set(users + products + categories)
        subg = self.graph.subgraph(sampled_nodes)

        nodes_list = [self.format_node_for_frontend(nid) for nid in subg.nodes()]
        edges_list = []
        for u, v, k, data in subg.edges(data=True, keys=True):
            edges_list.append(self.format_edge_for_frontend(u, v, k, data))

        return {"nodes": nodes_list, "edges": edges_list}
