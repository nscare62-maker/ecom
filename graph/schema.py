"""
Graph Schema Definitions for Customer Behavior Modeling.
Defines Node Types, Edge Types, Edge Weights, and Visual Styling Attributes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class NodeType(str, Enum):
    USER = "USER"
    PRODUCT = "PRODUCT"
    CATEGORY = "CATEGORY"


class EdgeType(str, Enum):
    VIEWED = "VIEWED"
    SEARCHED = "SEARCHED"
    CARTED = "CARTED"
    PURCHASED = "PURCHASED"
    RATED = "RATED"
    BELONGS_TO = "BELONGS_TO"


@dataclass
class EdgeWeightConfig:
    """Configurable weights for behavioral edge relations."""
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                EdgeType.VIEWED.value: 1.0,
                EdgeType.SEARCHED.value: 1.5,
                EdgeType.CARTED.value: 3.0,
                EdgeType.PURCHASED.value: 5.0,
                EdgeType.RATED.value: 3.5,
                EdgeType.BELONGS_TO.value: 2.0,
            }

    def get(self, edge_type: str, default: float = 1.0) -> float:
        return self.weights.get(edge_type, default)


# Visual styles for frontend graph rendering
NODE_VISUAL_STYLES: Dict[str, Dict[str, Any]] = {
    NodeType.USER.value: {
        "color": {"background": "#3B82F6", "border": "#1D4ED8", "highlight": {"background": "#60A5FA", "border": "#1E40AF"}},
        "shape": "dot",
        "size": 22,
        "font": {"color": "#F8FAFC", "size": 13, "face": "Inter, sans-serif"},
    },
    NodeType.PRODUCT.value: {
        "color": {"background": "#10B981", "border": "#047857", "highlight": {"background": "#34D399", "border": "#065F46"}},
        "shape": "dot",
        "size": 18,
        "font": {"color": "#F8FAFC", "size": 12, "face": "Inter, sans-serif"},
    },
    NodeType.CATEGORY.value: {
        "color": {"background": "#F59E0B", "border": "#B45309", "highlight": {"background": "#FBBF24", "border": "#92400E"}},
        "shape": "hexagon",
        "size": 28,
        "font": {"color": "#1E293B", "size": 14, "face": "Inter, sans-serif", "bold": True},
    }
}

EDGE_VISUAL_STYLES: Dict[str, Dict[str, Any]] = {
    EdgeType.VIEWED.value: {"color": "#94A3B8", "width": 1.2, "dashes": True},
    EdgeType.SEARCHED.value: {"color": "#38BDF8", "width": 1.5, "dashes": False},
    EdgeType.CARTED.value: {"color": "#F59E0B", "width": 2.2, "dashes": False},
    EdgeType.PURCHASED.value: {"color": "#10B981", "width": 3.2, "dashes": False},
    EdgeType.RATED.value: {"color": "#8B5CF6", "width": 2.0, "dashes": False},
    EdgeType.BELONGS_TO.value: {"color": "#64748B", "width": 1.5, "dashes": [4, 4]},
}
