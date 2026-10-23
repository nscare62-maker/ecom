"""
Data package for Graph-Based Customer Behavior Modeling.
"""
from .generator import EcommerceDataGenerator
from .dataset import RecommendationDataset

__all__ = ["EcommerceDataGenerator", "RecommendationDataset"]
