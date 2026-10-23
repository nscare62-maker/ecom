"""
Comprehensive Unit & Integration Tests for Graph-Based Recommendation System.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from data.generator import EcommerceDataGenerator
from data.dataset import RecommendationDataset
from graph.builder import HeterogeneousGraphBuilder
from graph.visualizer import GraphVisualizer
from graph.schema import NodeType, EdgeType
from models.lightgcn import LightGCN, LightGCNRecommender
from models.ppr_recommender import PersonalizedPageRankRecommender
from models.baselines import PopularityRecommender, ItemCFRecommender, MatrixFactorizationRecommender
from models.hybrid_ranker import HybridGraphRanker
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, mrr_at_k, catalog_coverage


class TestGraphRecommendationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate small dataset for fast testing
        cls.gen = EcommerceDataGenerator(num_users=30, num_products=20, seed=123)
        cls.users_df, cls.products_df, cls.interactions_df = cls.gen.generate_all()
        cls.dataset = RecommendationDataset(cls.users_df, cls.products_df, cls.interactions_df)
        cls.train_df, cls.test_df = cls.dataset.train_test_split(test_ratio=0.20, strategy="random")

    def test_data_generation(self):
        self.assertEqual(len(self.users_df), 30)
        self.assertEqual(len(self.products_df), 20)
        self.assertGreater(len(self.interactions_df), 50)
        self.assertTrue("user_id" in self.interactions_df.columns)
        self.assertTrue("product_id" in self.interactions_df.columns)

    def test_dataset_indexing_and_adjacency(self):
        self.assertEqual(self.dataset.num_users, 30)
        self.assertEqual(self.dataset.num_products, 20)
        adj = self.dataset.get_norm_adj_mat()
        self.assertIsInstance(adj, torch.Tensor)
        self.assertEqual(adj.shape[0], 30 + 20)
        self.assertEqual(adj.shape[1], 30 + 20)

    def test_graph_builder(self):
        builder = HeterogeneousGraphBuilder()
        builder.build_from_dataframes(self.users_df, self.products_df, self.train_df)
        stats = builder.get_summary_statistics()

        self.assertGreater(stats["total_nodes"], 40)
        self.assertGreater(stats["total_edges"], 30)
        self.assertIn(NodeType.USER.value, stats["node_counts"])
        self.assertIn(NodeType.PRODUCT.value, stats["node_counts"])
        self.assertIn(NodeType.CATEGORY.value, stats["node_counts"])

        # Test live interaction dynamic insertion
        user_id = self.dataset.user_ids[0]
        prod_id = self.dataset.product_ids[0]
        initial_edges = builder.graph.number_of_edges()
        builder.add_interaction(user_id, prod_id, "cart")
        self.assertEqual(builder.graph.number_of_edges(), initial_edges + 1)

    def test_graph_visualizer(self):
        builder = HeterogeneousGraphBuilder()
        builder.build_from_dataframes(self.users_df, self.products_df, self.train_df)
        vis = GraphVisualizer(builder)

        user_id = self.dataset.user_ids[0]
        ego = vis.get_user_ego_network(user_id, max_items=5)
        self.assertIn("nodes", ego)
        self.assertIn("edges", ego)
        self.assertGreater(len(ego["nodes"]), 0)

    def test_lightgcn_training_and_recommendation(self):
        recommender = LightGCNRecommender(latent_dim=16, num_layers=2, lr=0.01)
        recommender.fit(self.dataset, epochs=5, batch_size=64, verbose=False)

        user_id = self.dataset.user_ids[0]
        recs = recommender.recommend(user_id, k=5, filter_interacted=False)
        self.assertEqual(len(recs), 5)
        self.assertEqual(recs[0]["rank"], 1)
        self.assertIn("score", recs[0])

    def test_personalized_pagerank(self):
        builder = HeterogeneousGraphBuilder()
        builder.build_from_dataframes(self.users_df, self.products_df, self.train_df)
        ppr = PersonalizedPageRankRecommender(builder, alpha=0.85)

        user_id = self.dataset.user_ids[0]
        recs = ppr.recommend(user_id, k=4)
        self.assertEqual(len(recs), 4)
        self.assertGreater(recs[0]["score"], 0)

    def test_hybrid_ranker(self):
        builder = HeterogeneousGraphBuilder()
        builder.build_from_dataframes(self.users_df, self.products_df, self.train_df)

        lightgcn = LightGCNRecommender(latent_dim=16, num_layers=2)
        lightgcn.fit(self.dataset, epochs=3, batch_size=64, verbose=False)

        ppr = PersonalizedPageRankRecommender(builder)
        hybrid = HybridGraphRanker(lightgcn, ppr, builder, self.dataset)

        user_id = self.dataset.user_ids[0]
        recs = hybrid.recommend(user_id, k=5)
        self.assertEqual(len(recs), 5)
        self.assertIn("explanation", recs[0])
        self.assertIn("summary", recs[0]["explanation"])

    def test_evaluation_metrics(self):
        recommended = ["P001", "P002", "P003", "P004", "P005"]
        ground_truth = {"P002", "P005", "P010"}

        p5 = precision_at_k(recommended, ground_truth, k=5)
        r5 = recall_at_k(recommended, ground_truth, k=5)
        hr5 = hit_rate_at_k(recommended, ground_truth, k=5)
        mrr5 = mrr_at_k(recommended, ground_truth, k=5)
        ndcg5 = ndcg_at_k(recommended, ground_truth, k=5)

        self.assertAlmostEqual(p5, 2.0 / 5.0)
        self.assertAlmostEqual(r5, 2.0 / 3.0)
        self.assertEqual(hr5, 1.0)
        self.assertAlmostEqual(mrr5, 0.5)  # First hit at rank 2 -> 1/2 = 0.5
        self.assertGreater(ndcg5, 0.0)


if __name__ == "__main__":
    unittest.main()
