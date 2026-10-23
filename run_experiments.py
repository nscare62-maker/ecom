"""
Command-line Offline Experimentation & Benchmark Runner.
Executes data generation, graph construction, training across all models,
and produces comparative performance tables.
"""

import os
import argparse
import pandas as pd
from data.generator import EcommerceDataGenerator
from data.dataset import RecommendationDataset
from graph.builder import HeterogeneousGraphBuilder
from models.lightgcn import LightGCNRecommender
from models.ppr_recommender import PersonalizedPageRankRecommender
from models.baselines import PopularityRecommender, ItemCFRecommender, MatrixFactorizationRecommender
from models.hybrid_ranker import HybridGraphRanker
from evaluation.benchmark import ModelBenchmark


def run_pipeline(num_users: int = 120, num_products: int = 60, epochs: int = 25):
    print("=" * 70)
    print(" GRAPH-BASED CUSTOMER BEHAVIOR MODELING - BENCHMARK PIPELINE")
    print("=" * 70)

    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n[1/5] Generating E-Commerce Data ({num_users} users, {num_products} products)...")
    gen = EcommerceDataGenerator(num_users=num_users, num_products=num_products, seed=42)
    users_df, products_df, interactions_df = gen.generate_all(output_dir=data_dir)

    print("\n[2/5] Preparing Dataset & Temporal Train/Test Split (80/20)...")
    dataset = RecommendationDataset(users_df, products_df, interactions_df)
    train_df, test_df = dataset.train_test_split(test_ratio=0.20, strategy="temporal")
    print(f"  Training Interactions: {len(train_df)}")
    print(f"  Testing Interactions:  {len(test_df)}")

    print("\n[3/5] Constructing Heterogeneous Multi-Relational Graph...")
    builder = HeterogeneousGraphBuilder()
    builder.build_from_dataframes(users_df, products_df, train_df)
    stats = builder.get_summary_statistics()
    print(f"  Total Nodes: {stats['total_nodes']} ({stats['node_counts']})")
    print(f"  Total Edges: {stats['total_edges']} ({stats['edge_counts']})")

    print("\n[4/5] Training Recommender Architectures...")
    print("  -> Training Popularity baseline...")
    pop_model = PopularityRecommender()
    pop_model.fit(dataset)

    print("  -> Training Item-based Collaborative Filtering...")
    itemcf_model = ItemCFRecommender()
    itemcf_model.fit(dataset)

    print("  -> Training Matrix Factorization (Truncated SVD)...")
    mf_model = MatrixFactorizationRecommender(n_components=24)
    mf_model.fit(dataset)

    print("  -> Initializing Personalized PageRank (PPR) Engine...")
    ppr_model = PersonalizedPageRankRecommender(builder, alpha=0.85)

    print(f"  -> Training PyTorch LightGCN Neural Network ({epochs} epochs)...")
    lightgcn_model = LightGCNRecommender(latent_dim=48, num_layers=3, lr=0.008)
    lightgcn_model.fit(dataset, epochs=epochs, batch_size=256, verbose=True)

    print("  -> Initializing Hybrid GNN + PPR Ranker...")
    hybrid_model = HybridGraphRanker(lightgcn_model, ppr_model, builder, dataset)

    print("\n[5/5] Running Offline Benchmark on Held-Out Test Set...")
    models = {
        "Hybrid (GNN+PPR)": hybrid_model,
        "LightGCN (GNN)": lightgcn_model,
        "Personalized PageRank": ppr_model,
        "ItemCF": itemcf_model,
        "Matrix Factorization": mf_model,
        "Popularity Baseline": pop_model
    }

    benchmark = ModelBenchmark(dataset, models, k_values=[5, 10])
    results_df = benchmark.run(max_test_users=60)

    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS SUMMARY (Top-10 Metrics)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    output_chart = "benchmark_results.png"
    benchmark.plot_comparison(output_chart)
    print(f"\nComparative chart saved to '{output_chart}'")

    # Sample recommendation preview
    sample_user = dataset.user_ids[0]
    print(f"\n[Preview] Top-5 Hybrid Recommendations for Customer '{sample_user}':")
    recs = hybrid_model.recommend(sample_user, k=5)
    for r in recs:
        exp = r.get("explanation", {}).get("summary", "N/A")
        print(f"  #{r['rank']} {r['name']} (${r['price']:.2f}) [{r['category']}]")
        print(f"     Score: {r['adjusted_score']:.4f} | Rationale: {exp}")

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment Pipeline")
    parser.add_argument("--users", type=int, default=120, help="Number of synthetic users")
    parser.add_argument("--products", type=int, default=60, help="Number of products")
    parser.add_argument("--epochs", type=int, default=25, help="LightGCN training epochs")
    args = parser.parse_args()

    run_pipeline(num_users=args.users, num_products=args.products, epochs=args.epochs)
