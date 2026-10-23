# Graph-Based Customer Behavior Modeling for Personalized Recommendation Systems

A complete end-to-end framework and interactive application for personalized recommendations using **Multi-Relational Heterogeneous Graphs**, **PyTorch LightGCN (Graph Convolutional Networks)**, **Personalized PageRank (Random Walk with Restart)**, and **Multi-Objective Explainable Re-ranking**.

---

## 🌟 Executive Summary

Traditional recommendation systems (Matrix Factorization, Collaborative Filtering) treat user-item interactions as a flat matrix, losing the rich multi-hop relationships between customer sessions, browse clicks, cart additions, ratings, and product categories.

This project implements a graph-theoretic approach:
1. **Heterogeneous Graph Construction**: Models Customers, Products, and Categories as nodes, connected by multi-relational edges (`VIEWED`, `SEARCHED`, `CARTED`, `PURCHASED`, `RATED`, `BELONGS_TO`).
2. **Graph Representation Learning (LightGCN)**: Applies recursive neighborhood aggregation without non-linear feature transformation, optimized with Bayesian Personalized Ranking (BPR) loss.
3. **Graph Diffusion (Personalized PageRank)**: Propagates behavioral signals across multi-hop heterogeneous paths to capture high-order contextual affinity in real time.
4. **Hybrid Re-ranking & Explainability**: Balances deep GNN structural representations with real-time session intent, category diversity constraints, and human-readable explanation paths.
5. **Interactive Web Dashboard**: Glassmorphic UI featuring a physics-driven network graph visualizer (Vis.js), live behavior simulation, and offline performance benchmark suite.

---

## 🏗️ System Architecture

```
[Customer Interaction Stream]
   │ (Views, Clicks, Carts, Purchases, Ratings)
   ▼
[Data Preprocessing & Indexing Engine]
   │ 80/20 Temporal Split & Negative Sampling
   ├──► [Bipartite Graph Matrix] ──────► [PyTorch LightGCN (GNN)] ─────┐
   │    Normalized Laplacian D^-0.5 A D^-0.5                           │
   │                                                                   ▼
   └──► [Heterogeneous Graph] ────────► [Personalized PageRank] ─► [Hybrid Ranker]
        Nodes: Users, Products, Cats        Random Walk with Restart   │ Multi-Objective
        Edges: Multi-relational weights                                │ Re-ranking
                                                                       ▼
                                                          [Top-K Recommendations]
                                                          + Graph Explanations
                                                          + Interactive Web UI / REST API
```

---

## 📂 Project Directory Structure

```
Graph-Based Customer Behavior Modeling for Personalized Recommendation Systems/
├── data/
│   ├── generator.py          # Synthetic e-commerce behavioral data generator
│   ├── dataset.py            # ID mappings, train/test splitting, PyTorch tensors
│   └── raw/                  # Generated CSV datasets (users, products, interactions)
├── graph/
│   ├── schema.py             # Node/Edge schema, behavioral weights, and styles
│   ├── builder.py            # Heterogeneous & multi-relational NetworkX builder
│   └── visualizer.py         # Ego-network & explanation path extractor for Vis.js
├── models/
│   ├── lightgcn.py           # PyTorch LightGCN implementation + BPR loss
│   ├── ppr_recommender.py    # Personalized PageRank & Random Walk with Restart
│   ├── baselines.py          # Popularity, Item-based CF, and Matrix Factorization (SVD)
│   └── hybrid_ranker.py      # Multi-objective ranker with explainability & diversity
├── evaluation/
│   ├── metrics.py            # Precision@K, Recall@K, NDCG@K, HitRate@K, MRR, Coverage, Diversity
│   └── benchmark.py          # Offline benchmarking suite & matplotlib visualizer
├── server/
│   ├── app.py                # FastAPI server with full REST endpoints
│   └── static/
│       ├── index.html        # Interactive web application interface
│       ├── css/style.css     # Modern responsive glassmorphism styles
│       ├── js/app.js         # Vis.js network visualizer & client controller
│       └── benchmark_results.png # Comparative metrics chart
├── tests/
│   └── test_pipeline.py      # Unit & integration test suite
├── run_server.py             # Server launcher script
├── run_experiments.py        # Command-line benchmark runner
└── requirements.txt          # Project dependencies
```

---

## 🔬 Mathematical Formulations

### 1. LightGCN Graph Convolution
LightGCN simplifies graph convolutions by removing non-linear activation functions and feature transformations:
$$E^{(k+1)} = \tilde{A} E^{(k)} = \left(D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right) E^{(k)}$$

The final representation combines representations across all $K$ layers:
$$E = \sum_{k=0}^{K} \alpha_k E^{(k)}, \quad \alpha_k = \frac{1}{K+1}$$

### 2. Bayesian Personalized Ranking (BPR) Loss
The network is trained to rank observed positive interactions higher than unobserved negative items:
$$\mathcal{L}_{BPR} = -\sum_{(u, i, j) \in \mathcal{D}} \ln \sigma\left(\hat{y}_{ui} - \hat{y}_{uj}\right) + \lambda \|\Theta\|^2$$
where $\hat{y}_{ui} = \mathbf{e}_u^T \mathbf{e}_i$ denotes the predicted preference of user $u$ for item $i$.

### 3. Personalized PageRank Diffusion
For capturing multi-hop contextual affinity across the heterogeneous graph:
$$\mathbf{p} = \alpha P^T \mathbf{p} + (1 - \alpha) \mathbf{s}$$
where $\mathbf{s}$ is the restart vector (seeded at the customer and recent session items), and $P$ is the transition probability matrix.

---

## 📊 Comparative Performance Benchmark

Results obtained on held-out temporal test interactions (Top-10 metrics):

| Model Architecture | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 | MRR@10 | Catalog Coverage | Diversity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid (LightGCN + PPR)** | **0.1562** | 0.4500 | 0.0600 | 0.1839 | **0.2581** | **1.00** | **0.8085** |
| **LightGCN (PyTorch GNN)** | 0.1518 | **0.4833** | **0.0617** | **0.1907** | 0.2479 | **1.00** | 0.7330 |
| **Personalized PageRank** | 0.1256 | 0.4000 | 0.0517 | 0.1643 | 0.1781 | 0.96 | 0.7478 |
| **Item-based CF** | 0.1339 | 0.4667 | 0.0583 | 0.1773 | 0.1993 | **1.00** | 0.7456 |
| **Matrix Factorization (SVD)**| 0.1209 | 0.4000 | 0.0483 | 0.1450 | 0.1804 | 0.98 | 0.8419 |
| **Popularity Baseline** | 0.0511 | 0.2667 | 0.0317 | 0.0848 | 0.0659 | 0.56 | 0.6722 |

> **Key Finding**: Graph-based models significantly outperform standard matrix factorization and popularity baselines. The Hybrid architecture combines LightGCN's collaborative embedding accuracy with PPR's session-responsiveness, delivering high NDCG and balanced category diversity.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark & Experiments
To generate data, train all models, and run the comparative benchmark:
```bash
python run_experiments.py --users 100 --products 50 --epochs 25
```

### 3. Launch the Interactive Web Application
Start the FastAPI server:
```bash
python run_server.py --port 8000
```
Open your browser at:
- **Interactive Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger REST API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Run Unit Tests
Verify model math, graph construction, and evaluation metrics:
```bash
python -m unittest tests.test_pipeline
```

---

## ☁️ Deploy to Render

This application is ready for 1-click deployment on [Render](https://render.com):

### Option 1: Blueprint Deployment (Fastest)
1. Go to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** ➔ **Blueprint**.
3. Connect your GitHub repository (`nscare62-maker/ecom`).
4. Render automatically reads `render.yaml` and provisions the Web Service with Python 3.11 and CPU-optimized PyTorch.
5. Click **Apply**.

### Option 2: Manual Web Service Setup
1. In Render Dashboard, click **New +** ➔ **Web Service**.
2. Connect `nscare62-maker/ecom`.
3. Set the following configuration:
   - **Environment**: `Python`
   - **Region**: `Oregon (US West)` or nearest
   - **Branch**: `main`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     python run_server.py
     ```
4. Click **Create Web Service**. Your live URL will be active in ~2 minutes (e.g. `https://ecom-graph-rec.onrender.com`).

---

## 🌐 REST API Endpoints

- `GET /api/overview` - Network summary statistics, node counts, and edge distributions.
- `GET /api/users` - List all customer profiles and interest categories.
- `GET /api/products` - Catalog of items with pricing and categories.
- `GET /api/user/{user_id}/history` - Interaction history and active session items.
- `GET /api/recommendations` - Top-K recommendations using `hybrid`, `lightgcn`, `ppr`, `itemcf`, or `mf`.
- `POST /api/interact` - Real-time simulation of customer actions (`view`, `cart`, `purchase`, `search_click`), immediately updating graph structure and recalculating recommendations.
- `GET /api/graph/user-ego` - Vis.js format of user ego-network.
- `GET /api/graph/explain` - Shortest graph path connecting user to recommended item.
- `GET /api/benchmark` - Offline comparative metrics table and chart path.

---

## 📜 License
This project is developed for educational and research purposes in Advanced AI & Recommender Systems.
