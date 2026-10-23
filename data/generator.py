"""
E-commerce Customer Behavior Data Generator.
Generates realistic multi-entity networks with users, products, categories,
and heterogeneous interaction streams (views, clicks, carts, purchases, ratings).
"""

import os
import random
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class EcommerceDataGenerator:
    """
    Generates synthetic e-commerce interaction datasets with realistic
    user behavioral clusters, category affinities, and multi-relational interactions.
    """

    CATEGORIES = {
        "Electronics": ["Smartphone", "Wireless Earbuds", "Gaming Laptop", "Smartwatch", "4K Monitor", "Bluetooth Speaker", "Tablet Pro"],
        "Fashion": ["Leather Jacket", "Running Shoes", "Denim Jeans", "Cotton Hoodie", "Designer Sunglasses", "Wristwatch", "Sneakers"],
        "Home & Kitchen": ["Espresso Maker", "Air Fryer", "Chef Knife Set", "Robot Vacuum", "Ergonomic Chair", "Blender", "Aromatherapy Diffuser"],
        "Sports & Fitness": ["Yoga Mat", "Adjustable Dumbbells", "Resistance Bands", "Hydration Flask", "Treadmill", "Cycling Helmet"],
        "Books & Media": ["Machine Learning Handbook", "Sci-Fi Best-seller", "Productivity Guide", "Cooking Masterclass", "Historical Novel"],
        "Beauty & Wellness": ["Hydrating Serum", "Sunscreen SPF 50", "Electric Toothbrush", "Hair Dryer", "Organic Face Wash"]
    }

    ACTION_WEIGHTS = {
        "view": 1.0,
        "search_click": 1.5,
        "cart": 3.0,
        "rating": 3.5,
        "purchase": 5.0
    }

    def __init__(self, num_users: int = 120, num_products: int = 60, seed: int = 42):
        self.num_users = num_users
        self.num_products = num_products
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_products(self) -> pd.DataFrame:
        """Generate structured product catalog with category and price metadata."""
        products = []
        prod_idx = 1
        all_categories = list(self.CATEGORIES.keys())

        # Distribute items evenly across categories
        items_per_cat = max(1, self.num_products // len(all_categories))

        for cat in all_categories:
            items = self.CATEGORIES[cat]
            for i in range(items_per_cat):
                item_name = items[i % len(items)]
                if i >= len(items):
                    item_name = f"{item_name} V{i // len(items) + 1}"
                
                price = round(random.uniform(15.0, 850.0), 2)
                brand = f"Brand-{random.choice(['Apex', 'Nexus', 'Vertex', 'Nova', 'Pulse', 'Aura'])}"
                products.append({
                    "product_id": f"P{prod_idx:04d}",
                    "name": item_name,
                    "category": cat,
                    "price": price,
                    "brand": brand,
                    "tags": f"{cat.lower()},{item_name.lower().replace(' ', '_')}"
                })
                prod_idx += 1
                if prod_idx > self.num_products:
                    break
            if prod_idx > self.num_products:
                break

        # Fill remaining if needed
        while len(products) < self.num_products:
            cat = random.choice(all_categories)
            item_name = random.choice(self.CATEGORIES[cat]) + f" #{prod_idx}"
            products.append({
                "product_id": f"P{prod_idx:04d}",
                "name": item_name,
                "category": cat,
                "price": round(random.uniform(20.0, 500.0), 2),
                "brand": "Brand-Nexus",
                "tags": f"{cat.lower()}"
            })
            prod_idx += 1

        return pd.DataFrame(products)

    def generate_users(self) -> pd.DataFrame:
        """Generate user profiles with personas, demographic clusters, and category affinities."""
        users = []
        all_categories = list(self.CATEGORIES.keys())
        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Casey", "Riley", "Avery", "Jamie", "Dakota", "Reese"]
        last_names = ["Chen", "Smith", "Kumar", "Patel", "Garcia", "Johnson", "Kim", "Miller", "Davis", "Wilson"]

        for u in range(1, self.num_users + 1):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            # Each user has 1 or 2 primary interest categories (forming community clusters)
            primary_interests = random.sample(all_categories, k=random.choice([1, 2]))
            age_group = random.choice(["18-24", "25-34", "35-49", "50+"])
            activity_level = random.choice(["low", "medium", "high", "vip"])
            users.append({
                "user_id": f"U{u:04d}",
                "name": name,
                "age_group": age_group,
                "activity_level": activity_level,
                "primary_interest": primary_interests[0],
                "interests": ",".join(primary_interests)
            })

        return pd.DataFrame(users)

    def generate_interactions(self, users_df: pd.DataFrame, products_df: pd.DataFrame, 
                              target_interactions: int = 2500) -> pd.DataFrame:
        """
        Generate session-based behavioral interactions.
        Users interact preferentially with items in their interest categories,
        creating realistic graph topologies and collaborative filtering patterns.
        """
        interactions = []
        prod_by_category = {cat: products_df[products_df["category"] == cat]["product_id"].tolist() 
                            for cat in self.CATEGORIES.keys()}
        all_products = products_df["product_id"].tolist()

        start_time = datetime.datetime.now() - datetime.timedelta(days=30)
        session_counter = 1000

        user_records = users_df.to_dict("records")
        for u in user_records:
            user_id = u["user_id"]
            interests = u["interests"].split(",")
            activity_mult = {"low": 8, "medium": 18, "high": 35, "vip": 55}[u["activity_level"]]
            num_sessions = max(2, int(np.random.poisson(activity_mult / 3)))

            for _ in range(num_sessions):
                session_counter += 1
                session_id = f"S{session_counter:06d}"
                session_time = start_time + datetime.timedelta(
                    days=random.uniform(0, 30),
                    hours=random.uniform(0, 24),
                    minutes=random.uniform(0, 60)
                )

                # Pick candidate products: 80% chance from preferred categories, 20% exploration
                if random.random() < 0.82 and interests:
                    cat = random.choice(interests)
                    candidates = prod_by_category.get(cat, all_products)
                else:
                    candidates = all_products

                if not candidates:
                    candidates = all_products

                # Session trajectory: View -> (Search/Click) -> (Cart) -> (Purchase) -> (Rating)
                session_items = random.sample(candidates, k=min(len(candidates), random.randint(1, 4)))
                for p_idx, prod_id in enumerate(session_items):
                    # Step 1: View
                    interactions.append({
                        "user_id": user_id,
                        "product_id": prod_id,
                        "session_id": session_id,
                        "action_type": "view",
                        "weight": self.ACTION_WEIGHTS["view"],
                        "rating": None,
                        "timestamp": (session_time + datetime.timedelta(minutes=p_idx * 3)).strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Optional search click
                    if random.random() < 0.45:
                        interactions.append({
                            "user_id": user_id,
                            "product_id": prod_id,
                            "session_id": session_id,
                            "action_type": "search_click",
                            "weight": self.ACTION_WEIGHTS["search_click"],
                            "rating": None,
                            "timestamp": (session_time + datetime.timedelta(minutes=p_idx * 3 + 1)).strftime("%Y-%m-%d %H:%M:%S")
                        })

                    # Conversion funnel: Cart
                    if random.random() < 0.35:
                        interactions.append({
                            "user_id": user_id,
                            "product_id": prod_id,
                            "session_id": session_id,
                            "action_type": "cart",
                            "weight": self.ACTION_WEIGHTS["cart"],
                            "rating": None,
                            "timestamp": (session_time + datetime.timedelta(minutes=p_idx * 3 + 2)).strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # Conversion funnel: Purchase
                        if random.random() < 0.55:
                            interactions.append({
                                "user_id": user_id,
                                "product_id": prod_id,
                                "session_id": session_id,
                                "action_type": "purchase",
                                "weight": self.ACTION_WEIGHTS["purchase"],
                                "rating": None,
                                "timestamp": (session_time + datetime.timedelta(minutes=p_idx * 3 + 4)).strftime("%Y-%m-%d %H:%M:%S")
                            })

                            # Conversion funnel: Rating
                            if random.random() < 0.40:
                                star_rating = random.choices([3.0, 4.0, 5.0, 2.0, 1.0], weights=[0.2, 0.45, 0.25, 0.07, 0.03])[0]
                                interactions.append({
                                    "user_id": user_id,
                                    "product_id": prod_id,
                                    "session_id": session_id,
                                    "action_type": "rating",
                                    "weight": self.ACTION_WEIGHTS["rating"] + (star_rating - 3.0),
                                    "rating": star_rating,
                                    "timestamp": (session_time + datetime.timedelta(minutes=p_idx * 3 + 15)).strftime("%Y-%m-%d %H:%M:%S")
                                })

        df = pd.DataFrame(interactions)
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        return df

    def generate_all(self, output_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate users, products, and interactions, optionally saving to CSV."""
        users_df = self.generate_users()
        products_df = self.generate_products()
        interactions_df = self.generate_interactions(users_df, products_df)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            users_df.to_csv(os.path.join(output_dir, "users.csv"), index=False)
            products_df.to_csv(os.path.join(output_dir, "products.csv"), index=False)
            interactions_df.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)
            print(f"Generated {len(users_df)} users, {len(products_df)} products, and {len(interactions_df)} interactions in '{output_dir}'.")

        return users_df, products_df, interactions_df


if __name__ == "__main__":
    gen = EcommerceDataGenerator(num_users=100, num_products=50)
    u_df, p_df, i_df = gen.generate_all(output_dir="data/raw")
    print(f"Sample interactions:\n{i_df.head(5)}")
