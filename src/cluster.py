"""
Clustering utilities for Amazon product reviews.
Groups products into meta-categories using TF-IDF and K-Means.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

OPTIMAL_K = 5
MAX_FEATURES = 500
MAX_LENGTH = 128
PCA_COMPONENTS_CLUSTER = 10
PCA_COMPONENTS_VIZ = 2
RANDOM_STATE = 42

CLUSTER_NAMES = {
    0: "Streaming Devices",
    1: "Fire Tablets",
    2: "E-Readers & Accessories",
    3: "Alexa Devices",
    4: "E-Readers Premium"
}


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def aggregate_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews at product level for clustering.

    Args:
        df: Full reviews DataFrame.

    Returns:
        Product-level DataFrame with aggregated features.
    """
    product_df = (
        df.groupby("product_name")
        .agg(
            all_reviews=("review_text", " ".join),
            num_reviews=("review_text", "count"),
            avg_rating=("rating", "mean"),
            pct_positive=("predicted_sentiment_finetuned",
                          lambda x: (x == "positive").mean()),
            primary_category=("primary_category", "first")
        )
        .reset_index()
    )

    # Add product name twice for extra weight in TF-IDF
    product_df["combined_text"] = (
        product_df["product_name"] + " " +
        product_df["product_name"] + " " +
        product_df["all_reviews"]
    )

    print(f"✅ Product-level dataset: {product_df.shape[0]} products")
    return product_df


def build_tfidf_matrix(product_df: pd.DataFrame):
    """Build TF-IDF matrix from combined product text.

    Args:
        product_df: Product-level DataFrame with combined_text column.

    Returns:
        Tuple of (tfidf_matrix, vectorizer).
    """
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = tfidf.fit_transform(product_df["combined_text"])
    print(f"✅ TF-IDF matrix: {tfidf_matrix.shape}")
    return tfidf_matrix, tfidf


def reduce_dimensions(tfidf_matrix, n_components: int = PCA_COMPONENTS_CLUSTER):
    """Reduce TF-IDF matrix dimensions using PCA.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix.
        n_components: Number of PCA components.

    Returns:
        Reduced matrix as numpy array.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    variance = pca.explained_variance_ratio_.sum()
    print(f"✅ PCA ({n_components} components): {variance:.2%} variance explained")
    return coords, pca


def find_optimal_k(coords: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """Evaluate K-Means for different values of K.

    Args:
        coords: Reduced feature matrix.
        k_range: Range of K values to test.

    Returns:
        Dict with inertias and silhouette scores per K.
    """
    results = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(coords)
        sil = silhouette_score(coords, labels)
        results["k"].append(k)
        results["inertia"].append(kmeans.inertia_)
        results["silhouette"].append(sil)
        print(f"  K={k} | Inertia: {kmeans.inertia_:.2f} | Silhouette: {sil:.4f}")

    return results


def apply_kmeans(coords: np.ndarray, k: int = OPTIMAL_K) -> tuple:
    """Apply K-Means clustering with optimal K.

    Args:
        coords: Reduced feature matrix.
        k: Number of clusters.

    Returns:
        Array of cluster labels.
    """
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(coords)
    print(f"✅ K-Means applied with K={k}")
    return labels, kmeans


def assign_meta_categories(
    product_df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict = None
) -> pd.DataFrame:
    """Assign human-readable meta-category names to clusters.

    Args:
        product_df: Product-level DataFrame.
        labels: Cluster labels from K-Means.
        cluster_names: Mapping from cluster ID to category name.

    Returns:
        Product DataFrame with meta_category column.
    """
    if cluster_names is None:        # ← aquí, después del docstring
        cluster_names = CLUSTER_NAMES
    product_df = product_df.copy()
    product_df["cluster"] = labels
    product_df["meta_category"] = product_df["cluster"].map(cluster_names)
    print("✅ Meta-categories assigned")
    return product_df


def merge_categories(df: pd.DataFrame, product_df: pd.DataFrame) -> pd.DataFrame:
    """Merge meta-categories back into the full reviews DataFrame.

    Args:
        df: Full reviews DataFrame.
        product_df: Product-level DataFrame with meta_category column.

    Returns:
        Full reviews DataFrame with meta_category column added.
    """
    category_map = product_df.set_index("product_name")["meta_category"].to_dict()
    df = df.copy()
    df["meta_category"] = df["product_name"].map(category_map)
    print("✅ Meta-categories merged into reviews dataset")
    print(df["meta_category"].value_counts().to_string())
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    """Run the full clustering pipeline."""
    # Load data
    df = pd.read_csv("data/processed/reviews_with_predictions_finetuned.csv")
    print(f"✅ Data loaded: {df.shape}")

    # Aggregate reviews per product
    product_df = aggregate_reviews(df)

    # TF-IDF vectorization
    tfidf_matrix, _ = build_tfidf_matrix(product_df)

    # Dimensionality reduction
    coords, _ = reduce_dimensions(tfidf_matrix, n_components=PCA_COMPONENTS_CLUSTER)

    # Find optimal K
    print("\nEvaluating K values:")
    find_optimal_k(coords)

    # Apply K-Means
    labels, _ = apply_kmeans(coords, k=OPTIMAL_K)

    # Assign meta-categories
    product_df = assign_meta_categories(product_df, labels)

    # Merge into full dataset
    df = merge_categories(df, product_df)

    # Save results
    df.to_csv("data/processed/reviews_clustered.csv", index=False)
    product_df.to_csv("data/processed/products_clustered.csv", index=False)

    print("\n✅ Clustering pipeline complete")
    print("   Reviews saved: data/processed/reviews_clustered.csv")
    print("   Products saved: data/processed/products_clustered.csv")


if __name__ == "__main__":
    main()
