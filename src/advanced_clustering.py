"""
Advanced Clustering Module - Medium Task
Multiple clustering algorithms: K-Means, Agglomerative, DBSCAN
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA


def kmeans_clustering(features, n_clusters, random_state=42, n_init=10):
    """
    Perform K-Means clustering.
    
    Args:
        features: Input features for clustering
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
        n_init: Number of initializations
    
    Returns:
        clusters: Cluster labels
        kmeans: Fitted KMeans object
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    clusters = kmeans.fit_predict(features)
    
    return clusters, kmeans


def agglomerative_clustering(features, n_clusters, linkage='ward'):
    """
    Perform Agglomerative (Hierarchical) Clustering.
    
    Args:
        features: Input features for clustering
        n_clusters: Number of clusters
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
    
    Returns:
        clusters: Cluster labels
        agg: Fitted AgglomerativeClustering object
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = agg.fit_predict(features)
    
    return clusters, agg


def dbscan_clustering(features, eps=3.0, min_samples=5):
    """
    Perform DBSCAN clustering (density-based).
    
    Args:
        features: Input features for clustering
        eps: Maximum distance between samples in a neighborhood
        min_samples: Minimum samples in a neighborhood for a core point
    
    Returns:
        clusters: Cluster labels (-1 indicates noise)
        dbscan: Fitted DBSCAN object
        n_clusters: Number of clusters found (excluding noise)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    
    # Count clusters (excluding noise labeled as -1)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"  DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    return clusters, dbscan, n_clusters


def pca_kmeans_baseline(features, n_components, n_clusters, random_state=42):
    """
    Baseline: PCA + K-Means clustering.
    
    Args:
        features: Input features (flattened)
        n_components: Number of PCA components
        n_clusters: Number of clusters
        random_state: Random state for reproducibility
    
    Returns:
        clusters: Cluster labels
        pca_features: PCA-transformed features
        pca: Fitted PCA object
        kmeans: Fitted KMeans object
    """
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(features)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # K-Means clustering
    clusters, kmeans = kmeans_clustering(pca_features, n_clusters, random_state)
    
    return clusters, pca_features, pca, kmeans


class AdvancedClusteringPipeline:
    """
    Advanced clustering pipeline with multiple algorithms.
    """
    
    def __init__(self, n_clusters, random_state=42):
        """
        Initialize clustering pipeline.
        
        Args:
            n_clusters: Number of clusters for applicable algorithms
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Store results
        self.results = {}
    
    def run_all_clustering(self, combined_features, audio_only_features, pca_baseline_features):
        """
        Run all clustering algorithms on different feature sets.
        
        Args:
            combined_features: Hybrid (audio + lyrics) features
            audio_only_features: Audio-only features
            pca_baseline_features: PCA baseline features
        
        Returns:
            results: Dictionary of clustering results
        """
        print("Running all clustering algorithms...\\n")
        
        # 1. K-Means on Combined Features
        print("1. K-Means on Hybrid Features...")
        clusters, model = kmeans_clustering(combined_features, self.n_clusters, self.random_state)
        self.results['hybrid_kmeans'] = {
            'clusters': clusters,
            'features': combined_features,
            'model': model,
            'name': 'Conv VAE Hybrid + K-Means'
        }
        
        # 2. K-Means on Audio-Only Features
        print("2. K-Means on Audio-Only Features...")
        clusters, model = kmeans_clustering(audio_only_features, self.n_clusters, self.random_state)
        self.results['audio_kmeans'] = {
            'clusters': clusters,
            'features': audio_only_features,
            'model': model,
            'name': 'Conv VAE Audio + K-Means'
        }
        
        # 3. Agglomerative on Combined Features
        print("3. Agglomerative Clustering on Hybrid Features...")
        clusters, model = agglomerative_clustering(combined_features, self.n_clusters)
        self.results['hybrid_agg'] = {
            'clusters': clusters,
            'features': combined_features,
            'model': model,
            'name': 'Conv VAE Hybrid + Agglomerative'
        }
        
        # 4. DBSCAN on Combined Features
        print("4. DBSCAN on Hybrid Features...")
        clusters, model, n_found = dbscan_clustering(combined_features, eps=3.0, min_samples=5)
        self.results['hybrid_dbscan'] = {
            'clusters': clusters,
            'features': combined_features,
            'model': model,
            'name': 'Conv VAE Hybrid + DBSCAN',
            'n_clusters_found': n_found
        }
        
        # 5. PCA + K-Means Baseline
        print("5. PCA + K-Means Baseline...")
        clusters = None
        if pca_baseline_features is not None:
            clusters, _ = kmeans_clustering(pca_baseline_features, self.n_clusters, self.random_state)
        self.results['pca_kmeans'] = {
            'clusters': clusters,
            'features': pca_baseline_features,
            'model': None,
            'name': 'PCA + K-Means (Baseline)'
        }
        
        print("\\n✅ All clustering methods completed!")
        
        return self.results
    
    def get_results(self):
        """Get clustering results."""
        return self.results
    
    def get_clusters(self, method_key):
        """
        Get cluster labels for a specific method.
        
        Args:
            method_key: Key for the clustering method
        
        Returns:
            clusters: Cluster labels
        """
        if method_key in self.results:
            return self.results[method_key]['clusters']
        else:
            return None
    
    def get_features(self, method_key):
        """
        Get features used for a specific method.
        
        Args:
            method_key: Key for the clustering method
        
        Returns:
            features: Feature vectors
        """
        if method_key in self.results:
            return self.results[method_key]['features']
        else:
            return None
