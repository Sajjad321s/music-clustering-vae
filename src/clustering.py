"""
Clustering Algorithms Module
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def extract_vae_latent_features(vae_encoder, X_input, batch_size=32):
    """
    Extract latent features from VAE encoder.
    
    Args:
        vae_encoder: Trained VAE encoder
        X_input: Input features for VAE
        batch_size: Batch size for prediction
    
    Returns:
        z_mean: Latent features (mean)
        z_log_var: Latent features (log variance)
        z: Sampled latent features
    """
    z_mean, z_log_var, z = vae_encoder.predict(X_input, batch_size=batch_size)
    return z_mean, z_log_var, z


class ClusteringPipeline:
    """Clustering pipeline for VAE and baseline methods."""
    
    def __init__(self, n_clusters, latent_dim=32, random_state=42):
        """
        Initialize clustering pipeline.
        
        Args:
            n_clusters: Number of clusters
            latent_dim: Dimension of latent space (for PCA)
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.random_state = random_state
        
        self.vae_clusters = None
        self.pca_clusters = None
        self.z_mean = None
        self.pca_features = None
        
    def cluster_vae_features(self, vae_encoder, X_input, batch_size=32):
        """
        Cluster using VAE latent features.
        
        Args:
            vae_encoder: Trained VAE encoder
            X_input: Input features for VAE
            batch_size: Batch size for prediction
        
        Returns:
            vae_clusters: Cluster labels from VAE
        """
        # Extract latent features
        self.z_mean, _, _ = extract_vae_latent_features(vae_encoder, X_input, batch_size)
        
        # K-Means clustering
        self.vae_clusters, self.kmeans_vae = kmeans_clustering(
            self.z_mean, 
            self.n_clusters, 
            self.random_state
        )
        
        print(f"VAE K-Means clustering completed!")
        print(f"Cluster distribution: {np.bincount(self.vae_clusters)}")
        
        return self.vae_clusters
    
    def cluster_pca_baseline(self, X_normalized):
        """
        Baseline clustering using PCA + K-Means.
        
        Args:
            X_normalized: Normalized features
        
        Returns:
            pca_clusters: Cluster labels from PCA
        """
        # Flatten features
        X_flat = X_normalized.reshape(len(X_normalized), -1)
        
        # PCA + K-Means
        self.pca_clusters, self.pca_features, self.pca, self.kmeans_pca = pca_kmeans_baseline(
            X_flat, 
            self.latent_dim, 
            self.n_clusters, 
            self.random_state
        )
        
        print(f"PCA K-Means clustering completed!")
        print(f"Cluster distribution: {np.bincount(self.pca_clusters)}")
        
        return self.pca_clusters
    
    def get_vae_features(self):
        """Get VAE latent features."""
        return self.z_mean
    
    def get_pca_features(self):
        """Get PCA features."""
        return self.pca_features
    
    def get_vae_clusters(self):
        """Get VAE cluster labels."""
        return self.vae_clusters
    
    def get_pca_clusters(self):
        """Get PCA cluster labels."""
        return self.pca_clusters
