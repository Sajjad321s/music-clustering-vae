"""
Evaluation Metrics Module
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def compute_silhouette_score(features, clusters):
    """
    Compute Silhouette Score.
    
    Measures how similar an object is to its own cluster compared to other clusters.
    Range: [-1, 1], Higher is better.
    
    Args:
        features: Feature vectors
        clusters: Cluster labels
    
    Returns:
        score: Silhouette score
    """
    score = silhouette_score(features, clusters)
    return score


def compute_calinski_harabasz_score(features, clusters):
    """
    Compute Calinski-Harabasz Index.
    
    Ratio of between-cluster variance to within-cluster variance.
    Higher is better.
    
    Args:
        features: Feature vectors
        clusters: Cluster labels
    
    Returns:
        score: Calinski-Harabasz score
    """
    score = calinski_harabasz_score(features, clusters)
    return score


def compute_davies_bouldin_score(features, clusters):
    """
    Compute Davies-Bouldin Index.
    
    Average similarity between each cluster and its most similar cluster.
    Range: [0, inf], Lower is better.
    
    Args:
        features: Feature vectors
        clusters: Cluster labels
    
    Returns:
        score: Davies-Bouldin score
    """
    score = davies_bouldin_score(features, clusters)
    return score


def compute_adjusted_rand_score(true_labels, pred_clusters):
    """
    Compute Adjusted Rand Index (ARI).
    
    Measures similarity between predicted clusters and ground truth labels.
    Range: [-1, 1], Higher is better. 1 = perfect match, 0 = random labeling.
    
    Args:
        true_labels: Ground truth labels
        pred_clusters: Predicted cluster labels
    
    Returns:
        score: Adjusted Rand Index
    """
    score = adjusted_rand_score(true_labels, pred_clusters)
    return score


def compute_normalized_mutual_info(true_labels, pred_clusters):
    """
    Compute Normalized Mutual Information (NMI).
    
    Measures mutual information between predicted clusters and true labels.
    Range: [0, 1], Higher is better.
    
    Args:
        true_labels: Ground truth labels
        pred_clusters: Predicted cluster labels
    
    Returns:
        score: Normalized Mutual Information
    """
    score = normalized_mutual_info_score(true_labels, pred_clusters)
    return score


def compute_cluster_purity(true_labels, pred_clusters):
    """
    Compute Cluster Purity.
    
    Fraction of the dominant class in each cluster.
    Range: [0, 1], Higher is better.
    
    Args:
        true_labels: Ground truth labels
        pred_clusters: Predicted cluster labels
    
    Returns:
        purity: Cluster purity score
    """
    n = len(true_labels)
    unique_clusters = np.unique(pred_clusters)
    
    purity_sum = 0
    for cluster in unique_clusters:
        cluster_mask = pred_clusters == cluster
        cluster_labels = true_labels[cluster_mask]
        
        if len(cluster_labels) > 0:
            # Count of most common label in this cluster
            max_count = np.max(np.bincount(cluster_labels))
            purity_sum += max_count
    
    purity = purity_sum / n
    return purity


def evaluate_clustering(features, clusters, true_labels=None, method_name="Method"):
    """
    Evaluate clustering quality using multiple metrics.
    
    Args:
        features: Feature vectors
        clusters: Cluster labels
        true_labels: Ground truth labels (optional)
        method_name: Name of the clustering method
    
    Returns:
        metrics: Dictionary of metric scores
    """
    metrics = {}
    
    # Unsupervised metrics (don't need true labels)
    metrics['Silhouette Score'] = compute_silhouette_score(features, clusters)
    metrics['Calinski-Harabasz Index'] = compute_calinski_harabasz_score(features, clusters)
    metrics['Davies-Bouldin Index'] = compute_davies_bouldin_score(features, clusters)
    
    # Supervised metrics (need true labels)
    if true_labels is not None:
        metrics['Adjusted Rand Index'] = compute_adjusted_rand_score(true_labels, clusters)
        metrics['Normalized Mutual Information'] = compute_normalized_mutual_info(true_labels, clusters)
        metrics['Cluster Purity'] = compute_cluster_purity(true_labels, clusters)
    
    return metrics


def compare_methods(vae_features, vae_clusters, pca_features, pca_clusters, true_labels=None):
    """
    Compare VAE and PCA clustering methods.
    
    Args:
        vae_features: VAE latent features
        vae_clusters: VAE cluster labels
        pca_features: PCA features
        pca_clusters: PCA cluster labels
        true_labels: Ground truth labels (optional)
    
    Returns:
        results_df: DataFrame with comparison results
    """
    # Evaluate VAE method
    vae_metrics = evaluate_clustering(vae_features, vae_clusters, true_labels, "VAE + K-Means")
    
    # Evaluate PCA method
    pca_metrics = evaluate_clustering(pca_features, pca_clusters, true_labels, "PCA + K-Means")
    
    # Create results dataframe
    results = {
        'Method': ['VAE + K-Means', 'PCA + K-Means'],
        'Silhouette Score': [vae_metrics['Silhouette Score'], pca_metrics['Silhouette Score']],
        'Calinski-Harabasz Index': [vae_metrics['Calinski-Harabasz Index'], pca_metrics['Calinski-Harabasz Index']],
        'Davies-Bouldin Index': [vae_metrics['Davies-Bouldin Index'], pca_metrics['Davies-Bouldin Index']]
    }
    
    if true_labels is not None:
        results['Adjusted Rand Index'] = [vae_metrics['Adjusted Rand Index'], pca_metrics['Adjusted Rand Index']]
        results['Normalized Mutual Information'] = [vae_metrics['Normalized Mutual Information'], 
                                                     pca_metrics['Normalized Mutual Information']]
        results['Cluster Purity'] = [vae_metrics['Cluster Purity'], pca_metrics['Cluster Purity']]
    
    results_df = pd.DataFrame(results)
    
    return results_df


class MetricsEvaluator:
    """Metrics evaluator for clustering results."""
    
    def __init__(self):
        """Initialize metrics evaluator."""
        self.results = None
    
    def evaluate(self, vae_features, vae_clusters, pca_features, pca_clusters, true_labels=None):
        """
        Evaluate and compare clustering methods.
        
        Args:
            vae_features: VAE latent features
            vae_clusters: VAE cluster labels
            pca_features: PCA features
            pca_clusters: PCA cluster labels
            true_labels: Ground truth labels (optional)
        
        Returns:
            results_df: DataFrame with evaluation results
        """
        self.results = compare_methods(
            vae_features, vae_clusters,
            pca_features, pca_clusters,
            true_labels
        )
        
        return self.results
    
    def print_results(self):
        """Print evaluation results."""
        if self.results is not None:
            print("\n" + "="*80)
            print("CLUSTERING EVALUATION METRICS")
            print("="*80)
            print(self.results.to_string(index=False))
            print("="*80)
        else:
            print("No results to display. Run evaluate() first.")
    
    def save_results(self, filepath):
        """
        Save results to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.results is not None:
            self.results.to_csv(filepath, index=False)
            print(f"\nResults saved to '{filepath}'")
        else:
            print("No results to save. Run evaluate() first.")
    
    def get_results(self):
        """Get evaluation results."""
        return self.results
