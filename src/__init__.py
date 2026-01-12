"""
VAE Music Clustering - Source Package
"""

from .vae import create_vae, build_encoder, build_decoder, VAE, Sampling
from .dataset import DataLoader, extract_mfcc_features, load_gtzan_dataset
from .clustering import ClusteringPipeline, kmeans_clustering, pca_kmeans_baseline
from .evaluation import MetricsEvaluator, evaluate_clustering, compare_methods
from .visualization import (
    plot_mfcc_samples,
    plot_training_history,
    plot_metrics_comparison,
    plot_tsne_visualization,
    plot_umap_visualization,
    plot_vae_reconstructions
)

__version__ = "1.0.0"
__author__ = "Moin Mostakim"
