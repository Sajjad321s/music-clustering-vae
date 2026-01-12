"""
Visualization Module
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from sklearn.manifold import TSNE
import umap


def plot_mfcc_samples(X, y, genres, save_path='mfcc_samples.png'):
    """
    Visualize MFCC for one sample from each genre.
    
    Args:
        X: MFCC features
        y: Labels
        genres: List of genre names
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, genre in enumerate(genres):
        # Find first sample of this genre
        idx = np.where(y == i)[0][0]
        
        # Plot MFCC
        librosa.display.specshow(X[idx], x_axis='time', ax=axes[i], cmap='coolwarm')
        axes[i].set_title(f'{genre.capitalize()}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('MFCC Coefficients')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"MFCC samples saved to '{save_path}'")


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot VAE training history.
    
    Args:
        history: Training history object
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Total loss
    axes[0].plot(history.history['total_loss'], label='Total Loss', linewidth=2)
    if 'val_total_loss' in history.history:
        axes[0].plot(history.history['val_total_loss'], label='Val Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history.history['reconstruction_loss'], label='Reconstruction Loss', linewidth=2)
    if 'val_reconstruction_loss' in history.history:
        axes[1].plot(history.history['val_reconstruction_loss'], label='Val Reconstruction Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(history.history['kl_loss'], label='KL Loss', linewidth=2)
    if 'val_kl_loss' in history.history:
        axes[2].plot(history.history['val_kl_loss'], label='Val KL Loss', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Loss', fontsize=12)
    axes[2].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to '{save_path}'")


def plot_metrics_comparison(results_df, save_path='metrics_comparison.png'):
    """
    Visualize metrics comparison between VAE and PCA.
    
    Args:
        results_df: DataFrame with evaluation metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = results_df['Method'].values
    silhouette_scores = results_df['Silhouette Score'].values
    calinski_scores = results_df['Calinski-Harabasz Index'].values
    
    # Silhouette Score
    axes[0].bar(methods, silhouette_scores, color=['#2ecc71', '#3498db'], alpha=0.8)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(silhouette_scores):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Calinski-Harabasz Index
    axes[1].bar(methods, calinski_scores, color=['#2ecc71', '#3498db'], alpha=0.8)
    axes[1].set_ylabel('Calinski-Harabasz Index', fontsize=12)
    axes[1].set_title('Calinski-Harabasz Index Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(calinski_scores):
        axes[1].text(i, v + max(calinski_scores) * 0.02, f'{v:.2f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison saved to '{save_path}'")


def plot_tsne_visualization(vae_features, vae_clusters, pca_features, pca_clusters, 
                            y, genres, save_path='tsne_visualization.png'):
    """
    Visualize t-SNE projections for VAE and PCA methods.
    
    Args:
        vae_features: VAE latent features
        vae_clusters: VAE cluster labels
        pca_features: PCA features
        pca_clusters: PCA cluster labels
        y: True labels
        genres: List of genre names
        save_path: Path to save figure
    """
    # Apply t-SNE
    print("Applying t-SNE to VAE latent features...")
    tsne_vae = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne_vae.fit_transform(vae_features)
    
    print("Applying t-SNE to PCA features...")
    tsne_pca = TSNE(n_components=2, random_state=42, perplexity=30)
    pca_tsne = tsne_pca.fit_transform(pca_features)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # VAE - True genres
    for i, genre in enumerate(genres):
        mask = y == i
        axes[0, 0].scatter(z_tsne[mask, 0], z_tsne[mask, 1], 
                          c=[colors[i]], label=genre, alpha=0.6, s=30)
    axes[0, 0].set_title('VAE Latent Space (t-SNE) - True Genres', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('t-SNE Component 1')
    axes[0, 0].set_ylabel('t-SNE Component 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAE - Predicted clusters
    n_clusters = len(np.unique(vae_clusters))
    for i in range(n_clusters):
        mask = vae_clusters == i
        axes[0, 1].scatter(z_tsne[mask, 0], z_tsne[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
    axes[0, 1].set_title('VAE Latent Space (t-SNE) - K-Means Clusters', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA - True genres
    for i, genre in enumerate(genres):
        mask = y == i
        axes[1, 0].scatter(pca_tsne[mask, 0], pca_tsne[mask, 1], 
                          c=[colors[i]], label=genre, alpha=0.6, s=30)
    axes[1, 0].set_title('PCA Features (t-SNE) - True Genres', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # PCA - Predicted clusters
    for i in range(n_clusters):
        mask = pca_clusters == i
        axes[1, 1].scatter(pca_tsne[mask, 0], pca_tsne[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
    axes[1, 1].set_title('PCA Features (t-SNE) - K-Means Clusters', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('t-SNE Component 1')
    axes[1, 1].set_ylabel('t-SNE Component 2')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to '{save_path}'")


def plot_umap_visualization(vae_features, vae_clusters, pca_features, pca_clusters, 
                            y, genres, save_path='umap_visualization.png'):
    """
    Visualize UMAP projections for VAE and PCA methods.
    
    Args:
        vae_features: VAE latent features
        vae_clusters: VAE cluster labels
        pca_features: PCA features
        pca_clusters: PCA cluster labels
        y: True labels
        genres: List of genre names
        save_path: Path to save figure
    """
    # Apply UMAP
    print("Applying UMAP to VAE latent features...")
    umap_vae = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    z_umap = umap_vae.fit_transform(vae_features)
    
    print("Applying UMAP to PCA features...")
    umap_pca = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    pca_umap = umap_pca.fit_transform(pca_features)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # VAE - True genres
    for i, genre in enumerate(genres):
        mask = y == i
        axes[0, 0].scatter(z_umap[mask, 0], z_umap[mask, 1], 
                          c=[colors[i]], label=genre, alpha=0.6, s=30)
    axes[0, 0].set_title('VAE Latent Space (UMAP) - True Genres', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP Component 1')
    axes[0, 0].set_ylabel('UMAP Component 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # VAE - Predicted clusters
    n_clusters = len(np.unique(vae_clusters))
    for i in range(n_clusters):
        mask = vae_clusters == i
        axes[0, 1].scatter(z_umap[mask, 0], z_umap[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
    axes[0, 1].set_title('VAE Latent Space (UMAP) - K-Means Clusters', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('UMAP Component 1')
    axes[0, 1].set_ylabel('UMAP Component 2')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA - True genres
    for i, genre in enumerate(genres):
        mask = y == i
        axes[1, 0].scatter(pca_umap[mask, 0], pca_umap[mask, 1], 
                          c=[colors[i]], label=genre, alpha=0.6, s=30)
    axes[1, 0].set_title('PCA Features (UMAP) - True Genres', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP Component 1')
    axes[1, 0].set_ylabel('UMAP Component 2')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # PCA - Predicted clusters
    for i in range(n_clusters):
        mask = pca_clusters == i
        axes[1, 1].scatter(pca_umap[mask, 0], pca_umap[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
    axes[1, 1].set_title('PCA Features (UMAP) - K-Means Clusters', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('UMAP Component 1')
    axes[1, 1].set_ylabel('UMAP Component 2')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP visualization saved to '{save_path}'")


def plot_vae_reconstructions(vae, X_input, X_normalized, y, genres, n_samples=5, save_path='vae_reconstructions.png'):
    """
    Visualize VAE reconstructions.
    
    Args:
        vae: Trained VAE model
        X_input: VAE input features
        X_normalized: Normalized features
        y: True labels
        genres: List of genre names
        n_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    sample_indices = np.random.choice(len(X_input), n_samples, replace=False)
    reconstructions = vae.predict(X_input[sample_indices])
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    
    for i, idx in enumerate(sample_indices):
        # Original
        librosa.display.specshow(X_normalized[idx], x_axis='time', ax=axes[i, 0], cmap='coolwarm')
        axes[i, 0].set_title(f'Original - {genres[y[idx]].capitalize()}', fontweight='bold')
        axes[i, 0].set_ylabel('MFCC Coefficients')
        
        # Reconstructed
        librosa.display.specshow(reconstructions[i, :, :, 0], x_axis='time', ax=axes[i, 1], cmap='coolwarm')
        axes[i, 1].set_title(f'Reconstructed - {genres[y[idx]].capitalize()}', fontweight='bold')
        axes[i, 1].set_ylabel('MFCC Coefficients')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"VAE reconstructions saved to '{save_path}'")
