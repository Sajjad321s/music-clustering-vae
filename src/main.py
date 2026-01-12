"""
Main script to run the complete VAE music clustering pipeline.
This script can be run directly with: python main.py
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from src.dataset import DataLoader
from src.vae import create_vae
from src.clustering import ClusteringPipeline
from src.evaluation import MetricsEvaluator
from src.visualization import (
    plot_mfcc_samples,
    plot_training_history,
    plot_metrics_comparison,
    plot_tsne_visualization,
    plot_umap_visualization,
    plot_vae_reconstructions
)


# Configuration
DATA_PATH = 'genres'
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
N_MFCC = 20
SAMPLE_RATE = 22050
DURATION = 30
LATENT_DIM = 32
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
N_CLUSTERS = len(GENRES)


def main():
    """Run the complete VAE music clustering pipeline."""
    
    print("="*80)
    print("VAE MUSIC CLUSTERING - EASY TASK")
    print("="*80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/latent_visualization', exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_loader = DataLoader(DATA_PATH, GENRES, N_MFCC, SAMPLE_RATE, DURATION)
    X_input, y = data_loader.load_and_preprocess()
    X_normalized = data_loader.get_features()
    
    # Step 2: Visualize sample MFCCs
    print("\n2. Generating MFCC samples visualization...")
    plot_mfcc_samples(X_normalized, y, GENRES, 'results/mfcc_samples.png')
    
    # Step 3: Create and train VAE
    print("\n3. Creating VAE model...")
    vae, encoder, decoder = create_vae(X_input.shape[1:], LATENT_DIM, learning_rate=1e-3)
    
    print("\n4. Training VAE...")
    print(f"   Configuration:")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Latent dimension: {LATENT_DIM}")
    
    # Callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    early_stopping = EarlyStopping(
        monitor='total_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='total_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    history = vae.fit(
        X_input,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Step 5: Plot training history
    print("\n5. Plotting training history...")
    plot_training_history(history, 'results/training_history.png')
    
    # Step 6: Perform clustering
    print("\n6. Performing clustering...")
    clustering = ClusteringPipeline(N_CLUSTERS, LATENT_DIM)
    
    print("   - Clustering with VAE features...")
    vae_clusters = clustering.cluster_vae_features(encoder, X_input, BATCH_SIZE)
    
    print("   - Clustering with PCA baseline...")
    pca_clusters = clustering.cluster_pca_baseline(X_normalized)
    
    # Step 7: Evaluate clustering
    print("\n7. Evaluating clustering performance...")
    evaluator = MetricsEvaluator()
    results = evaluator.evaluate(
        clustering.get_vae_features(), vae_clusters,
        clustering.get_pca_features(), pca_clusters,
        y
    )
    
    evaluator.print_results()
    evaluator.save_results('results/clustering_metrics.csv')
    
    # Step 8: Plot metrics comparison
    print("\n8. Plotting metrics comparison...")
    plot_metrics_comparison(results, 'results/metrics_comparison.png')
    
    # Step 9: Generate t-SNE visualization
    print("\n9. Generating t-SNE visualization...")
    plot_tsne_visualization(
        clustering.get_vae_features(), vae_clusters,
        clustering.get_pca_features(), pca_clusters,
        y, GENRES,
        'results/latent_visualization/tsne_visualization.png'
    )
    
    # Step 10: Generate UMAP visualization
    print("\n10. Generating UMAP visualization...")
    plot_umap_visualization(
        clustering.get_vae_features(), vae_clusters,
        clustering.get_pca_features(), pca_clusters,
        y, GENRES,
        'results/latent_visualization/umap_visualization.png'
    )
    
    # Step 11: Generate reconstruction examples
    print("\n11. Generating VAE reconstruction examples...")
    plot_vae_reconstructions(
        vae, X_input, X_normalized, y, GENRES,
        n_samples=5,
        save_path='results/vae_reconstructions.png'
    )
    
    # Step 12: Save models and features
    print("\n12. Saving models and features...")
    vae.save('results/vae_model.keras')
    encoder.save('results/encoder_model.keras')
    decoder.save('results/decoder_model.keras')
    
    np.save('results/vae_latent_features.npy', clustering.get_vae_features())
    np.save('results/pca_features.npy', clustering.get_pca_features())
    np.save('results/vae_clusters.npy', vae_clusters)
    np.save('results/pca_clusters.npy', pca_clusters)
    np.save('results/true_labels.npy', y)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nAll results saved in 'results/' directory:")
    print("  - Visualizations in results/ and results/latent_visualization/")
    print("  - Metrics in results/clustering_metrics.csv")
    print("  - Models in results/*.keras")
    print("  - Features in results/*.npy")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
