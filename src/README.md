# VAE for Music Clustering - Medium Task

## Enhanced Implementation with Multi-Modal Features

**Course**: Neural Networks  
**Prepared By**: Moin Mostakim  
**Task Level**: Medium

## Overview

This implementation extends the Easy Task with:

### ✨ New Features

1. **Convolutional VAE Architecture**
   - 3 Conv2D layers (32 → 64 → 128 filters)
   - Max pooling for dimensionality reduction
   - Transpose convolutions for reconstruction
   - Better feature extraction from spectrograms

2. **Hybrid Multi-Modal Features**
   - Audio features: MFCC via Convolutional VAE
   - Lyrics features: Sentence embeddings (384-dim)
   - Combined latent space: 64 dimensions

3. **Multiple Clustering Algorithms**
   - K-Means (baseline)
   - Agglomerative Clustering (hierarchical)
   - DBSCAN (density-based)

4. **Enhanced Evaluation Metrics**
   - Silhouette Score
   - Calinski-Harabasz Index
   - **Davies-Bouldin Index** (NEW)
   - **Adjusted Rand Index** (NEW)
   - **Normalized Mutual Information** (NEW)

## Repository Structure

```
medium_task/
├── src_medium/
│   ├── conv_vae.py                 # Convolutional VAE models
│   ├── advanced_clustering.py      # Multiple clustering algorithms
│   └── enhanced_evaluation.py      # Comprehensive metrics
├── medium_task_vae_clustering.ipynb  # Main Colab notebook
└── README.md                       # This file
```

## Quick Start

### Running in Google Colab

1. **Upload** `medium_task_vae_clustering.ipynb` to Colab
2. **Enable GPU** (Runtime → Change runtime type → T4 GPU)
3. **Upload** your `kaggle.json` when prompted
4. **Run all cells**

### Expected Runtime

```
Setup & Installation:      2 min
Dataset Download:          2-3 min  (Kaggle)
Lyrics Dataset:            1-2 min
Audio Feature Extraction:  8-12 min
Lyrics Embedding:          2-3 min
Conv VAE Training:         15-20 min (with GPU)
Clustering (all methods):  3-5 min
Visualization:             5-7 min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                     ~40-55 min
```

## Model Architecture

### Convolutional Audio Encoder
```
Input: (20, 1293, 1)
↓
Conv2D(32) + BN + MaxPool → (10, 647, 32)
Conv2D(64) + BN + MaxPool → (5, 324, 64)
Conv2D(128) + BN + MaxPool → (3, 162, 128)
Flatten + Dense(256) + Dropout
↓
Latent Space: z_mean, z_log_var (32-dim)
```

### Lyrics Encoder
```
Input: (384,) [Sentence-BERT embeddings]
↓
Dense(256) + BN + Dropout
Dense(128) + BN
↓
Latent Space: z_mean, z_log_var (32-dim)
```

### Hybrid Combination
```
Audio Latent (32) + Lyrics Latent (32)
↓
Concatenate (64-dim)
↓
Dense(64) - Combined Latent Space
```

## Clustering Methods

### 1. K-Means (Baseline)
- Works on: Hybrid features, Audio-only, PCA baseline
- Parameters: k=10 (number of genres)
- Centroids-based partitioning

### 2. Agglomerative Clustering
- Works on: Hybrid features
- Linkage: Ward (minimizes variance)
- Hierarchical approach

### 3. DBSCAN
- Works on: Hybrid features
- Parameters: eps=3.0, min_samples=5
- Density-based, finds arbitrary shapes
- Can identify noise/outliers

### 4. PCA + K-Means
- Baseline comparison
- PCA: 64 components
- Traditional dimensionality reduction

## Evaluation Metrics

| Metric | Range | Better | Use Case |
|--------|-------|--------|----------|
| Silhouette Score | -1 to 1 | Higher | Cluster cohesion & separation |
| Calinski-Harabasz | 0 to ∞ | Higher | Between vs within variance |
| Davies-Bouldin | 0 to ∞ | **Lower** | Cluster similarity |
| Adjusted Rand Index | -1 to 1 | Higher | Agreement with ground truth |
| Normalized Mutual Info | 0 to 1 | Higher | Information overlap with labels |

## Expected Results

### Typical Performance (approximate)

```
Method                           Silhouette  Calinski-H  Davies-B  ARI    NMI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conv VAE Hybrid + K-Means        0.18-0.25   180-250     1.8-2.5   0.08   0.18
Conv VAE Audio + K-Means         0.15-0.22   150-220     2.0-2.8   0.06   0.15
Conv VAE Hybrid + Agglomerative  0.16-0.23   170-240     1.9-2.7   0.07   0.16
Conv VAE Hybrid + DBSCAN         variable    variable    variable  0.05   0.12
PCA + K-Means (Baseline)         0.10-0.18   100-180     2.5-3.5   0.04   0.10
```

**Key Finding**: Hybrid features typically outperform audio-only and significantly outperform PCA baseline.

## Key Improvements Over Easy Task

| Aspect | Easy Task | Medium Task |
|--------|-----------|-------------|
| **Architecture** | Dense layers | Convolutional layers |
| **Features** | Audio only | Audio + Lyrics (hybrid) |
| **Clustering** | K-Means only | K-Means, Agg, DBSCAN |
| **Metrics** | 2 metrics | 5 comprehensive metrics |
| **Latent Dim** | 32 | 64 (combined) |

## Outputs

### Visualizations
1. `medium_training_history.png` - Loss curves (total, reconstruction, KL)
2. `medium_metrics_comparison.png` - Bar charts for all 4 metrics
3. `medium_tsne_visualization.png` - t-SNE plots (true labels vs clusters)

### Data Files
1. `medium_task_metrics.csv` - All metrics for all methods
2. `medium_combined_latent.npy` - Hybrid latent representations
3. `medium_audio_latent.npy` - Audio-only latent features
4. `medium_clusters_*.npy` - Cluster assignments for each method
5. `medium_*_encoder.keras` - Trained models

### Final Package
- `medium_task_results.zip` - Everything bundled

## Analysis & Discussion

### Why Convolutional Layers?

MFCCs have 2D structure (frequency × time). Convolutional layers:
- Preserve spatial relationships
- Learn local patterns (textures in spectrograms)
- Reduce parameters compared to dense layers
- Better generalization

### Why Hybrid Features?

Music genre depends on:
1. **Audio**: Rhythm, melody, instrumentation, timbre
2. **Lyrics**: Themes, vocabulary, sentiment, topics

Combining both modalities provides richer representation.

### Algorithm Comparison

**K-Means**:
- ✓ Fast, scalable
- ✓ Works well with spherical clusters
- ✗ Requires k to be specified

**Agglomerative**:
- ✓ Hierarchical structure
- ✓ No need to specify k upfront
- ✗ Slower on large datasets

**DBSCAN**:
- ✓ Finds arbitrary shapes
- ✓ Identifies outliers
- ✗ Sensitive to parameter selection
- ✗ Variable cluster count

## Troubleshooting

### Issue: DBSCAN finds only 1-2 clusters
**Solution**: Adjust `eps` parameter (try 2.0, 2.5, 3.0, 4.0)

### Issue: Training takes too long
**Solution**: 
- Reduce epochs to 30
- Use smaller batch size (16)
- Ensure GPU is enabled

### Issue: Lyrics embeddings fail
**Solution**: Synthetic lyrics are used as fallback (genre-based themes)

## Next Steps: Hard Task

To extend to Hard Task:
- Implement Conditional VAE (CVAE) or Beta-VAE
- Add genre conditioning
- Include additional modalities (tempo, key, mood)
- Use Cluster Purity metric
- More sophisticated lyrics processing

## References

1. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
2. Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
3. Reimers, N., & Gurevych, I. (2019). Sentence-BERT. EMNLP.
4. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.

## License

Educational project for Neural Networks course.
