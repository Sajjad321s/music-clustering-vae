# Implementation Details

## Architecture Design

### Easy Task - Basic Autoencoder

**Purpose:** Establish baseline deep learning performance over traditional PCA.

**Architecture:**
```
Input: MFCC features [25,860 dims]
  ↓
Encoder Layer 1: Dense(128, activation='relu')
  ↓
Encoder Layer 2: Dense(32, activation='relu')  ← Latent Space
  ↓
Decoder Layer 1: Dense(128, activation='relu')
  ↓
Decoder Layer 2: Dense(25,860, activation='linear')
  ↓
Output: Reconstructed MFCC [25,860 dims]
```

**Training:**
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.002)
- Epochs: 150
- No regularization

**Results:**
- Silhouette: 0.16 (+6.7% over PCA)
- Training time: ~5 minutes

---

### Medium Task - Multi-Modal Autoencoder

**Purpose:** Demonstrate benefits of multi-modal fusion.

**Audio Branch:**
```
MFCC [25,860] → Dense(128) → Dense(32) → Audio Latent [32]
```

**Lyrics Branch:**
```
Sentence-BERT [384] → PCA reduction → Lyrics Latent [16]
```

**Fusion:**
```
Concatenate [Audio(32) + Lyrics(16)] → Multi-Modal Latent [48]
```

**Key Innovation:**
- Separate encoding paths for different modalities
- Simple concatenation for fusion
- No cross-modal attention (keeps it simple)

**Results:**
- Silhouette: 0.20 (+33% over PCA, +25% over Easy)
- Training time: ~8 minutes
- **Largest single contribution: +26.6%**

---

### Hard Task - Conditional Multi-Modal Autoencoder

**Purpose:** Add genre conditioning for disentangled representations.

**Conditional Encoding:**
```
Input: [Audio(25,860) + Genre_OneHot(10)]
  ↓
Dense(256, 'relu')
  ↓
Dense(128, 'relu')
  ↓
Dense(32, 'relu') → Conditional Audio Latent [32]
```

**Final Multi-Modal Representation:**
```
z_final = [z_cond_audio(32) + z_lyrics(16) + genre(10)] → [58 dims]
```

**Key Innovation:**
- Genre information injected during encoding
- Enables disentanglement of genre-specific vs. within-genre features
- Similar to Conditional VAE (CVAE) but deterministic

**Results:**
- Silhouette: 0.22 (+47% over PCA)
- Training time: ~15-20 minutes
- Conditional encoding adds: +13.3%

---

## Feature Engineering

### Audio Features (MFCC)

**Extraction Process:**
```python
import librosa

# Load audio
y, sr = librosa.load(audio_path, sr=22050, duration=30)

# Extract MFCCs
mfcc = librosa.feature.mfcc(
    y=y, 
    sr=sr, 
    n_mfcc=20,      # 20 coefficients
    n_fft=2048,
    hop_length=512
)

# Result: shape (20, 1293) → flatten to (25,860,)
```

**Why MFCC?**
- Captures timbral characteristics
- Compact representation
- Standard in music information retrieval
- 20 coefficients balance detail vs. dimensionality

### Lyrics Features (Sentence-BERT)

**Extraction Process:**
```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Genre-representative themes (proxy for actual lyrics)
themes = {
    'blues': 'sad emotional heartbreak soulful pain',
    'classical': 'orchestral symphonic elegant refined',
    'country': 'rural countryside heartbreak trucks',
    # ... etc
}

# Encode
embeddings = model.encode(themes[genre])  # (384,)

# Reduce to 16 dims via PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
lyrics_reduced = pca.fit_transform(embeddings)  # (16,)
```

**Why Proxy Lyrics?**
- Actual GTZAN doesn't include lyrics
- Demonstrates concept validity
- Future work: real lyrics from lyrics datasets

### Genre Encoding (One-Hot)

```python
genre_to_idx = {
    'blues': 0, 'classical': 1, 'country': 2,
    'disco': 3, 'hiphop': 4, 'jazz': 5,
    'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
}

# One-hot encoding
genre_vec = np.zeros(10)
genre_vec[genre_to_idx[genre]] = 1  # (10,)
```

---

## Clustering Methods

### K-Means (Primary Method)

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=10,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

clusters = kmeans.fit_predict(latent_features)
```

**Why K-Means?**
- Simple and interpretable
- Known number of clusters (10 genres)
- Fast convergence
- Industry standard baseline

### Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=10,
    linkage='ward'
)

clusters = agg.fit_predict(latent_features)
```

**Advantage:** Hierarchical structure reveals genre relationships

### Spectral Clustering

```python
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=10,
    affinity='nearest_neighbors',
    n_neighbors=10,
    random_state=42
)

clusters = spectral.fit_predict(latent_features)
```

**Advantage:** Handles non-convex clusters

---

## Evaluation Metrics

### 1. Silhouette Score

**Formula:**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
  a(i) = avg distance to points in same cluster
  b(i) = avg distance to nearest different cluster
```

**Implementation:**
```python
from sklearn.metrics import silhouette_score

score = silhouette_score(
    X=latent_features,
    labels=cluster_assignments,
    metric='euclidean'
)
```

**Range:** [-1, 1], higher is better

### 2. Normalized Mutual Information (NMI)

**Formula:**
```
NMI(U,V) = 2 * I(U;V) / (H(U) + H(V))

where:
  I(U;V) = mutual information
  H(U), H(V) = entropies
```

**Implementation:**
```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(
    labels_true=true_genres,
    labels_pred=cluster_assignments
)
```

**Range:** [0, 1], higher is better

### 3. Adjusted Rand Index (ARI)

**Formula:**
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])

where:
  RI = Rand Index (fraction of pairs correctly assigned)
```

**Implementation:**
```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(
    labels_true=true_genres,
    labels_pred=cluster_assignments
)
```

**Range:** [-1, 1], higher is better

### 4. Cluster Purity

**Formula:**
```
Purity = (1/N) * Σ max_j |c_k ∩ t_j|

where:
  c_k = cluster k
  t_j = true class j
  N = total points
```

**Implementation:**
```python
def cluster_purity(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)
```

**Range:** [0, 1], higher is better

---

## Hyperparameter Tuning

### Learning Rate Search

```python
learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.005]

# Best: 0.002 (good balance of speed and stability)
```

### Latent Dimension Search

```python
latent_dims = [16, 32, 64, 128]

# Best: 32 (sufficient capacity without overfitting)
```

### Epoch Count

```python
epochs = [50, 100, 150, 200]

# Best: 150 (convergence without overtraining)
```

---

## Computational Efficiency

### Training Time Breakdown

**Easy Task (~5 min):**
- Data loading: 30 sec
- Feature extraction: 2 min
- Training: 2 min
- Clustering + evaluation: 30 sec

**Medium Task (~8 min):**
- Data loading: 30 sec
- Feature extraction: 3 min (audio + lyrics)
- Training: 3 min
- Clustering + evaluation: 1.5 min

**Hard Task (~15-20 min):**
- Data loading: 30 sec
- Feature extraction: 4 min
- Training: 8-10 min (larger network)
- Clustering + evaluation: 2-3 min
- Visualization generation: 1-2 min

### Memory Usage

- Peak RAM: ~8 GB (during MFCC extraction)
- Model size: <5 MB (sklearn MLPRegressor)
- Dataset: ~400 audio files × 30 sec = ~2 GB

---

## Code Quality Practices

### Error Handling

```python
try:
    audio, sr = librosa.load(path, sr=22050)
except Exception as e:
    print(f"Failed to load {path}: {e}")
    continue
```

### Progress Tracking

```python
from tqdm import tqdm

for file in tqdm(audio_files, desc="Processing"):
    # Process file
```

### Reproducibility

```python
import random
import numpy as np

# Set seeds
random.seed(42)
np.random.seed(42)
```

---

## Future Improvements

### 1. True VAE Implementation
- Add KL divergence term
- Probabilistic latent space
- Better disentanglement

### 2. Attention Mechanisms
- Cross-modal attention between audio and lyrics
- Self-attention in encoder

### 3. Temporal Modeling
- Use RNNs/Transformers for sequential audio
- Capture long-range dependencies

### 4. Real Lyrics
- Integrate actual lyrics datasets
- More accurate semantic information

### 5. Larger Dataset
- Use full GTZAN (1000 songs)
- Add Million Song Dataset
- Multi-language evaluation

---

## Troubleshooting

### Common Issues

**Issue 1: librosa installation fails**
```bash
# Solution: Install dependencies first
pip install numba
pip install librosa
```

**Issue 2: Kaggle API not working**
```bash
# Solution: Check kaggle.json permissions
chmod 600 ~/.kaggle/kaggle.json
```

**Issue 3: Out of memory**
```bash
# Solution: Reduce batch size or use fewer songs
n_samples_per_genre = 20  # instead of 40
```

**Issue 4: Training doesn't converge**
```bash
# Solution: Try different learning rate
lr = 0.001  # lower if oscillating
```

---

## Performance Benchmarks

| Task | Hardware | Training Time | Inference Time |
|------|----------|---------------|----------------|
| Easy | Colab CPU | 5 min | <1 sec |
| Medium | Colab CPU | 8 min | <1 sec |
| Hard | Colab CPU | 15-20 min | <2 sec |
| Easy | Local GPU | 2 min | <0.1 sec |
| Medium | Local GPU | 4 min | <0.1 sec |
| Hard | Local GPU | 8 min | <0.2 sec |

---

## References

1. scikit-learn MLPRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
2. librosa documentation: https://librosa.org/doc/latest/index.html
3. Sentence-BERT: https://www.sbert.net/
4. GTZAN dataset: http://marsyas.info/downloads/datasets.html
