# Dataset Documentation

## 📊 Primary Dataset: GTZAN Genre Collection

### Overview

The GTZAN Genre Collection is the most widely used public dataset for music genre recognition research. Created by George Tzanetakis in 2000-2001, it has become the standard benchmark for music information retrieval (MIR) tasks.

**Dataset Statistics:**
- **Total Tracks:** 1,000 audio files
- **Genres:** 10 (100 tracks per genre)
- **Duration:** 30 seconds per track
- **Format:** WAV files (22,050 Hz, 16-bit, mono)
- **Total Size:** ~1.2 GB

### Genres Included

| Genre | Count | Description |
|-------|-------|-------------|
| **Blues** | 100 | Traditional and contemporary blues music |
| **Classical** | 100 | Orchestral, chamber, solo classical pieces |
| **Country** | 100 | Country and western music |
| **Disco** | 100 | 1970s-80s disco and funk |
| **Hip-hop** | 100 | Rap and hip-hop from various eras |
| **Jazz** | 100 | Jazz standards, bebop, swing |
| **Metal** | 100 | Heavy metal and hard rock |
| **Pop** | 100 | Mainstream pop music |
| **Reggae** | 100 | Reggae and dancehall |
| **Rock** | 100 | Rock music from various subgenres |

---

## 🔗 Download Links

### Official Sources

**1. Kaggle (Recommended - Easiest)**
```
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
```
- **Pros:** Easy download, well-organized, includes metadata
- **Cons:** Requires Kaggle account
- **Size:** ~1.2 GB

**2. Marsyas (Original Source)**
```
http://marsyas.info/downloads/datasets.html
```
- **Pros:** Original official source
- **Cons:** Direct download, no organization
- **Size:** ~1.2 GB

**3. Google Drive Mirror (Community)**
```
https://drive.google.com/drive/folders/1-6kJgUM62d-qJCVDqr4_hqr9F2qLqSLN
```
- **Pros:** Fast download, no account needed
- **Cons:** Unofficial mirror
- **Size:** ~1.2 GB

---

## 📥 Download Instructions

### Method 1: Using Kaggle API (Recommended for Notebooks)

**Step 1: Setup Kaggle API**

1. Create Kaggle account: https://www.kaggle.com/
2. Go to Account settings → API → "Create New API Token"
3. Download `kaggle.json` file

**Step 2: Configure in Google Colab/Jupyter**

```python
# Upload kaggle.json when prompted
from google.colab import files
files.upload()

# Install Kaggle API
!pip install -q kaggle

# Setup credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Step 3: Download Dataset**

```python
# Download GTZAN dataset
!kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification

# Unzip
!unzip -q gtzan-dataset-music-genre-classification.zip

# Dataset will be in: Data/genres_original/
```

### Method 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
2. Click "Download" button
3. Extract ZIP file
4. Upload to your workspace

### Method 3: Direct Download (Python Script)

```python
import requests
import zipfile
import os

def download_gtzan():
    """Download GTZAN dataset from mirror"""
    url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    
    print("Downloading GTZAN dataset...")
    response = requests.get(url, stream=True)
    
    with open("genres.tar.gz", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Extracting...")
    os.system("tar -xzf genres.tar.gz")
    print("Done!")

# download_gtzan()
```

---

## 📁 Dataset Structure

### After Download

```
Data/
├── genres_original/          # Original WAV files
│   ├── blues/
│   │   ├── blues.00000.wav
│   │   ├── blues.00001.wav
│   │   └── ... (100 files)
│   ├── classical/
│   │   ├── classical.00000.wav
│   │   └── ... (100 files)
│   ├── country/
│   ├── disco/
│   ├── hiphop/
│   ├── jazz/
│   ├── metal/
│   ├── pop/
│   ├── reggae/
│   └── rock/
│
├── images_original/          # Spectrograms (optional)
│   └── ... (1000 PNG files)
│
└── features_30_sec.csv       # Pre-extracted features (optional)
```

### File Naming Convention

```
{genre}.{index}.wav

Examples:
- blues.00000.wav    → First blues track
- classical.00099.wav → Last classical track
- hiphop.00050.wav   → 51st hip-hop track
```

---

## 🎵 Audio Specifications

### Technical Details

| Property | Value |
|----------|-------|
| **Sample Rate** | 22,050 Hz |
| **Bit Depth** | 16-bit |
| **Channels** | Mono |
| **Duration** | 30 seconds |
| **Format** | WAV (uncompressed) |
| **File Size** | ~1.3 MB per track |

### Audio Quality

- **Codec:** PCM (uncompressed)
- **Bitrate:** ~706 kbps
- **Frequency Range:** 0-11,025 Hz (Nyquist)
- **Dynamic Range:** ~96 dB (16-bit)

---

## 🔧 Dataset Usage in This Project

### Data Sampling

For efficient experimentation, we use a **stratified sample**:
- **40 songs per genre** (instead of 100)
- **Total: 400 songs** (instead of 1,000)
- **Maintains class balance:** 10% of dataset per genre

```python
import os
import random

def load_sample_dataset(data_dir, samples_per_genre=40):
    """Load stratified sample of GTZAN dataset"""
    
    genres = ['blues', 'classical', 'country', 'disco', 
              'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    audio_files = []
    labels = []
    
    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        all_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        
        # Random sample
        sample_files = random.sample(all_files, samples_per_genre)
        
        for file in sample_files:
            audio_files.append(os.path.join(genre_path, file))
            labels.append(genre)
    
    return audio_files, labels

# Usage
audio_files, labels = load_sample_dataset('Data/genres_original/', samples_per_genre=40)
print(f"Loaded {len(audio_files)} audio files")
```

### Feature Extraction

**Audio Features (MFCC):**

```python
import librosa
import numpy as np

def extract_mfcc_features(audio_path):
    """Extract MFCC features from audio file"""
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, duration=30)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20,        # 20 coefficients
        n_fft=2048,       # FFT window size
        hop_length=512    # Hop size
    )
    
    # Flatten: (20, 1293) → (25,860)
    mfcc_flat = mfcc.flatten()
    
    return mfcc_flat

# Usage
features = extract_mfcc_features('Data/genres_original/blues/blues.00000.wav')
print(f"Feature shape: {features.shape}")  # (25860,)
```

**Lyrics Features (Proxy):**

Since GTZAN doesn't include lyrics, we use **genre-representative text themes**:

```python
from sentence_transformers import SentenceTransformer

# Genre themes (proxy for actual lyrics)
GENRE_THEMES = {
    'blues': 'sad emotional heartbreak soulful pain melancholy lonely crying',
    'classical': 'orchestral symphonic elegant refined sophisticated artistic',
    'country': 'rural countryside heartbreak trucks love small town',
    'disco': 'dance party funky groove celebration nightlife energetic',
    'hiphop': 'urban street rhythm beats rap culture city',
    'jazz': 'improvisation sophisticated smooth swing instrumental',
    'metal': 'aggressive powerful intense heavy dark loud',
    'pop': 'catchy mainstream love relationships upbeat radio friendly',
    'reggae': 'relaxed island rhythm peaceful jamaican positive vibes',
    'rock': 'guitars energetic rebellious loud attitude electric'
}

# Extract embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_lyrics_features(genre):
    """Get lyrics embedding for genre"""
    theme = GENRE_THEMES[genre]
    embedding = model.encode(theme)  # (384,)
    return embedding
```

---

## ⚠️ Known Issues & Considerations

### Dataset Limitations

1. **Size:** Only 1,000 tracks (relatively small for deep learning)
2. **Quality:** Some tracks have inconsistent audio quality
3. **Genre Overlap:** Fuzzy genre boundaries (e.g., blues vs. jazz)
4. **Era Bias:** Tracks mostly from 1950s-1990s
5. **No Lyrics:** Only audio, no textual information
6. **Duplicate Issue:** Some tracks appear multiple times (jazz.00054.wav)

### Preprocessing Recommendations

**1. Remove Duplicates:**
```python
# Known duplicates in jazz category
duplicates = ['jazz.00054.wav']  # Check and remove if found
```

**2. Normalize Audio:**
```python
# Normalize volume levels
y = librosa.util.normalize(y)
```

**3. Handle Silence:**
```python
# Trim leading/trailing silence
y, _ = librosa.effects.trim(y, top_db=20)
```

---

## 📊 Dataset Statistics

### Genre Distribution (Our Sample)

| Genre | Tracks | Percentage |
|-------|--------|------------|
| Blues | 40 | 10% |
| Classical | 40 | 10% |
| Country | 40 | 10% |
| Disco | 40 | 10% |
| Hip-hop | 40 | 10% |
| Jazz | 40 | 10% |
| Metal | 40 | 10% |
| Pop | 40 | 10% |
| Reggae | 40 | 10% |
| Rock | 40 | 10% |
| **Total** | **400** | **100%** |

### Audio Characteristics

**Tempo Distribution:**
- Blues: 60-120 BPM (slow to moderate)
- Classical: Variable (40-180 BPM)
- Country: 80-140 BPM
- Disco: 110-130 BPM (dance tempo)
- Hip-hop: 80-110 BPM
- Jazz: 100-200 BPM (wide range)
- Metal: 120-180 BPM (fast)
- Pop: 100-130 BPM
- Reggae: 60-90 BPM (slow, steady)
- Rock: 100-160 BPM

---

## 🔄 Alternative Datasets (For Future Work)

### 1. Million Song Dataset (MSD)
- **Size:** 1 million tracks
- **Features:** Pre-extracted audio features, metadata
- **Link:** http://millionsongdataset.com/
- **Pros:** Very large, includes metadata
- **Cons:** No raw audio (features only)

### 2. Free Music Archive (FMA)
- **Size:** 106,574 tracks
- **Genres:** 161 genres
- **Link:** https://github.com/mdeff/fma
- **Pros:** Large, diverse, open license
- **Cons:** Imbalanced classes

### 3. Jamendo Dataset
- **Size:** 18,486 tracks
- **Features:** Audio + metadata + lyrics
- **Link:** https://www.kaggle.com/datasets/andradaolteanu/jamendo-music-dataset
- **Pros:** Includes lyrics, diverse
- **Cons:** Smaller than MSD

### 4. MIR-1K Dataset
- **Size:** 1,000 clips
- **Languages:** Mandarin + English
- **Link:** https://sites.google.com/site/unvoicedsoundseparation/mir-1k
- **Pros:** Multi-language
- **Cons:** Small, specialized

---

## 📚 Citation

If you use the GTZAN dataset, please cite:

```bibtex
@article{tzanetakis2002gtzan,
  title={Musical genre classification of audio signals},
  author={Tzanetakis, George and Cook, Perry},
  journal={IEEE Transactions on Speech and Audio Processing},
  volume={10},
  number={5},
  pages={293--302},
  year={2002},
  publisher={IEEE}
}
```

**APA Format:**
```
Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio 
signals. IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.
```

---

## 🛠️ Quick Start Code

### Complete Data Loading Pipeline

```python
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class GTZANDataLoader:
    """Complete GTZAN dataset loader"""
    
    def __init__(self, data_dir, samples_per_genre=40):
        self.data_dir = data_dir
        self.samples_per_genre = samples_per_genre
        self.genres = ['blues', 'classical', 'country', 'disco', 
                      'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def load_dataset(self):
        """Load audio files and labels"""
        audio_files = []
        labels = []
        
        for genre in self.genres:
            genre_path = os.path.join(self.data_dir, genre)
            files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            
            # Sample
            import random
            random.seed(42)
            sampled = random.sample(files, self.samples_per_genre)
            
            for file in sampled:
                audio_files.append(os.path.join(genre_path, file))
                labels.append(genre)
        
        return audio_files, labels
    
    def extract_features(self, audio_files, labels):
        """Extract MFCC features from all files"""
        features = []
        
        for audio_path in tqdm(audio_files, desc="Extracting features"):
            try:
                # Load audio
                y, sr = librosa.load(audio_path, sr=22050, duration=30)
                
                # Extract MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_flat = mfcc.flatten()
                
                features.append(mfcc_flat)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        return np.array(features), np.array(labels)

# Usage
loader = GTZANDataLoader('Data/genres_original/', samples_per_genre=40)
audio_files, labels = loader.load_dataset()
X, y = loader.extract_features(audio_files, labels)

print(f"Features shape: {X.shape}")  # (400, 25860)
print(f"Labels shape: {y.shape}")    # (400,)
```

---

## 🔍 Data Exploration

### Genre Analysis Script

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_genre_distribution(labels):
    """Plot genre distribution"""
    from collections import Counter
    
    genre_counts = Counter(labels)
    
    plt.figure(figsize=(12, 6))
    plt.bar(genre_counts.keys(), genre_counts.values())
    plt.title('GTZAN Dataset - Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Tracks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_audio_characteristics(audio_files):
    """Analyze audio characteristics"""
    durations = []
    sample_rates = []
    
    for file in audio_files[:50]:  # Sample
        y, sr = librosa.load(file, sr=None)
        durations.append(librosa.get_duration(y=y, sr=sr))
        sample_rates.append(sr)
    
    print(f"Average duration: {np.mean(durations):.2f} seconds")
    print(f"Sample rate: {sample_rates[0]} Hz")

# Run analysis
analyze_genre_distribution(labels)
analyze_audio_characteristics(audio_files)
```

---

## 💾 Storage Requirements

### Disk Space

**Full Dataset:**
- Raw audio (WAV): ~1.2 GB
- Extracted features (NPY): ~400 MB
- Total: ~1.6 GB

**Our Sample (400 tracks):**
- Raw audio: ~500 MB
- Extracted features: ~160 MB
- Total: ~660 MB

### Memory Requirements

**During Processing:**
- Loading all audio: ~2 GB RAM
- Feature extraction: ~4 GB RAM
- Model training: ~2 GB RAM
- **Recommended:** 8 GB RAM minimum

---

## 📧 Support

**Dataset Issues:**
- Report to: marsyas@cs.uvic.ca
- GitHub Issues: https://github.com/marsyas/marsyas

**Project Issues:**
- Check our Issues page on GitHub
- Contact project maintainers

---

## ✅ Verification Checklist

After downloading:
- [ ] Confirm 10 genre folders
- [ ] Each genre has 100 WAV files
- [ ] Files are 30 seconds long
- [ ] Sample rate is 22,050 Hz
- [ ] Files are mono (1 channel)
- [ ] Total size ~1.2 GB

---

## 🎯 Summary

- **Dataset:** GTZAN Genre Collection
- **Size:** 1,000 tracks (we use 400)
- **Source:** Kaggle or Marsyas
- **Format:** WAV, 22,050 Hz, mono
- **Usage:** Feature extraction → Clustering
- **Citation:** Tzanetakis & Cook (2002)

**Ready to use in your notebooks!** 🎵
