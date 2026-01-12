"""
Dataset Loading and Preprocessing Module
"""

import numpy as np
import librosa
import os
from tqdm import tqdm


def extract_mfcc_features(file_path, sr=22050, duration=30, n_mfcc=20, hop_length=512, max_len=1293):
    """
    Extract MFCC features from audio file.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Duration to load (seconds)
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for MFCC computation
        max_len: Maximum length of MFCC sequence
    
    Returns:
        mfcc: MFCC features of shape (n_mfcc, max_len)
    """
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        
        # Pad or truncate to max_len
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_gtzan_dataset(data_path, genres, n_mfcc=20, sample_rate=22050, duration=30):
    """
    Load GTZAN dataset and extract MFCC features.
    
    Args:
        data_path: Path to GTZAN dataset
        genres: List of genre names
        n_mfcc: Number of MFCC coefficients
        sample_rate: Sample rate for audio loading
        duration: Duration to load (seconds)
    
    Returns:
        X: MFCC features array of shape (n_samples, n_mfcc, time_steps)
        y: Labels array of shape (n_samples,)
        file_paths: List of file paths
    """
    features_list = []
    labels_list = []
    file_paths = []
    
    print("Loading GTZAN dataset and extracting MFCC features...")
    
    for genre_idx, genre in enumerate(genres):
        genre_path = os.path.join(data_path, genre)
        
        if not os.path.exists(genre_path):
            print(f"Warning: {genre_path} not found, skipping...")
            continue
        
        audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        print(f"Processing {genre}: {len(audio_files)} files")
        
        for audio_file in tqdm(audio_files, desc=f"  {genre}"):
            file_path = os.path.join(genre_path, audio_file)
            mfcc = extract_mfcc_features(file_path, sr=sample_rate, duration=duration, n_mfcc=n_mfcc)
            
            if mfcc is not None:
                features_list.append(mfcc)
                labels_list.append(genre_idx)
                file_paths.append(file_path)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    
    return X, y, file_paths


def normalize_features(X):
    """
    Normalize features using mean and standard deviation.
    
    Args:
        X: Input features of shape (n_samples, n_mfcc, time_steps)
    
    Returns:
        X_normalized: Normalized features
    """
    X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    return X_normalized


def prepare_vae_input(X):
    """
    Prepare features for VAE input by adding channel dimension.
    
    Args:
        X: Normalized features of shape (n_samples, n_mfcc, time_steps)
    
    Returns:
        X_input: Features with channel dimension (n_samples, n_mfcc, time_steps, 1)
    """
    X_input = X[..., np.newaxis]
    return X_input


class DataLoader:
    """Data loader class for music dataset."""
    
    def __init__(self, data_path, genres, n_mfcc=20, sample_rate=22050, duration=30):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to dataset
            genres: List of genre names
            n_mfcc: Number of MFCC coefficients
            sample_rate: Sample rate
            duration: Duration to load (seconds)
        """
        self.data_path = data_path
        self.genres = genres
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.duration = duration
        
        self.X = None
        self.y = None
        self.file_paths = None
        self.X_normalized = None
        self.X_input = None
    
    def load_and_preprocess(self):
        """Load dataset and preprocess features."""
        # Load dataset
        self.X, self.y, self.file_paths = load_gtzan_dataset(
            self.data_path, 
            self.genres, 
            self.n_mfcc, 
            self.sample_rate, 
            self.duration
        )
        
        # Normalize features
        self.X_normalized = normalize_features(self.X)
        
        # Prepare VAE input
        self.X_input = prepare_vae_input(self.X_normalized)
        
        print(f"\nPreprocessing completed!")
        print(f"Normalized input shape: {self.X_input.shape}")
        
        return self.X_input, self.y
    
    def get_features(self):
        """Get normalized features."""
        return self.X_normalized
    
    def get_vae_input(self):
        """Get VAE input features."""
        return self.X_input
    
    def get_labels(self):
        """Get labels."""
        return self.y
