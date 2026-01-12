"""
Convolutional VAE Models for Medium Task
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class Sampling(layers.Layer):
    """Sampling layer for VAE."""
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_conv_encoder(input_shape, latent_dim):
    """
    Build convolutional encoder for audio features.
    
    Args:
        input_shape: Shape of input (n_mfcc, time_steps, 1)
        latent_dim: Dimension of latent space
    
    Returns:
        encoder: Keras Model with convolutional layers
    """
    encoder_inputs = keras.Input(shape=input_shape, name='audio_input')
    
    # Convolutional layers with batch normalization
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Latent space
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='conv_encoder')
    return encoder


def build_conv_decoder(latent_dim, output_shape):
    """
    Build convolutional decoder for audio reconstruction.
    
    Args:
        latent_dim: Dimension of latent space
        output_shape: Shape of output (n_mfcc, time_steps, 1)
    
    Returns:
        decoder: Keras Model with transpose convolutions
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    
    # Calculate intermediate dimensions after pooling
    # After 3 max pooling (2,2): height and width are reduced by factor of 8
    intermediate_h = (output_shape[0] + 7) // 8
    intermediate_w = (output_shape[1] + 7) // 8
    intermediate_channels = 128
    
    x = layers.Dense(intermediate_h * intermediate_w * intermediate_channels, activation='relu')(latent_inputs)
    x = layers.Reshape((intermediate_h, intermediate_w, intermediate_channels))(x)
    
    # Transpose convolution layers (upsampling)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output layer
    x = layers.Conv2DTranspose(1, (3, 3), activation='linear', padding='same')(x)
    
    # Crop to exact output shape if needed
    current_h = intermediate_h * 8
    current_w = intermediate_w * 8
    
    if current_h > output_shape[0] or current_w > output_shape[1]:
        crop_h = (current_h - output_shape[0]) // 2
        crop_w = (current_w - output_shape[1]) // 2
        x = layers.Cropping2D(cropping=((crop_h, current_h - output_shape[0] - crop_h), 
                                        (crop_w, current_w - output_shape[1] - crop_w)))(x)
    
    decoder = Model(latent_inputs, x, name='conv_decoder')
    return decoder


def build_lyrics_encoder(input_dim, latent_dim):
    """
    Build encoder for lyrics embeddings.
    
    Args:
        input_dim: Dimension of lyrics embeddings
        latent_dim: Dimension of latent space
    
    Returns:
        encoder: Keras Model for lyrics
    """
    encoder_inputs = keras.Input(shape=(input_dim,), name='lyrics_input')
    
    x = layers.Dense(256, activation='relu')(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Latent space
    z_mean = layers.Dense(latent_dim, name='lyrics_z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='lyrics_z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='lyrics_encoder')
    return encoder


class HybridVAE(keras.Model):
    """
    Hybrid VAE combining audio and lyrics modalities.
    """
    
    def __init__(self, audio_encoder, audio_decoder, lyrics_encoder, combined_latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.audio_encoder = audio_encoder
        self.audio_decoder = audio_decoder
        self.lyrics_encoder = lyrics_encoder
        
        # Combined latent space projection
        self.combine_layer = layers.Dense(combined_latent_dim, activation='relu', name='combine')
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.audio_recon_loss_tracker = keras.metrics.Mean(name="audio_recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.audio_recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        audio_data, lyrics_data = data
        
        with tf.GradientTape() as tape:
            # Encode both modalities
            audio_z_mean, audio_z_log_var, audio_z = self.audio_encoder(audio_data)
            lyrics_z_mean, lyrics_z_log_var, lyrics_z = self.lyrics_encoder(lyrics_data)
            
            # Combine latent representations
            combined_z = tf.concat([audio_z, lyrics_z], axis=1)
            combined_latent = self.combine_layer(combined_z)
            
            # Reconstruct audio
            audio_recon = self.audio_decoder(audio_z)
            
            # Reconstruction loss
            audio_recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(audio_data - audio_recon), axis=(1, 2, 3)
                )
            )
            
            # KL divergence (both modalities)
            audio_kl = -0.5 * tf.reduce_sum(
                1 + audio_z_log_var - tf.square(audio_z_mean) - tf.exp(audio_z_log_var), axis=1
            )
            lyrics_kl = -0.5 * tf.reduce_sum(
                1 + lyrics_z_log_var - tf.square(lyrics_z_mean) - tf.exp(lyrics_z_log_var), axis=1
            )
            kl_loss = tf.reduce_mean(audio_kl + lyrics_kl)
            
            # Total loss
            total_loss = audio_recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.audio_recon_loss_tracker.update_state(audio_recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "audio_recon_loss": self.audio_recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        audio_data, lyrics_data = data
        
        # Encode
        audio_z_mean, audio_z_log_var, audio_z = self.audio_encoder(audio_data)
        lyrics_z_mean, lyrics_z_log_var, lyrics_z = self.lyrics_encoder(lyrics_data)
        
        # Combine
        combined_z = tf.concat([audio_z, lyrics_z], axis=1)
        combined_latent = self.combine_layer(combined_z)
        
        # Reconstruct
        audio_recon = self.audio_decoder(audio_z)
        
        # Losses
        audio_recon_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(audio_data - audio_recon), axis=(1, 2, 3))
        )
        
        audio_kl = -0.5 * tf.reduce_sum(
            1 + audio_z_log_var - tf.square(audio_z_mean) - tf.exp(audio_z_log_var), axis=1
        )
        lyrics_kl = -0.5 * tf.reduce_sum(
            1 + lyrics_z_log_var - tf.square(lyrics_z_mean) - tf.exp(lyrics_z_log_var), axis=1
        )
        kl_loss = tf.reduce_mean(audio_kl + lyrics_kl)
        
        total_loss = audio_recon_loss + kl_loss
        
        return {
            "total_loss": total_loss,
            "audio_recon_loss": audio_recon_loss,
            "kl_loss": kl_loss,
        }
    
    def get_combined_latent(self, audio_data, lyrics_data):
        """Extract combined latent representation."""
        audio_z_mean, _, _ = self.audio_encoder(audio_data)
        lyrics_z_mean, _, _ = self.lyrics_encoder(lyrics_data)
        combined_z = tf.concat([audio_z_mean, lyrics_z_mean], axis=1)
        combined_latent = self.combine_layer(combined_z)
        return combined_latent


def create_hybrid_vae(audio_input_shape, lyrics_input_dim, 
                      audio_latent_dim=32, combined_latent_dim=64,
                      learning_rate=1e-3):
    """
    Create and compile Hybrid VAE model.
    
    Args:
        audio_input_shape: Shape of audio input
        lyrics_input_dim: Dimension of lyrics embeddings
        audio_latent_dim: Latent dimension for each modality
        combined_latent_dim: Dimension of combined latent space
        learning_rate: Learning rate for optimizer
    
    Returns:
        hybrid_vae: Compiled Hybrid VAE model
        audio_encoder: Audio encoder
        audio_decoder: Audio decoder
        lyrics_encoder: Lyrics encoder
    """
    audio_encoder = build_conv_encoder(audio_input_shape, audio_latent_dim)
    audio_decoder = build_conv_decoder(audio_latent_dim, audio_input_shape)
    lyrics_encoder = build_lyrics_encoder(lyrics_input_dim, audio_latent_dim)
    
    hybrid_vae = HybridVAE(audio_encoder, audio_decoder, lyrics_encoder, combined_latent_dim)
    hybrid_vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    return hybrid_vae, audio_encoder, audio_decoder, lyrics_encoder
