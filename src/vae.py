"""
VAE Model Implementation for Music Clustering
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(input_shape, latent_dim):
    """
    Build encoder network.
    
    Args:
        input_shape: Shape of input (n_mfcc, time_steps, 1)
        latent_dim: Dimension of latent space
    
    Returns:
        encoder: Keras Model
    """
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_input')
    
    x = layers.Flatten()(encoder_inputs)
    x = layers.Dense(512, activation='relu', name='encoder_dense_1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu', name='encoder_dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu', name='encoder_dense_3')(x)
    
    # Latent space parameters
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sample from latent space
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder


def build_decoder(latent_dim, output_shape):
    """
    Build decoder network.
    
    Args:
        latent_dim: Dimension of latent space
        output_shape: Shape of output (n_mfcc, time_steps, 1)
    
    Returns:
        decoder: Keras Model
    """
    latent_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    
    x = layers.Dense(128, activation='relu', name='decoder_dense_1')(latent_inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(256, activation='relu', name='decoder_dense_2')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(512, activation='relu', name='decoder_dense_3')(x)
    x = layers.BatchNormalization()(x)
    
    # Output layer
    output_dim = np.prod(output_shape)
    x = layers.Dense(output_dim, activation='linear', name='decoder_output')(x)
    
    # Reshape to original shape
    decoder_outputs = layers.Reshape(output_shape)(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    return decoder


class VAE(keras.Model):
    """Variational Autoencoder model."""
    
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction), axis=(1, 2, 3)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Total loss
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction


def create_vae(input_shape, latent_dim=32, learning_rate=1e-3):
    """
    Create and compile VAE model.
    
    Args:
        input_shape: Shape of input data
        latent_dim: Dimension of latent space
        learning_rate: Learning rate for optimizer
    
    Returns:
        vae: Compiled VAE model
        encoder: Encoder model
        decoder: Decoder model
    """
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    return vae, encoder, decoder
