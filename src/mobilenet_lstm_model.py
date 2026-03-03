# ==========================================
# STEP 1: NEW MODEL ARCHITECTURE
# File: mobilenet_lstm_model.py
# ==========================================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import numpy as np

@tf.keras.utils.register_keras_serializable(package="Custom", name="GrayscaleToRGB")
class GrayscaleToRGB(Layer):
    """Module-level custom layer to convert a single-channel (grayscale)
    image to 3 channels by repeating the channel. Registering this at
    module import time ensures Keras can locate the class when loading
    models saved with this custom layer.
    """
    def __init__(self, **kwargs):
        super(GrayscaleToRGB, self).__init__(**kwargs)

    def call(self, inputs):
        # Repeat the single channel 3 times
        return tf.concat([inputs, inputs, inputs], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)

    def get_config(self):
        return super(GrayscaleToRGB, self).get_config()

def build_mobilenet_lstm_model(num_frames=22, num_classes=10, img_height=80, img_width=112):
    """
    MobileNet + LSTM Architecture for Lip Reading
    
    Args:
        num_frames: Number of frames in video sequence (default: 22)
        num_classes: Number of word classes to predict
        img_height: Height of lip region (default: 80)
        img_width: Width of lip region (default: 112)
    
    Returns:
        Keras Model
    """
    
    # Input layer for video sequence
    input_frames = Input(shape=(num_frames, img_height, img_width, 1), name='video_input')
    
    # Convert grayscale to RGB using module-level custom layer
    x = TimeDistributed(GrayscaleToRGB(), name='grayscale_to_rgb')(input_frames)
    
    # Load MobileNetV2 as feature extractor
    mobilenet_base = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,  # Remove classification head
        weights='imagenet',  # Use pre-trained weights
        pooling='avg'  # Global average pooling
    )
    
    # Freeze early layers (transfer learning)
    # Only fine-tune last 30 layers
    for layer in mobilenet_base.layers[:-30]:
        layer.trainable = False
    
    # Apply MobileNet to each frame independently
    # TimeDistributed applies the same model to each time step
    x = TimeDistributed(mobilenet_base, name='mobilenet_features')(x)
    
    # Now x has shape: (batch, num_frames, 1280)
    # 1280 is MobileNetV2 feature dimension
    
    # ==========================================
    # TEMPORAL MODELING WITH LSTM
    # ==========================================
    
    # Bidirectional LSTM to capture temporal patterns
    # Goes forward and backward through the sequence
    x = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='lstm_1'
    )(x)
    
    # Add attention mechanism (optional but improves accuracy)
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(256)(attention)  # 256 = 128*2 (bidirectional)
    attention = Permute([2, 1])(attention)
    
    # Apply attention weights
    x = Multiply()([x, attention])
    
    # Second LSTM layer
    x = Bidirectional(
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        name='lstm_2'
    )(x)
    
    # Dropout for regularization
    x = Dropout(0.3)(x)
    
    # ==========================================
    # CLASSIFICATION HEAD
    # ==========================================
    
    # Fully connected layers
    x = Dense(128, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='relu', name='fc2')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=input_frames, outputs=outputs, name='MobileNet_LSTM_LipReader')
    
    return model