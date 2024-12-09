import tensorflow as tf
import cv2
import numpy as np
from typing import List, Tuple

def train_autoencoder(cropped_correct_frames: List[str], model_path: str, input_shape: Tuple[int, int, int] = (128, 128, 3)):
    """
    Trains an autoencoder using cropped frames of correct images.

    Args:
        cropped_correct_frames (List[str]): List of file paths of correctly cropped frames.
        model_path (str): Path to save the trained autoencoder model.
        input_shape (Tuple[int, int, int]): Input shape for the autoencoder (default: 128x128 RGB).
    """
    # Load and preprocess images
    images = []
    for image_path in cropped_correct_frames:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (input_shape[0], input_shape[1]))
        images.append(img / 255.0)  # Normalize to [0, 1]

    images = np.array(images)

    input_layer = tf.keras.layers.Input(shape=input_shape)
    encoded = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    decoded = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    output_layer = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)

    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(images, images, epochs=50, batch_size=16, validation_split=0.2)

    autoencoder.save(model_path)
