"""
Data Loading and Preprocessing Module

This module handles:
- Loading CIFAR-10 dataset
- Data sampling (optional)
- Preprocessing (normalization)
- PCA dimensionality reduction (optional)
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
import config


def load_cifar10_data():
    """
    Load CIFAR-10 dataset.

    Returns:
        tuple: (X_train, y_train), (X_test, y_test)
    """
    print("=" * config.SEPARATOR_LENGTH)
    print("Loading CIFAR-10 dataset...")
    print("=" * config.SEPARATOR_LENGTH)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Reshape labels from (n, 1) to (n,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"Original training data shape: {X_train.shape}")
    print(f"Original test data shape: {X_test.shape}")

    return (X_train, y_train), (X_test, y_test)


def sample_data(X, y, samples_per_class):
    """
    Sample a subset of data with stratification (equal samples per class).

    Args:
        X: Input data
        y: Labels
        samples_per_class: Number of samples to take from each class

    Returns:
        tuple: (X_sampled, y_sampled)
    """
    indices = []
    num_classes = len(config.CLASS_NAMES)

    for class_idx in range(num_classes):
        class_mask = y == class_idx
        class_indices = np.where(class_mask)[0]
        sampled_indices = np.random.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False
        )
        indices.extend(sampled_indices)

    indices = np.array(indices)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def preprocess_data(X_train, y_train, X_test, y_test, use_sampling=True, use_pca=True):
    """
    Preprocess the data: sampling, flattening, normalization, and PCA.

    Args:
        X_train: Training images
        y_train: Training labels
        X_test: Test images
        y_test: Test labels
        use_sampling: Whether to sample a subset of data
        use_pca: Whether to apply PCA

    Returns:
        tuple: (X_train_processed, y_train, X_test_processed, y_test, pca_model)
               pca_model is None if use_pca is False
    """
    # Sample data if enabled
    if use_sampling:
        print(f"\nSampling {config.SAMPLES_PER_CLASS_TRAIN} samples per class for training...")
        print(f"Sampling {config.SAMPLES_PER_CLASS_TEST} samples per class for testing...")

        X_train, y_train = sample_data(X_train, y_train, config.SAMPLES_PER_CLASS_TRAIN)
        X_test, y_test = sample_data(X_test, y_test, config.SAMPLES_PER_CLASS_TEST)

        print(f"\nSampled training data shape: {X_train.shape}")
        print(f"Sampled test data shape: {X_test.shape}")

    # Convert images to 1D vectors (32x32x3 -> 3072)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Normalize pixel values (0-255 -> 0-1)
    X_train_flat = X_train_flat.astype('float32') / 255.0
    X_test_flat = X_test_flat.astype('float32') / 255.0

    print(f"Flattened training data shape: {X_train_flat.shape}")
    print(f"Flattened test data shape: {X_test_flat.shape}")

    # Apply PCA if enabled
    pca_model = None
    if use_pca:
        print(f"\nApplying PCA (reducing from {X_train_flat.shape[1]} to {config.PCA_N_COMPONENTS} dimensions)...")

        pca_model = PCA(
            n_components=config.PCA_N_COMPONENTS,
            whiten=config.PCA_WHITEN,
            random_state=config.RANDOM_SEED
        )

        # Fit PCA on training data
        X_train_flat = pca_model.fit_transform(X_train_flat)

        # Transform test data
        X_test_flat = pca_model.transform(X_test_flat)

        # Print explained variance
        explained_variance = np.sum(pca_model.explained_variance_ratio_)
        print(f"PCA completed: {explained_variance:.4f} ({explained_variance*100:.2f}%) of variance explained")
        print(f"New training data shape: {X_train_flat.shape}")
        print(f"New test data shape: {X_test_flat.shape}")

    print(f"\nFinal preprocessed data:")
    print(f"  Training: {X_train_flat.shape}")
    print(f"  Test: {X_test_flat.shape}")
    print(f"  Labels: {y_train.shape}, {y_test.shape}")
    print()

    return X_train_flat, y_train, X_test_flat, y_test, pca_model


def load_and_preprocess_data(use_sampling=None, use_pca=None):
    """
    Main function to load and preprocess CIFAR-10 data.

    Args:
        use_sampling: Whether to use sampling (None = use config default)
        use_pca: Whether to use PCA (None = use config default)

    Returns:
        tuple: (X_train, y_train, X_test, y_test, pca_model)
    """
    # Use config defaults if not specified
    if use_sampling is None:
        use_sampling = config.USE_SAMPLING
    if use_pca is None:
        use_pca = config.USE_PCA

    # Load data
    (X_train, y_train), (X_test, y_test) = load_cifar10_data()

    # Preprocess
    X_train, y_train, X_test, y_test, pca_model = preprocess_data(
        X_train, y_train, X_test, y_test,
        use_sampling=use_sampling,
        use_pca=use_pca
    )

    return X_train, y_train, X_test, y_test, pca_model
