"""
KNN Model Training and Evaluation Module

This module contains functions for:
- Task 1: Simple train/test split
- Task 2: Validation-based hyperparameter tuning
- Task 3: Cross-validation
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config
from . import metrics


def task1_simple_train_test(X_train, y_train, X_test, y_test, k=None):
    """
    Task 1: KNN classification using simple Train/Test split

    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        k: Number of neighbors for KNN (None = use config default)

    Returns:
        dict: Performance metrics
    """
    if k is None:
        k = config.DEFAULT_K

    print("\n" + "=" * config.SEPARATOR_LENGTH)
    print("TASK 1: KNN Classification with Train/Test Split Only")
    print("=" * config.SEPARATOR_LENGTH)
    print(f"Using k value: {k}")

    # Create and train KNN classifier
    print("Training KNN model...")
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)

    # Make predictions
    print("Predicting on test data...")
    y_pred = knn.predict(X_test)

    # Print performance metrics
    task_metrics = metrics.print_metrics(y_test, y_pred, "Task 1")

    return task_metrics


def task2_validation_split(X_train, y_train, X_test, y_test, k_values=None):
    """
    Task 2: Hyperparameter tuning using Train/Validation/Test split

    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        k_values: List of k values to test (None = use config default)

    Returns:
        tuple: (best k value, performance metrics, validation results per k)
    """
    if k_values is None:
        k_values = config.K_VALUES_VALIDATION

    print("\n" + "=" * config.SEPARATOR_LENGTH)
    print("TASK 2: Hyperparameter Tuning with Train/Validation/Test Split")
    print("=" * config.SEPARATOR_LENGTH)

    # Split train into train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=y_train
    )

    print(f"Train data: {X_tr.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    print(f"\nTesting k values: {k_values}")

    # Calculate validation accuracy for each k value
    val_accuracies = []
    print("\nEvaluating each k value on validation set...")

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_tr, y_tr)
        y_val_pred = knn.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_accuracies.append(val_acc)
        print(f"k={k:2d}: Validation Accuracy = {val_acc:.4f}")

    # Select the best k value
    best_k = k_values[np.argmax(val_accuracies)]
    best_val_acc = max(val_accuracies)

    print(f"\nBest k value: {best_k} (Validation Accuracy: {best_val_acc:.4f})")

    # Retrain with best k using full training data
    print(f"\nRetraining with best k={best_k} using full training data...")
    knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn_best.fit(X_train, y_train)

    # Final evaluation on test set
    y_pred = knn_best.predict(X_test)
    task_metrics = metrics.print_metrics(y_test, y_pred, "Task 2 (Best k)")

    return best_k, task_metrics, list(zip(k_values, val_accuracies))


def task3_cross_validation(X_train, y_train, k_values=None, n_folds=None):
    """
    Task 3: Model evaluation using 5-fold Cross-Validation

    Args:
        X_train: Training data
        y_train: Training labels
        k_values: List of k values to test (None = use config default)
        n_folds: Number of folds (None = use config default)

    Returns:
        dict: Mean performance metrics and standard deviation per k value
    """
    if k_values is None:
        k_values = config.K_VALUES_CV
    if n_folds is None:
        n_folds = config.N_FOLDS

    print("\n" + "=" * config.SEPARATOR_LENGTH)
    print(f"TASK 3: {n_folds}-Fold Cross-Validation")
    print("=" * config.SEPARATOR_LENGTH)
    print(f"Testing k values: {k_values}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    results = {}

    for k in k_values:
        print(f"\nEvaluating k={k}...")

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            # Train and predict with KNN model
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            knn.fit(X_tr, y_tr)
            y_pred = knn.predict(X_val)

            # Calculate performance metrics
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average='macro')
            rec = recall_score(y_val, y_pred, average='macro')
            f1 = f1_score(y_val, y_pred, average='macro')

            fold_accuracies.append(acc)
            fold_precisions.append(prec)
            fold_recalls.append(rec)
            fold_f1s.append(f1)

            if config.SHOW_PROGRESS:
                print(f"  Fold {fold_idx}: Accuracy = {acc:.4f}")

        # Calculate mean and standard deviation
        results[k] = {
            'accuracy_mean': np.mean(fold_accuracies),
            'accuracy_std': np.std(fold_accuracies),
            'precision_mean': np.mean(fold_precisions),
            'precision_std': np.std(fold_precisions),
            'recall_mean': np.mean(fold_recalls),
            'recall_std': np.std(fold_recalls),
            'f1_mean': np.mean(fold_f1s),
            'f1_std': np.std(fold_f1s)
        }

        print(f"  Mean Accuracy: {results[k]['accuracy_mean']:.4f} ± {results[k]['accuracy_std']:.4f}")

    # Print results summary
    metrics.print_cv_summary(results, k_values)

    # Find the best k value
    best_k = max(results.keys(), key=lambda k: results[k]['accuracy_mean'])
    print(f"\nBest performing k value: {best_k} (Accuracy: {results[best_k]['accuracy_mean']:.4f})")

    return results
