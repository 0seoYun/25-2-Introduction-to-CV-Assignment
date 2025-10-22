"""
Performance Metrics Module

This module provides functions to calculate and display various performance metrics.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1_score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def print_metrics(y_true, y_pred, task_name):
    """
    Calculate and print performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        task_name: Name of the task for display

    Returns:
        dict: Dictionary containing performance metrics
    """
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n{'=' * config.SEPARATOR_LENGTH}")
    print(f"{task_name} - Performance Metrics")
    print(f"{'=' * config.SEPARATOR_LENGTH}")
    print(f"Accuracy:  {metrics['accuracy']:.{config.METRIC_DECIMALS}f}")
    print(f"Precision: {metrics['precision']:.{config.METRIC_DECIMALS}f} (macro average)")
    print(f"Recall:    {metrics['recall']:.{config.METRIC_DECIMALS}f} (macro average)")
    print(f"F1-Score:  {metrics['f1_score']:.{config.METRIC_DECIMALS}f} (macro average)")
    print(f"{'=' * config.SEPARATOR_LENGTH}\n")

    return metrics


def format_metric(value, decimals=None):
    """
    Format a metric value to specified decimal places.

    Args:
        value: Metric value to format
        decimals: Number of decimal places (None = use config default)

    Returns:
        str: Formatted string
    """
    if decimals is None:
        decimals = config.METRIC_DECIMALS

    return f"{value:.{decimals}f}"


def print_cv_summary(results, k_values):
    """
    Print a summary table of cross-validation results.

    Args:
        results: Dictionary of CV results per k value
        k_values: List of k values
    """
    print("\n" + "=" * config.SEPARATOR_LENGTH)
    print(f"{config.N_FOLDS}-Fold Cross-Validation Results Summary")
    print("=" * config.SEPARATOR_LENGTH)
    print(f"{'k':>3} | {'Accuracy':^20} | {'Precision':^20} | {'Recall':^20} | {'F1-Score':^20}")
    print("-" * 100)

    for k in k_values:
        r = results[k]
        print(f"{k:3d} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
              f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f} | "
              f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f} | "
              f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}")

    print("=" * 100)
