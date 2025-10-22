"""
Visualization Module

This module provides functions for creating plots and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import config


def plot_cv_results(cv_results, save_path=None):
    """
    Visualize Cross-Validation results.

    Args:
        cv_results: Dictionary of CV results per k value
        save_path: Path to save the plot (None = use config default)
    """
    if save_path is None:
        save_path = config.OUTPUT_PLOT_PATH

    print(f"\n{'=' * config.SEPARATOR_LENGTH}")
    print("Generating Cross-Validation results plot...")
    print(f"{'=' * config.SEPARATOR_LENGTH}")

    k_values = sorted(cv_results.keys())
    accuracies = [cv_results[k]['accuracy_mean'] for k in k_values]
    std_devs = [cv_results[k]['accuracy_std'] for k in k_values]

    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.errorbar(k_values, accuracies, yerr=std_devs,
                 marker='o', linestyle='-', linewidth=2, markersize=8,
                 capsize=5, capthick=2, label='Accuracy with Error Bars')

    plt.xlabel('k (Number of Neighbors)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('5-Fold Cross-Validation: Accuracy vs. k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(k_values)

    # Mark the best performing k value
    best_k = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies)
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    plt.plot(best_k, best_acc, 'r*', markersize=15, label=f'Best: k={best_k}, acc={best_acc:.4f}')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close()


def plot_pca_variance(pca_model, save_path='result/pca_variance.png'):
    """
    Plot cumulative explained variance ratio from PCA.

    Args:
        pca_model: Fitted PCA model
        save_path: Path to save the plot
    """
    if pca_model is None:
        print("No PCA model provided, skipping variance plot.")
        return

    print(f"\nGenerating PCA variance plot...")

    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))

    # Plot individual variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Individual Explained Variance')
    plt.grid(True, alpha=0.3)

    # Plot cumulative variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    print(f"PCA variance plot saved: {save_path}")
    plt.close()
