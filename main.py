"""
Main execution script for KNN CIFAR-10 Classification

This script orchestrates the entire workflow:
1. Load and preprocess data (with optional PCA)
2. Run Task 1: Simple train/test split
3. Run Task 2: Validation-based hyperparameter tuning
4. Run Task 3: Cross-validation
5. Generate visualizations
6. Create reports
"""

import warnings
warnings.filterwarnings('ignore')

import config
from src import data_loader, knn_models, visualization, report_generator


def print_header():
    """Print the header banner."""
    print("\n" + "=" * config.SEPARATOR_LENGTH)
    print(" " * 15 + "CIFAR-10 KNN Classification")
    print("=" * config.SEPARATOR_LENGTH + "\n")


def print_configuration():
    """Print current configuration."""
    print("=" * config.SEPARATOR_LENGTH)
    print("Configuration:")
    print("=" * config.SEPARATOR_LENGTH)
    print(f"Data Sampling: {config.USE_SAMPLING}")
    if config.USE_SAMPLING:
        print(f"  - Train samples per class: {config.SAMPLES_PER_CLASS_TRAIN}")
        print(f"  - Test samples per class: {config.SAMPLES_PER_CLASS_TEST}")
    print(f"PCA: {config.USE_PCA}")
    if config.USE_PCA:
        print(f"  - Components: {config.PCA_N_COMPONENTS}")
        print(f"  - Whiten: {config.PCA_WHITEN}")
    print(f"K-values for validation: {config.K_VALUES_VALIDATION}")
    print(f"K-values for CV: {config.K_VALUES_CV}")
    print(f"Cross-validation folds: {config.N_FOLDS}")
    print("=" * config.SEPARATOR_LENGTH + "\n")


def print_summary(train_size, test_size):
    """Print completion summary."""
    print("\n" + "=" * config.SEPARATOR_LENGTH)
    print(" " * 20 + "All tasks completed!")
    print("=" * config.SEPARATOR_LENGTH)
    print(f"\nDataset sizes:")
    print(f"  - Training: {train_size} samples")
    print(f"  - Test: {test_size} samples")
    print(f"\nGenerated files:")
    print(f"  - {config.OUTPUT_PLOT_PATH}: Cross-validation results plot")
    print(f"  - {config.OUTPUT_REPORT_KR_PATH}: Korean results report")
    print(f"  - {config.OUTPUT_REPORT_EN_PATH}: English results report")
    if config.USE_PCA:
        print("  - result/pca_variance.png: PCA variance plot")
    print("=" * config.SEPARATOR_LENGTH + "\n")


def main():
    """
    Main execution function.
    """
    # Print header
    print_header()

    # Print configuration
    print_configuration()

    # Step 1: Load and preprocess data
    X_train, y_train, X_test, y_test, pca_model = data_loader.load_and_preprocess_data()

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    # Step 2: Task 1 - Simple Train/Test Split
    task1_metrics = knn_models.task1_simple_train_test(
        X_train, y_train, X_test, y_test
    )

    # Step 3: Task 2 - Train/Validation/Test Split
    task2_best_k, task2_metrics, task2_val_results = knn_models.task2_validation_split(
        X_train, y_train, X_test, y_test
    )

    # Step 4: Task 3 - 5-Fold Cross-Validation
    cv_results = knn_models.task3_cross_validation(X_train, y_train)

    # Step 5: Visualize results
    visualization.plot_cv_results(cv_results)

    # Optional: Plot PCA variance if PCA was used
    if config.USE_PCA and pca_model is not None:
        visualization.plot_pca_variance(pca_model)

    # Step 6: Generate reports (both Korean and English)
    report_generator.generate_and_save_reports(
        task1_metrics, task2_best_k, task2_metrics,
        task2_val_results, cv_results,
        train_size, test_size, config.USE_PCA
    )

    # Print summary
    print_summary(train_size, test_size)


if __name__ == "__main__":
    main()
