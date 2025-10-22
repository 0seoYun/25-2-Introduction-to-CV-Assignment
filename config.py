"""
Configuration file for KNN CIFAR-10 Classification

This file centralizes all configuration parameters for the project.
"""

# ==================== Data Sampling Parameters ====================
# Set to True to use a subset of data for faster execution
USE_SAMPLING = True

# Number of samples per class for training (total = SAMPLES_PER_CLASS_TRAIN * 10)
SAMPLES_PER_CLASS_TRAIN = 500  # Default: 500 (total: 5000)

# Number of samples per class for testing (total = SAMPLES_PER_CLASS_TEST * 10)
SAMPLES_PER_CLASS_TEST = 100   # Default: 100 (total: 1000)


# ==================== PCA Parameters ====================
# Set to True to apply PCA for dimensionality reduction
USE_PCA = True

# Number of principal components to keep
# Original dimension: 3072 (32x32x3)
# Recommended: 200-300 for good balance between performance and speed
PCA_N_COMPONENTS = 200

# Whiten the PCA components (recommended for better performance)
PCA_WHITEN = False


# ==================== KNN Parameters ====================
# K values to test in validation and cross-validation
K_VALUES_VALIDATION = [1, 3, 5, 7, 9, 11, 13, 15]
K_VALUES_CV = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

# Default k value for Task 1
DEFAULT_K = 5

# Number of folds for cross-validation
N_FOLDS = 5

# Validation split ratio (for Task 2)
VALIDATION_SPLIT = 0.15  # 15% for validation


# ==================== Output Parameters ====================
# Output file paths
OUTPUT_PLOT_PATH = 'result/cv_accuracy_vs_k.png'
OUTPUT_REPORT_KR_PATH = 'result/knn_result.md'
OUTPUT_REPORT_EN_PATH = 'result/knn_result_eng.md'

# Plot parameters
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 6)


# ==================== CIFAR-10 Class Names ====================
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


# ==================== Random Seed ====================
RANDOM_SEED = 42


# ==================== Display Settings ====================
# Number of decimal places for metrics display
METRIC_DECIMALS = 4

# Progress bar settings
SHOW_PROGRESS = True
SEPARATOR_LENGTH = 70
