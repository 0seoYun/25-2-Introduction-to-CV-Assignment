# CIFAR-10 KNN Classification

Dongguk University, 2025-2, Introduction to Computer Vision Assignment

This project implements K-Nearest Neighbors (KNN) classification on the CIFAR-10 dataset with PCA dimensionality reduction.

## Project Structure

```
Assignment/
├── config.py                    # Central configuration file
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and PCA preprocessing
│   ├── metrics.py              # Performance metrics calculation
│   ├── knn_models.py           # KNN model implementations (Task 1, 2, 3)
│   ├── visualization.py        # Plotting and visualization
│   └── report_generator.py     # Report generation (Korean & English)
├── result/                      # Generated output files (created after first run)
│   ├── cv_accuracy_vs_k.png    # Cross-validation accuracy plot
│   ├── pca_variance.png        # PCA variance plot
│   ├── knn_result.md           # Korean results report
│   └── knn_result_eng.md       # English results report
└── README.md                   # This file
```

## Features

- **Task 1**: Simple train/test split KNN classification
- **Task 2**: Hyperparameter tuning with train/validation/test split
- **Task 3**: 5-fold cross-validation with multiple k values
- **PCA**: Dimensionality reduction from 3072 to 200 dimensions
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Accuracy vs k plots with error bars
- **Reports**: Automatic generation of Korean and English reports

## Requirements

- Python 3.8 or higher
- Virtual environment (recommended)

## Installation and Execution

### Step 1: Clone the Repository

```bash
cd /path/to/your/workspace
git clone <repository-url>
cd Assignment
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```cmd
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (data processing)
- scikit-learn (KNN model and metrics)
- tensorflow (CIFAR-10 dataset loading)
- matplotlib (visualization)

### Step 5: Run the Program

```bash
python main.py
```

The program will:
1. Download CIFAR-10 dataset automatically (first run only)
2. Apply PCA for dimensionality reduction
3. Execute all three tasks (train/test, validation, cross-validation)
4. Generate plots and reports

**Execution time**: Approximately 3-5 minutes on a standard laptop

## Configuration

You can modify the parameters in `config.py`:

### Data Sampling
```python
USE_SAMPLING = True
SAMPLES_PER_CLASS_TRAIN = 500  # 500 samples per class (5000 total)
SAMPLES_PER_CLASS_TEST = 100   # 100 samples per class (1000 total)
```

### PCA Settings
```python
USE_PCA = True
PCA_N_COMPONENTS = 200  # Number of principal components
PCA_WHITEN = False      # Whitening (set to False for better KNN performance)
```

### K Values
```python
K_VALUES_VALIDATION = [1, 3, 5, 7, 9, 11, 13, 15]
K_VALUES_CV = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
N_FOLDS = 5  # Number of folds for cross-validation
```

## Output Files

After execution, the following files will be generated in the `result/` directory:

1. **result/cv_accuracy_vs_k.png**: Cross-validation accuracy plot with error bars
2. **result/pca_variance.png**: PCA explained variance plot
3. **result/knn_result.md**: Korean results report
4. **result/knn_result_eng.md**: English results report