# CIFAR-10 KNN Classification Assignment Results

**Note**: This experiment used a subset of the data for faster execution.
- Training samples: 5000
- Test samples: 1000
- PCA applied: Yes (dimension: 3072 → 200)

## Task 1: KNN Classification with Train/Test Split Only

### Configuration
- k value: 5
- Train data: 5000 samples
- Test data: 1000 samples

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.2780 |
| Precision (macro) | 0.3457 |
| Recall (macro) | 0.2780 |
| F1-Score (macro) | 0.2668 |

## Task 2: Hyperparameter Tuning with Train/Validation/Test Split

### Configuration
- Train data: 4250 samples (85%)
- Validation data: 750 samples (15%)
- Test data: 1000 samples

### Validation Results

| k | Validation Accuracy |
|---|---------------------|
| 1 | 0.2760 |
| 3 | 0.2680 |
| 5 | 0.2733 |
| 7 | 0.2800 |
| 9 | 0.2613 |
| 11 | 0.2573 |
| 13 | 0.2533 |
| 15 | 0.2627 |

**Best k value: 7**

### Test Set Evaluation with Best k

| Metric | Score |
|--------|-------|
| Accuracy | 0.2750 |
| Precision (macro) | 0.3605 |
| Recall (macro) | 0.2750 |
| F1-Score (macro) | 0.2609 |

## Task 3: 5-Fold Cross-Validation

### Configuration
- Number of folds: 5
- Train data per fold: 4000 samples
- Validation data per fold: 1000 samples

### Cross-Validation Results

| k | Accuracy | Precision | Recall | F1-Score |
|---|----------|-----------|--------|----------|
| 1 | 0.2730 ± 0.0177 | 0.3099 ± 0.0153 | 0.2737 ± 0.0161 | 0.2645 ± 0.0156 |
| 3 | 0.2616 ± 0.0086 | 0.3379 ± 0.0133 | 0.2621 ± 0.0065 | 0.2464 ± 0.0074 |
| 5 | 0.2694 ± 0.0110 | 0.3436 ± 0.0135 | 0.2689 ± 0.0087 | 0.2516 ± 0.0096 |
| 7 | 0.2714 ± 0.0093 | 0.3563 ± 0.0096 | 0.2714 ± 0.0070 | 0.2492 ± 0.0098 |
| 9 | 0.2780 ± 0.0145 | 0.3563 ± 0.0125 | 0.2782 ± 0.0110 | 0.2523 ± 0.0126 |
| 11 | 0.2802 ± 0.0167 | 0.3699 ± 0.0168 | 0.2805 ± 0.0128 | 0.2540 ± 0.0138 |
| 13 | 0.2812 ± 0.0186 | 0.3655 ± 0.0243 | 0.2811 ± 0.0148 | 0.2547 ± 0.0191 |
| 15 | 0.2780 ± 0.0229 | 0.3603 ± 0.0294 | 0.2783 ± 0.0192 | 0.2493 ± 0.0209 |
| 17 | 0.2782 ± 0.0208 | 0.3680 ± 0.0325 | 0.2790 ± 0.0160 | 0.2487 ± 0.0199 |
| 19 | 0.2786 ± 0.0239 | 0.3692 ± 0.0280 | 0.2791 ± 0.0185 | 0.2471 ± 0.0220 |
| 21 | 0.2766 ± 0.0213 | 0.3793 ± 0.0256 | 0.2773 ± 0.0152 | 0.2439 ± 0.0181 |

**Best performing k value: 13** (Accuracy: 0.2812)

### Accuracy vs. k Plot

![Accuracy vs k](result/cv_accuracy_vs_k.png)

## Analysis and Conclusions

### Key Findings

1. **Optimal k value**: Using validation split, k=7 was optimal, while 5-fold CV showed k=13 as the best performer.

2. **Performance Comparison**:
   - Task 1 (k=5): Accuracy = 0.2780
   - Task 2 (k=7): Accuracy = 0.2750
   - Task 3 (k=13): Accuracy = 0.2812

3. **Advantages of Cross-Validation**: 5-fold CV provides more reliable evaluation of model stability and generalization performance. Error bars show the variability in performance for each k value.

4. **Effect of PCA**: PCA was applied to reduce dimensionality from 3072 to 200. This helps with noise reduction and improved computation speed.

5. **Effect of k value**: Too small k values lead to overfitting, while too large values lead to underfitting. The plot helps identify the optimal balance point.

### Limitations of KNN on CIFAR-10

While KNN is simple and intuitive, it has limitations for complex image datasets like CIFAR-10:

- Pixel-based distance measures fail to capture semantic similarity
- Curse of dimensionality in high-dimensional data
- Lower accuracy compared to deep learning models like CNNs
- Slow prediction time (requires distance calculation with all training data)

