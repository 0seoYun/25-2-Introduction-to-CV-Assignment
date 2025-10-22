"""
Report Generation Module

This module generates markdown reports in Korean and English.
"""

import config


def generate_report(task1_metrics, task2_best_k, task2_metrics, task2_val_results,
                   cv_results, train_size, test_size, use_pca, language='ko'):
    """
    Generate a markdown report.

    Args:
        task1_metrics: Task 1 performance metrics
        task2_best_k: Task 2 best k value
        task2_metrics: Task 2 performance metrics
        task2_val_results: Task 2 validation results
        cv_results: Cross-validation results
        train_size: Training data size
        test_size: Test data size
        use_pca: Whether PCA was used
        language: 'ko' for Korean, 'en' for English

    Returns:
        str: Markdown content
    """
    if language == 'ko':
        return _generate_korean_report(
            task1_metrics, task2_best_k, task2_metrics, task2_val_results,
            cv_results, train_size, test_size, use_pca
        )
    else:
        return _generate_english_report(
            task1_metrics, task2_best_k, task2_metrics, task2_val_results,
            cv_results, train_size, test_size, use_pca
        )


def _generate_korean_report(task1_metrics, task2_best_k, task2_metrics, task2_val_results,
                            cv_results, train_size, test_size, use_pca):
    """Generate Korean markdown report."""
    content = []

    # Header
    content.append("# CIFAR-10 KNN 분류 과제 결과\n\n")

    # Note section
    content.append(f"**참고**: 이 실험은 빠른 실행을 위해 데이터의 일부만 사용했습니다.\n")
    content.append(f"- 학습 샘플: {train_size}개\n")
    content.append(f"- 테스트 샘플: {test_size}개\n")
    if use_pca:
        content.append(f"- PCA 적용: Yes (차원: 3072 → {config.PCA_N_COMPONENTS})\n\n")
    else:
        content.append("- PCA 적용: No\n\n")

    # Task 1
    content.append("## Task 1: Train/Test Split만 사용한 KNN 분류\n\n")
    content.append("### 설정\n")
    content.append(f"- k 값: {config.DEFAULT_K}\n")
    content.append(f"- 학습 데이터: {train_size}개 샘플\n")
    content.append(f"- 테스트 데이터: {test_size}개 샘플\n\n")
    content.append("### 성능 지표\n\n")
    content.append("| 지표 | 점수 |\n")
    content.append("|--------|-------|\n")
    content.append(f"| Accuracy | {task1_metrics['accuracy']:.4f} |\n")
    content.append(f"| Precision (macro) | {task1_metrics['precision']:.4f} |\n")
    content.append(f"| Recall (macro) | {task1_metrics['recall']:.4f} |\n")
    content.append(f"| F1-Score (macro) | {task1_metrics['f1_score']:.4f} |\n\n")

    # Task 2
    content.append("## Task 2: Train/Validation/Test Split을 사용한 하이퍼파라미터 튜닝\n\n")
    val_size = int(train_size * config.VALIDATION_SPLIT)
    train_tr_size = train_size - val_size
    content.append("### 설정\n")
    content.append(f"- 학습 데이터: {train_tr_size}개 샘플 ({int((1-config.VALIDATION_SPLIT)*100)}%)\n")
    content.append(f"- 검증 데이터: {val_size}개 샘플 ({int(config.VALIDATION_SPLIT*100)}%)\n")
    content.append(f"- 테스트 데이터: {test_size}개 샘플\n\n")
    content.append("### 검증 결과\n\n")
    content.append("| k | 검증 정확도 |\n")
    content.append("|---|---------------------|\n")
    for k, acc in task2_val_results:
        content.append(f"| {k} | {acc:.4f} |\n")
    content.append(f"\n**최적 k 값: {task2_best_k}**\n\n")
    content.append("### 최적 k로 테스트 세트 평가\n\n")
    content.append("| 지표 | 점수 |\n")
    content.append("|--------|-------|\n")
    content.append(f"| Accuracy | {task2_metrics['accuracy']:.4f} |\n")
    content.append(f"| Precision (macro) | {task2_metrics['precision']:.4f} |\n")
    content.append(f"| Recall (macro) | {task2_metrics['recall']:.4f} |\n")
    content.append(f"| F1-Score (macro) | {task2_metrics['f1_score']:.4f} |\n\n")

    # Task 3
    content.append("## Task 3: 5-Fold Cross-Validation\n\n")
    content.append("### 설정\n")
    content.append(f"- Fold 수: {config.N_FOLDS}\n")
    content.append(f"- Fold당 학습 데이터: {int(train_size * 0.8)}개 샘플\n")
    content.append(f"- Fold당 검증 데이터: {int(train_size * 0.2)}개 샘플\n\n")
    content.append("### Cross-Validation 결과\n\n")
    content.append("| k | Accuracy | Precision | Recall | F1-Score |\n")
    content.append("|---|----------|-----------|--------|----------|\n")

    k_values = sorted(cv_results.keys())
    for k in k_values:
        r = cv_results[k]
        content.append(f"| {k} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
                      f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f} | "
                      f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f} | "
                      f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} |\n")

    best_k = max(cv_results.keys(), key=lambda k: cv_results[k]['accuracy_mean'])
    content.append(f"\n**최고 성능 k 값: {best_k}** (Accuracy: {cv_results[best_k]['accuracy_mean']:.4f})\n\n")

    # Plot
    content.append("### Accuracy vs. k 그래프\n\n")
    content.append(f"![Accuracy vs k]({config.OUTPUT_PLOT_PATH})\n\n")

    # Analysis
    content.append("## 분석 및 결론\n\n")
    content.append("### 주요 발견사항\n\n")
    content.append(f"1. **최적 k 값**: Validation split 방법에서는 k={task2_best_k}가 최적이었고, "
                  f"5-fold CV에서는 k={best_k}이 최고 성능을 보였습니다.\n\n")
    content.append("2. **성능 비교**:\n")
    content.append(f"   - Task 1 (k={config.DEFAULT_K}): Accuracy = {task1_metrics['accuracy']:.4f}\n")
    content.append(f"   - Task 2 (k={task2_best_k}): Accuracy = {task2_metrics['accuracy']:.4f}\n")
    content.append(f"   - Task 3 (k={best_k}): Accuracy = {cv_results[best_k]['accuracy_mean']:.4f}\n\n")
    content.append("3. **Cross-Validation의 장점**: 5-fold CV는 모델의 안정성과 일반화 성능을 "
                  "더 신뢰성 있게 평가할 수 있습니다. Error bars는 각 k 값에 대한 성능의 "
                  "변동성을 보여줍니다.\n\n")

    if use_pca:
        content.append(f"4. **PCA의 효과**: PCA를 적용하여 차원을 3072에서 {config.PCA_N_COMPONENTS}으로 축소했습니다. "
                      "이를 통해 노이즈 제거와 계산 속도 향상 효과를 얻었습니다.\n\n")
        content.append("5. **k 값의 영향**: k 값이 너무 작으면 overfitting, 너무 크면 underfitting "
                      "경향이 있습니다. 그래프를 통해 최적의 균형점을 찾을 수 있습니다.\n\n")
    else:
        content.append("4. **k 값의 영향**: k 값이 너무 작으면 overfitting, 너무 크면 underfitting "
                      "경향이 있습니다. 그래프를 통해 최적의 균형점을 찾을 수 있습니다.\n\n")

    content.append("### CIFAR-10에서 KNN의 한계\n\n")
    content.append("KNN은 간단하고 직관적인 알고리즘이지만, CIFAR-10과 같은 복잡한 이미지 데이터셋에는 "
                  "다음과 같은 한계가 있습니다:\n\n")
    content.append("- 픽셀 기반 거리 측정은 의미론적 유사성을 잘 포착하지 못함\n")
    content.append("- 고차원 데이터에서 'curse of dimensionality' 문제\n")
    content.append("- CNN과 같은 딥러닝 모델에 비해 낮은 정확도\n")
    content.append("- 예측 시간이 느림 (모든 학습 데이터와 거리 계산 필요)\n\n")

    return ''.join(content)


def _generate_english_report(task1_metrics, task2_best_k, task2_metrics, task2_val_results,
                             cv_results, train_size, test_size, use_pca):
    """Generate English markdown report."""
    content = []

    # Header
    content.append("# CIFAR-10 KNN Classification Assignment Results\n\n")

    # Note section
    content.append(f"**Note**: This experiment used a subset of the data for faster execution.\n")
    content.append(f"- Training samples: {train_size}\n")
    content.append(f"- Test samples: {test_size}\n")
    if use_pca:
        content.append(f"- PCA applied: Yes (dimension: 3072 → {config.PCA_N_COMPONENTS})\n\n")
    else:
        content.append("- PCA applied: No\n\n")

    # Task 1
    content.append("## Task 1: KNN Classification with Train/Test Split Only\n\n")
    content.append("### Configuration\n")
    content.append(f"- k value: {config.DEFAULT_K}\n")
    content.append(f"- Train data: {train_size} samples\n")
    content.append(f"- Test data: {test_size} samples\n\n")
    content.append("### Performance Metrics\n\n")
    content.append("| Metric | Score |\n")
    content.append("|--------|-------|\n")
    content.append(f"| Accuracy | {task1_metrics['accuracy']:.4f} |\n")
    content.append(f"| Precision (macro) | {task1_metrics['precision']:.4f} |\n")
    content.append(f"| Recall (macro) | {task1_metrics['recall']:.4f} |\n")
    content.append(f"| F1-Score (macro) | {task1_metrics['f1_score']:.4f} |\n\n")

    # Task 2
    content.append("## Task 2: Hyperparameter Tuning with Train/Validation/Test Split\n\n")
    val_size = int(train_size * config.VALIDATION_SPLIT)
    train_tr_size = train_size - val_size
    content.append("### Configuration\n")
    content.append(f"- Train data: {train_tr_size} samples ({int((1-config.VALIDATION_SPLIT)*100)}%)\n")
    content.append(f"- Validation data: {val_size} samples ({int(config.VALIDATION_SPLIT*100)}%)\n")
    content.append(f"- Test data: {test_size} samples\n\n")
    content.append("### Validation Results\n\n")
    content.append("| k | Validation Accuracy |\n")
    content.append("|---|---------------------|\n")
    for k, acc in task2_val_results:
        content.append(f"| {k} | {acc:.4f} |\n")
    content.append(f"\n**Best k value: {task2_best_k}**\n\n")
    content.append("### Test Set Evaluation with Best k\n\n")
    content.append("| Metric | Score |\n")
    content.append("|--------|-------|\n")
    content.append(f"| Accuracy | {task2_metrics['accuracy']:.4f} |\n")
    content.append(f"| Precision (macro) | {task2_metrics['precision']:.4f} |\n")
    content.append(f"| Recall (macro) | {task2_metrics['recall']:.4f} |\n")
    content.append(f"| F1-Score (macro) | {task2_metrics['f1_score']:.4f} |\n\n")

    # Task 3
    content.append("## Task 3: 5-Fold Cross-Validation\n\n")
    content.append("### Configuration\n")
    content.append(f"- Number of folds: {config.N_FOLDS}\n")
    content.append(f"- Train data per fold: {int(train_size * 0.8)} samples\n")
    content.append(f"- Validation data per fold: {int(train_size * 0.2)} samples\n\n")
    content.append("### Cross-Validation Results\n\n")
    content.append("| k | Accuracy | Precision | Recall | F1-Score |\n")
    content.append("|---|----------|-----------|--------|----------|\n")

    k_values = sorted(cv_results.keys())
    for k in k_values:
        r = cv_results[k]
        content.append(f"| {k} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
                      f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f} | "
                      f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f} | "
                      f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} |\n")

    best_k = max(cv_results.keys(), key=lambda k: cv_results[k]['accuracy_mean'])
    content.append(f"\n**Best performing k value: {best_k}** (Accuracy: {cv_results[best_k]['accuracy_mean']:.4f})\n\n")

    # Plot
    content.append("### Accuracy vs. k Plot\n\n")
    content.append(f"![Accuracy vs k]({config.OUTPUT_PLOT_PATH})\n\n")

    # Analysis
    content.append("## Analysis and Conclusions\n\n")
    content.append("### Key Findings\n\n")
    content.append(f"1. **Optimal k value**: Using validation split, k={task2_best_k} was optimal, "
                  f"while 5-fold CV showed k={best_k} as the best performer.\n\n")
    content.append("2. **Performance Comparison**:\n")
    content.append(f"   - Task 1 (k={config.DEFAULT_K}): Accuracy = {task1_metrics['accuracy']:.4f}\n")
    content.append(f"   - Task 2 (k={task2_best_k}): Accuracy = {task2_metrics['accuracy']:.4f}\n")
    content.append(f"   - Task 3 (k={best_k}): Accuracy = {cv_results[best_k]['accuracy_mean']:.4f}\n\n")
    content.append("3. **Advantages of Cross-Validation**: 5-fold CV provides more reliable evaluation "
                  "of model stability and generalization performance. Error bars show the variability "
                  "in performance for each k value.\n\n")

    if use_pca:
        content.append(f"4. **Effect of PCA**: PCA was applied to reduce dimensionality from 3072 to {config.PCA_N_COMPONENTS}. "
                      "This helps with noise reduction and improved computation speed.\n\n")
        content.append("5. **Effect of k value**: Too small k values lead to overfitting, while too large "
                      "values lead to underfitting. The plot helps identify the optimal balance point.\n\n")
    else:
        content.append("4. **Effect of k value**: Too small k values lead to overfitting, while too large "
                      "values lead to underfitting. The plot helps identify the optimal balance point.\n\n")

    content.append("### Limitations of KNN on CIFAR-10\n\n")
    content.append("While KNN is simple and intuitive, it has limitations for complex image datasets like CIFAR-10:\n\n")
    content.append("- Pixel-based distance measures fail to capture semantic similarity\n")
    content.append("- Curse of dimensionality in high-dimensional data\n")
    content.append("- Lower accuracy compared to deep learning models like CNNs\n")
    content.append("- Slow prediction time (requires distance calculation with all training data)\n\n")

    return ''.join(content)


def save_report(content, save_path):
    """
    Save report content to a file.

    Args:
        content: Report content string
        save_path: Path to save the report
    """
    print(f"\n{'=' * config.SEPARATOR_LENGTH}")
    print(f"Generating results report: {save_path}")
    print(f"{'=' * config.SEPARATOR_LENGTH}")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Results report saved: {save_path}")


def generate_and_save_reports(task1_metrics, task2_best_k, task2_metrics, task2_val_results,
                              cv_results, train_size, test_size, use_pca):
    """
    Generate and save both Korean and English reports.

    Args:
        task1_metrics: Task 1 performance metrics
        task2_best_k: Task 2 best k value
        task2_metrics: Task 2 performance metrics
        task2_val_results: Task 2 validation results
        cv_results: Cross-validation results
        train_size: Training data size
        test_size: Test data size
        use_pca: Whether PCA was used
    """
    # Generate Korean report
    korean_content = generate_report(
        task1_metrics, task2_best_k, task2_metrics, task2_val_results,
        cv_results, train_size, test_size, use_pca, language='ko'
    )
    save_report(korean_content, config.OUTPUT_REPORT_KR_PATH)

    # Generate English report
    english_content = generate_report(
        task1_metrics, task2_best_k, task2_metrics, task2_val_results,
        cv_results, train_size, test_size, use_pca, language='en'
    )
    save_report(english_content, config.OUTPUT_REPORT_EN_PATH)
