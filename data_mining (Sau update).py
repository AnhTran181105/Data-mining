"""**Code sau khi sửa theo phần feedback**"""

# =====================================================
# ENN EXPERIMENT ON WAVEFORM-NOISE DATASET
# =====================================================
# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from imblearn.under_sampling import EditedNearestNeighbours
# =====================================================
# 2. LOAD DATASET
# =====================================================
file_path = "waveform-+noise.data"
df = pd.read_csv(
    file_path,
    header=None
)
print("========== DATASET HEAD ==========")
print(df.head())
print()
# =====================================================
# 3. DEFINE FEATURES AND LABELS
# =====================================================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print("========== CLASS DISTRIBUTION ==========")
print(y.value_counts().sort_index())
print()
# =====================================================
# 4. TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)
print("========== TRAIN / TEST SIZE ==========")
print("Train Size:", X_train.shape)
print("Test Size :", X_test.shape)
print()
# =====================================================
# 5. STANDARDIZATION
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# =====================================================
# 6. BASELINE MODEL (WITHOUT ENN)
# =====================================================
print("========================================")
print("BASELINE MODEL (WITHOUT ENN)")
print("========================================")
knn_baseline = KNeighborsClassifier(
    n_neighbors=5
)
knn_baseline.fit(
    X_train_scaled,
    y_train
)
y_pred_baseline = knn_baseline.predict(
    X_test_scaled
)
baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)
baseline_precision = precision_score(
    y_test,
    y_pred_baseline,
    average="weighted"
)
baseline_recall = recall_score(
    y_test,
    y_pred_baseline,
    average="weighted"
)
baseline_f1 = f1_score(
    y_test,
    y_pred_baseline,
    average="weighted"
)
baseline_macro_f1 = f1_score(
    y_test,
    y_pred_baseline,
    average="macro"
)
print(f"Accuracy  : {baseline_accuracy:.4f}")
print(f"Precision : {baseline_precision:.4f}")
print(f"Recall    : {baseline_recall:.4f}")
print(f"F1-score  : {baseline_f1:.4f}")
print(f"Macro F1  : {baseline_macro_f1:.4f}")
print()
print("========== CONFUSION MATRIX ==========")
print(confusion_matrix(
    y_test,
    y_pred_baseline
))
print()
print("========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred_baseline
))
print()
# =====================================================
# 7. ENN EXPERIMENTS
# =====================================================
enn_values = [3, 5, 7]
for k in enn_values:
    print("========================================")
    print(f"ENN EXPERIMENT - k = {k}")
    print("========================================")
    # =================================================
    # APPLY ENN ONLY ON TRAIN SET
    # =================================================
    enn = EditedNearestNeighbours(
        n_neighbors=k,
        sampling_strategy="all",
        kind_sel="all"
    )
    X_train_enn, y_train_enn = enn.fit_resample(
        X_train_scaled,
        y_train
    )
    # =================================================
    # SAMPLE ANALYSIS
    # =================================================
    before_samples = len(y_train)
    after_samples = len(y_train_enn)
    removed_samples = before_samples - after_samples
    print("========== SAMPLE ANALYSIS ==========")
    print("Before ENN :", before_samples)
    print("After ENN  :", after_samples)
    print("Removed    :", removed_samples)
    print()
    # =================================================
    # CLASS ANALYSIS
    # =================================================
    original_class = pd.Series(
        y_train
    ).value_counts().sort_index()
    enn_class = pd.Series(
        y_train_enn
    ).value_counts().sort_index()
    print("========== REMOVED SAMPLES BY CLASS ==========")
    for class_label in sorted(original_class.index):
        removed_count = (
            original_class[class_label]
            - enn_class.get(class_label, 0)
        )
        print(
            f"Removed Class {class_label}: "
            f"{removed_count}"
        )
    print()
    # =================================================
    # TRAIN KNN ON CLEANED DATA
    # =================================================
    knn_enn = KNeighborsClassifier(
        n_neighbors=5
    )
    knn_enn.fit(
        X_train_enn,
        y_train_enn
    )
    y_pred_enn = knn_enn.predict(
        X_test_scaled
    )
    # =================================================
    # EVALUATION
    # =================================================
    accuracy = accuracy_score(
        y_test,
        y_pred_enn
    )
    precision = precision_score(
        y_test,
        y_pred_enn,
        average="weighted"
    )
    recall = recall_score(
        y_test,
        y_pred_enn,
        average="weighted"
    )
    f1 = f1_score(
        y_test,
        y_pred_enn,
        average="weighted"
    )
    macro_f1 = f1_score(
        y_test,
        y_pred_enn,
        average="macro"
    )
    print("========== PERFORMANCE ==========")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"Macro F1  : {macro_f1:.4f}")
    print()
    # =================================================
    # CONFUSION MATRIX
    # =================================================
    print("========== CONFUSION MATRIX ==========")
    print(confusion_matrix(
        y_test,
        y_pred_enn
    ))
    print()
    # =================================================
    # CLASSIFICATION REPORT
    # =================================================
    print("========== CLASSIFICATION REPORT ==========")
    print(classification_report(
        y_test,
        y_pred_enn
    ))
    print()

# =====================================================
# 8. FINISH
# =====================================================
print("========================================")
print("EXPERIMENT FINISHED")
print("========================================")