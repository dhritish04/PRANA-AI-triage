import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("real_labeled_ecg_dataset.csv")

print("=" * 50)
print("REAL ECG DATASET")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nLabel Distribution:")
print(df["Label"].value_counts())

# =====================================================
# Features and Labels
# =====================================================

X = df.drop("Label", axis=1)
y = df["Label"]

# =====================================================
# Random Forest Model
# =====================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# =====================================================
# 5-Fold Cross Validation
# =====================================================

print("\n" + "=" * 50)
print("5-FOLD CROSS VALIDATION")
print("=" * 50)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nFold Accuracies:")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print("\nAverage Accuracy:")
print(f"{cv_scores.mean():.4f}")

print("\nStandard Deviation:")
print(f"{cv_scores.std():.4f}")

# =====================================================
# Train/Test Split Evaluation
# =====================================================

print("\n" + "=" * 50)
print("TRAIN-TEST EVALUATION")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nTest Accuracy:")
print(f"{accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions
))

# =====================================================
# Feature Importance
# =====================================================

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

importances = model.feature_importances_

for feature, importance in sorted(
    zip(X.columns, importances),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature:<10} : {importance:.4f}")