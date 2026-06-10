import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    "real_labeled_ecg_dataset.csv"
)

print("Dataset Shape:")
print(df.shape)

print("\nLabel Distribution:")
print(df["Label"].value_counts())

X = df.drop(
    "Label",
    axis=1
)

y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

joblib.dump(
    model,
    "real_ecg_model.pkl"
)

print("Real ECG model saved!")

predictions = model.predict(
    X_test
)

print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nFeature Importance:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        feature,
        ":",
        round(importance,4)
    )