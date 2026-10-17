"""
Diabetes Prediction Model Training
Dataset: PIMA Indians Diabetes Dataset
Model: SVM with linear kernel
Preprocessing: StandardScaler
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "diabetes.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

def train():
    print("=" * 50)
    print("Training Diabetes Prediction Model (SVM)")
    print("=" * 50)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Class distribution:\n{df['Outcome'].value_counts()}")

    X = df.drop(["Outcome", "SkinThickness", "DiabetesPedigreeFunction"], axis=1).values
    y = df["Outcome"].values

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test_scaled),
                                target_names=["Not Diabetic", "Diabetic"]))

    # Save artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    print(f"\n[OK] model.pkl  saved -> {ARTIFACTS_DIR}")
    print(f"[OK] scaler.pkl saved -> {ARTIFACTS_DIR}")

    # Quick sanity check
    sample = np.array([[5, 166, 72, 175, 25.8, 51]])
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)
    prob = model.predict_proba(sample_scaled)[0]
    print(f"\nSanity check (5, 166, 72, 175, 25.8, 51):")
    print(f"  Prediction : {'Diabetic' if pred[0] == 1 else 'Not Diabetic'}")
    print(f"  Confidence : {max(prob) * 100:.1f}%  (expected: Diabetic)")

if __name__ == "__main__":
    train()
