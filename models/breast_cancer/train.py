"""
Breast Cancer Classification Model Training (Lifestyle/Genetics)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "The_Cancer_data_1500_V2.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

FEATURES = [
    "Age",
    "BMI",
    "Smoking",
    "GeneticRisk",
    "PhysicalActivity",
    "AlcoholIntake",
    "CancerHistory"
]

def train():
    print("=" * 60)
    print("Training Breast Cancer Classification Model (Lifestyle/Genetics)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Check Target
    if 'Diagnosis' in df.columns:
        X = df[FEATURES]
        Y = df['Diagnosis']
    else:
        raise ValueError("Diagnosis column not found in dataset")

    print(f"Target distribution:\n{Y.value_counts()}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_std, Y_train)

    train_acc = accuracy_score(Y_train, model.predict(X_train_std))
    test_acc  = accuracy_score(Y_test,  model.predict(X_test_std))
    print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing  Accuracy : {test_acc  * 100:.2f}%")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(FEATURES, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    print(f"\n[OK] model.pkl         saved -> {ARTIFACTS_DIR}")
    print(f"[OK] scaler.pkl        saved -> {ARTIFACTS_DIR}")
    print(f"[OK] feature_names.pkl saved -> {ARTIFACTS_DIR}")

    # Sanity check with a sample (row 0: Malignant/1)
    sample_df = df.iloc[[0]][FEATURES]
    input_std  = scaler.transform(sample_df)
    pred = model.predict(input_std)[0]
    prob = model.predict_proba(input_std)[0]
    print(f"\nSanity check (sample 0, label={Y.iloc[0]}):")
    print(f"  Predicted: {pred} ({'Negative' if pred == 0 else 'Positive'})")
    print(f"  Confidence: {max(prob) * 100:.1f}%")

if __name__ == "__main__":
    train()
