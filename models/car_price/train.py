"""
High-Accuracy Car Price Prediction Model Training
===================================================
Trained on Indian Pre-Owned Cars Marketplace Dataset (6,019 rows).

Features:
  - GradientBoostingRegressor with TransformedTargetRegressor(log1p)
  - Vehicle Model Series and Brand resolution
  - Engine density and annual mileage intensity engineering
  - Precomputed specs lookup for zero-effort UX
"""

import re
import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "used_cars_india.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CURRENT_YEAR = datetime.date.today().year


def extract_number(value):
    if pd.isna(value):
        return np.nan
    val_str = str(value).strip().lower()
    if val_str in ["null bhp", "null", "nan", ""]:
        return np.nan
    match = re.search(r"[-+]?\d*\.?\d+", val_str)
    return float(match.group()) if match else np.nan


def clean_data(df):
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    df["Mileage"] = df["Mileage"].apply(extract_number)
    df.loc[df["Mileage"] <= 0, "Mileage"] = np.nan
    df["Engine"] = df["Engine"].apply(extract_number)
    df["Power"] = df["Power"].apply(extract_number)

    # Correct extreme data entry typo
    df.loc[df["Kilometers_Driven"] > 1000000, "Kilometers_Driven"] = 65000

    df["Brand"] = df["Name"].str.split().str[0]
    df["Model_Series"] = df["Name"].str.split().str.slice(0, 2).str.join(" ")
    df["Car_Age"] = np.maximum(0, CURRENT_YEAR - df["Year"])

    df["Kms_Per_Year"] = df["Kilometers_Driven"] / np.maximum(1, df["Car_Age"])
    df["Power_Per_Engine"] = df["Power"] / np.maximum(1, df["Engine"])

    df = df.dropna(subset=["Price"])
    df = df[df["Price"] > 0]
    return df


def generate_specs_lookup(df):
    """Generate lookup dictionary of standard specs per model series for instant UI auto-fill."""
    specs = {}
    grouped = df.groupby("Model_Series")

    for model_name, grp in grouped:
        mileage = float(grp["Mileage"].median()) if not grp["Mileage"].dropna().empty else 18.0
        engine = float(grp["Engine"].median()) if not grp["Engine"].dropna().empty else 1400.0
        power = float(grp["Power"].median()) if not grp["Power"].dropna().empty else 90.0
        seats = float(grp["Seats"].median()) if not grp["Seats"].dropna().empty else 5.0
        fuel = str(grp["Fuel_Type"].mode().iloc[0]) if not grp["Fuel_Type"].dropna().empty else "Petrol"
        transmission = str(grp["Transmission"].mode().iloc[0]) if not grp["Transmission"].dropna().empty else "Manual"

        specs[model_name] = {
            "mileage": round(mileage, 2),
            "engine": int(engine),
            "power": round(power, 1),
            "seats": int(seats),
            "default_fuel": fuel,
            "default_transmission": transmission,
            "sample_count": int(len(grp))
        }

    return specs


def train():
    print("=" * 65)
    print("Training High-Accuracy Car Price Valuation Engine (Gradient Boosting)")
    print("=" * 65)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Raw dataset rows: {len(df):,}")

    df = clean_data(df)
    print(f"Cleaned valid records: {len(df):,}")

    specs_lookup = generate_specs_lookup(df)

    companies = sorted(df["Brand"].unique().tolist())
    fuel_types = sorted(df["Fuel_Type"].unique().tolist())
    transmissions = sorted(df["Transmission"].unique().tolist())
    locations = sorted(df["Location"].unique().tolist())
    owner_types = ["First", "Second", "Third", "Fourth & Above"]
    car_names = sorted(df["Model_Series"].unique().tolist())

    models_by_company = {}
    for comp in companies:
        models = sorted(df[df["Brand"] == comp]["Model_Series"].unique().tolist())
        models_by_company[comp] = models

    owner_order = [["Fourth & Above", "Third", "Second", "First"]]
    num_cols = ["Kilometers_Driven", "Mileage", "Engine", "Power", "Seats", "Car_Age", "Kms_Per_Year", "Power_Per_Engine"]
    cat_cols = ["Location", "Fuel_Type", "Transmission", "Brand", "Model_Series"]
    ord_cols = ["Owner_Type"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
        ("ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(categories=owner_order)),
        ]), ord_cols),
    ])

    gbr = GradientBoostingRegressor(
        n_estimators=650,
        learning_rate=0.038,
        max_depth=5,
        subsample=0.85,
        random_state=42
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(regressor=gbr, func=np.log1p, inverse_func=np.expm1))
    ])

    feature_cols = num_cols + cat_cols + ord_cols
    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training pipeline across {len(X_train):,} listings...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    print("\nBenchmark Evaluation:")
    print(f"  R² Score  : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAE       : {mae:.3f} Lakh INR (~Rs. {mae*100000:,.0f})")
    print(f"  RMSE      : {rmse:.3f} Lakh INR (~Rs. {rmse*100000:,.0f})")

    # Save artifacts
    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(companies, os.path.join(ARTIFACTS_DIR, "companies.pkl"))
    joblib.dump(fuel_types, os.path.join(ARTIFACTS_DIR, "fuel_types.pkl"))
    joblib.dump(car_names, os.path.join(ARTIFACTS_DIR, "car_names.pkl"))

    options_data = {
        "companies": companies,
        "fuel_types": fuel_types,
        "transmissions": transmissions,
        "locations": locations,
        "owner_types": owner_types,
        "car_names": car_names,
        "models_by_company": models_by_company,
        "specs_lookup": specs_lookup,
        "global_defaults": {
            "mileage": float(df["Mileage"].median()),
            "engine": float(df["Engine"].median()),
            "power": float(df["Power"].median()),
            "seats": int(df["Seats"].median()),
            "location": "Mumbai",
            "transmission": "Manual",
            "owner_type": "First"
        }
    }
    joblib.dump(options_data, os.path.join(ARTIFACTS_DIR, "options.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "options.json"), "w") as f:
        json.dump(options_data, f, indent=2)

    metadata = {
        "algorithm": "Gradient Boosting Regressor (log1p TransformedTarget)",
        "dataset": "Indian Pre-Owned Cars Marketplace",
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "r2_score": round(r2 * 100, 2),
        "mae_lakhs": round(mae, 3),
        "mae_inr": int(mae * 100000),
        "rmse_lakhs": round(rmse, 3),
        "features": len(feature_cols),
        "current_year": CURRENT_YEAR
    }
    with open(os.path.join(ARTIFACTS_DIR, "car_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll artifacts successfully saved to {ARTIFACTS_DIR}")
    return pipeline, metadata


if __name__ == "__main__":
    train()
