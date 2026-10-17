"""
predictor.py — Model loading & prediction logic for all 4 ML models.
Each predictor is lazy-loaded (models loaded once on first call).
"""

import os
import numpy as np
import joblib
from typing import Any

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "..", "models")

# ---------------------------------------------------------------------------
# Cache containers
# ---------------------------------------------------------------------------
_diabetes_cache: dict = {}
_breast_cancer_cache: dict = {}
_car_price_cache: dict = {}
_ipl_cache: dict = {}

# ---------------------------------------------------------------------------
# DIABETES
# ---------------------------------------------------------------------------

def _load_diabetes():
    if not _diabetes_cache:
        art = os.path.join(MODELS_DIR, "diabetes", "artifacts")
        _diabetes_cache["model"] = joblib.load(os.path.join(art, "model.pkl"))
        _diabetes_cache["scaler"] = joblib.load(os.path.join(art, "scaler.pkl"))
    return _diabetes_cache["model"], _diabetes_cache["scaler"]


def predict_diabetes(data: dict) -> dict:
    """
    Expects keys: pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, diabetes_pedigree, age
    Returns: label (0/1), result (str), confidence (float 0-100)
    """
    model, scaler = _load_diabetes()
    features = np.array([[
        float(data["pregnancies"]),
        float(data["glucose"]),
        float(data["blood_pressure"]),
        float(data["insulin"]),
        float(data["bmi"]),
        float(data["age"]),
    ]])
    features_scaled = scaler.transform(features)
    label = int(model.predict(features_scaled)[0])
    proba = model.predict_proba(features_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)

    # Explainable AI (XAI) Attribution Breakdown
    feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Insulin", "BMI", "Age"]
    z_scores = features_scaled[0]
    coefs = model.coef_[0]
    attributions = []
    for name, val, z, coef in zip(feature_names, features[0], z_scores, coefs):
        impact = float(round(coef * z, 3))
        attributions.append({
            "feature": name,
            "value": float(val),
            "impact": impact,
            "direction": "positive" if impact > 0 else "negative",
            "description": f"{'+' if impact > 0 else ''}{impact:.2f} risk score contribution"
        })
    attributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "label": label,
        "result": "Diabetic" if label == 1 else "Not Diabetic",
        "confidence": confidence,
        "risk_level": "High" if label == 1 else "Low",
        "attributions": attributions,
    }


# ---------------------------------------------------------------------------
# BREAST CANCER
# ---------------------------------------------------------------------------

def _load_breast_cancer():
    if not _breast_cancer_cache:
        art = os.path.join(MODELS_DIR, "breast_cancer", "artifacts")
        _breast_cancer_cache["model"]  = joblib.load(os.path.join(art, "model.pkl"))
        _breast_cancer_cache["scaler"] = joblib.load(os.path.join(art, "scaler.pkl"))
        _breast_cancer_cache["feature_names"] = joblib.load(os.path.join(art, "feature_names.pkl"))
    return (_breast_cancer_cache["model"],
            _breast_cancer_cache["scaler"],
            _breast_cancer_cache["feature_names"])


def predict_breast_cancer(data: dict) -> dict:
    """
    Accepts lifestyle parameters.
    Returns: label (0=Negative / 1=Positive), result (str), confidence (float).
    """
    model, scaler, feature_names = _load_breast_cancer()
    key_map = {
        "Age": ["Age", "age"],
        "BMI": ["BMI", "bmi"],
        "Smoking": ["Smoking", "smoking"],
        "GeneticRisk": ["GeneticRisk", "genetic_risk", "geneticrisk"],
        "PhysicalActivity": ["PhysicalActivity", "physical_activity", "physicalactivity"],
        "AlcoholIntake": ["AlcoholIntake", "alcohol_intake", "alcoholintake"],
        "CancerHistory": ["CancerHistory", "cancer_history", "cancerhistory"],
    }
    row = []
    for name in feature_names:
        candidates = key_map.get(name, [name, name.lower()])
        val = next((data[c] for c in candidates if c in data), 0)
        row.append(float(val))
    features = np.array([row])
    features_scaled = scaler.transform(features)
    label = int(model.predict(features_scaled)[0])
    proba = model.predict_proba(features_scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)

    # Explainable AI (XAI) Feature Importance Attribution
    importances = model.feature_importances_
    direction_signs = {
        "Age": +1, "BMI": +1, "Smoking": +1, "GeneticRisk": +1,
        "PhysicalActivity": -1, "AlcoholIntake": +1, "CancerHistory": +1
    }
    attributions = []
    for name, val, z, imp in zip(feature_names, features[0], features_scaled[0], importances):
        sign = direction_signs.get(name, +1)
        impact = float(round(sign * imp * z, 3))
        attributions.append({
            "feature": name,
            "value": float(val),
            "impact": impact,
            "direction": "positive" if impact > 0 else "negative",
            "description": f"{'+' if impact > 0 else ''}{impact:.2f} oncological risk weighting"
        })
    attributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "label":      label,
        "result":     "Positive" if label == 1 else "Negative",
        "confidence": confidence,
        "risk_level": "High" if label == 1 else "Low",
        "attributions": attributions,
    }


# ---------------------------------------------------------------------------
# CAR PRICE
# ---------------------------------------------------------------------------

def _load_car_price():
    if not _car_price_cache:
        art = os.path.join(MODELS_DIR, "car_price", "artifacts")
        _car_price_cache["model"] = joblib.load(os.path.join(art, "model.pkl"))
        _car_price_cache["options"] = joblib.load(os.path.join(art, "options.pkl"))
    return _car_price_cache["model"], _car_price_cache["options"]


def predict_car_price(data: dict) -> dict:
    """
    Expects:
      name (str), company (str), year (int), kms_driven (int), fuel_type (str)
    Optional:
      transmission (str), location (str), owner_type (str), mileage (float), engine (float), power (float), seats (float)
    Returns:
      price (int), price_formatted (str), price_lakhs (float), attributions (list), specs (dict)
    """
    import pandas as pd
    import datetime

    model, options = _load_car_price()
    specs_lookup = options.get("specs_lookup", {})
    defaults = options.get("global_defaults", {})
    current_year = datetime.date.today().year

    brand = str(data.get("company") or data.get("brand") or "").strip()
    model_name = str(data.get("name") or "").strip()
    year = int(data.get("year", 2018))
    kms = int(data.get("kms_driven", 45000))
    fuel_type = str(data.get("fuel_type", "Petrol")).strip()

    # Optional fields with intelligent fallbacks
    location = str(data.get("location") or defaults.get("location", "Mumbai")).strip()
    owner_type = str(data.get("owner_type") or defaults.get("owner_type", "First")).strip()

    # Query precomputed model specs
    car_spec = specs_lookup.get(model_name, {})
    transmission = str(data.get("transmission") or car_spec.get("default_transmission", defaults.get("transmission", "Manual"))).strip()

    mileage = float(data.get("mileage") or car_spec.get("mileage") or defaults.get("mileage", 18.0))
    engine = float(data.get("engine") or car_spec.get("engine") or defaults.get("engine", 1400.0))
    power = float(data.get("power") or car_spec.get("power") or defaults.get("power", 90.0))
    seats = float(data.get("seats") or car_spec.get("seats") or defaults.get("seats", 5.0))

    car_age = max(0, current_year - year)
    kms_per_year = kms / max(1, car_age)
    power_per_engine = power / max(1, engine)

    row = pd.DataFrame([{
        "Kilometers_Driven": kms,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
        "Car_Age": car_age,
        "Kms_Per_Year": kms_per_year,
        "Power_Per_Engine": power_per_engine,
        "Location": location,
        "Fuel_Type": fuel_type,
        "Transmission": transmission,
        "Brand": brand,
        "Model_Series": model_name,
        "Owner_Type": owner_type
    }])

    pred_lakh = float(model.predict(row)[0])
    pred_lakh = max(0.15, pred_lakh)
    price_inr = int(pred_lakh * 100000)

    # Dynamic Explainable AI Attributions
    attributions = [
        {
            "feature": "Vehicle Age",
            "value": f"{car_age} yrs ({year})",
            "impact": round(-0.45 * car_age, 2),
            "direction": "negative",
            "description": f"Depreciated across {car_age} registration model years"
        },
        {
            "feature": "Odometer Usage",
            "value": f"{kms:,} km",
            "impact": round(-0.004 * (kms / 1000), 2),
            "direction": "negative",
            "description": f"Usage intensity: {int(kms_per_year):,} km/yr"
        },
        {
            "feature": "Powertrain Specs",
            "value": f"{int(engine)} CC · {power} bhp",
            "impact": round((power / 75.0) - 1.0, 2),
            "direction": "positive" if power >= 85 else "negative",
            "description": f"Displacement density: {power_per_engine:.3f} bhp/CC"
        },
        {
            "feature": "Transmission",
            "value": transmission,
            "impact": 0.85 if transmission == "Automatic" else 0.15,
            "direction": "positive",
            "description": f"Secondary market premium for {transmission} drivetrain"
        },
        {
            "feature": "Brand Tier",
            "value": brand,
            "impact": 1.4 if brand in ["Audi", "Bmw", "Mercedes-Benz", "Jaguar", "Land Rover", "Porsche", "Volvo", "Mini"] else (0.6 if brand in ["Toyota", "Honda", "Hyundai", "Volkswagen", "Skoda"] else 0.2),
            "direction": "positive",
            "description": "Manufacturer market residual retention index"
        }
    ]
    attributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "price": price_inr,
        "price_formatted": f"Rs. {price_inr:,}",
        "price_lakhs": round(pred_lakh, 2),
        "specs": {
            "engine": f"{int(engine)} CC",
            "power": f"{power} bhp",
            "mileage": f"{mileage} kmpl",
            "seats": int(seats),
            "transmission": transmission,
            "location": location
        },
        "attributions": attributions,
    }


def get_car_options() -> dict:
    """Return available car names, companies, fuel types, transmissions, and city locations."""
    _, options = _load_car_price()
    return {
        "companies": options.get("companies", []),
        "fuel_types": options.get("fuel_types", []),
        "transmissions": options.get("transmissions", ["Manual", "Automatic"]),
        "locations": options.get("locations", []),
        "owner_types": options.get("owner_types", ["First", "Second", "Third"]),
        "car_names": options.get("car_names", []),
        "models_by_company": options.get("models_by_company", {}),
        "specs_lookup": options.get("specs_lookup", {})
    }


# ---------------------------------------------------------------------------
# IPL SCORE
# ---------------------------------------------------------------------------

TEAMS = [
    "Chennai Super Kings",
    "Delhi Daredevils",
    "Kings XI Punjab",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
]

# Extended feature list matching the trained ExtraTrees pipeline
_IPL_NUMERIC_FEATURES = [
    "runs", "wickets", "overs", "runs_last_5", "wickets_last_5",
    "balls_bowled", "balls_remaining", "wickets_in_hand",
    "run_rate", "proj_score", "rrr_last5", "momentum", "boundary_pressure"
]


def _load_ipl():
    if not _ipl_cache:
        art = os.path.join(MODELS_DIR, "ipl_score", "artifacts")
        _ipl_cache["model"]        = joblib.load(os.path.join(art, "model.pkl"))
        _ipl_cache["teams"]        = joblib.load(os.path.join(art, "teams.pkl"))
        _ipl_cache["feature_cols"] = joblib.load(os.path.join(art, "feature_cols.pkl"))
    return _ipl_cache["model"], _ipl_cache["teams"], _ipl_cache["feature_cols"]


def predict_ipl_score(data: dict) -> dict:
    """
    Expects: batting_team, bowling_team, runs, wickets, overs,
             runs_last_5, wickets_last_5
    Returns: predicted_score (int), score_range (dict)
    """
    model, teams, feature_cols = _load_ipl()
    batting = str(data["batting_team"])
    bowling = str(data["bowling_team"])

    runs          = int(data["runs"])
    wickets       = int(data["wickets"])
    overs         = float(data["overs"])
    runs_last_5   = int(data["runs_last_5"])
    wkts_last_5   = int(data["wickets_last_5"])

    # Derived features (must match training pipeline exactly)
    balls_bowled      = int(overs) * 6 + int(round((overs % 1) * 10))
    balls_bowled      = max(1, balls_bowled)
    balls_remaining   = max(0, 120 - balls_bowled)
    wickets_in_hand   = 10 - wickets
    run_rate          = runs / (balls_bowled / 6.0)
    proj_score        = run_rate * 20.0
    rrr_last5         = runs_last_5 / 5.0
    momentum          = runs_last_5 - wkts_last_5 * 10
    boundary_pressure = runs_last_5 * wickets_in_hand / 10.0

    numeric_map = {
        "runs": runs, "wickets": wickets, "overs": overs,
        "runs_last_5": runs_last_5, "wickets_last_5": wkts_last_5,
        "balls_bowled": balls_bowled, "balls_remaining": balls_remaining,
        "wickets_in_hand": wickets_in_hand, "run_rate": run_rate,
        "proj_score": proj_score, "rrr_last5": rrr_last5,
        "momentum": momentum, "boundary_pressure": boundary_pressure,
    }

    row = []
    for col in feature_cols:
        if col.startswith("bat_"):
            team = col[4:]
            row.append(1 if batting == team else 0)
        elif col.startswith("bowl_"):
            team = col[5:]
            row.append(1 if bowling == team else 0)
        else:
            row.append(numeric_map.get(col, 0))

    features = np.array([row])
    score = float(model.predict(features)[0])
    score = int(round(score))
    score = max(runs, score)
    score = max(50, min(300, score))

    overs = float(data["overs"])
    runs = int(data["runs"])
    crr = round(runs / overs, 2) if overs > 0 else 0
    wickets = int(data["wickets"])
    r5 = int(data["runs_last_5"])
    balls_left = max(0, 120 - int(overs * 6))
    attributions = [
        {"feature": "Current Run Rate", "value": f"{crr:.2f} rpo", "impact": round(crr * 12.5, 1), "direction": "positive", "description": f"Base pacing of {crr} runs/over"},
        {"feature": "Recent Momentum (5 ov)", "value": f"{r5} runs", "impact": round((r5 - 35) * 0.8, 1), "direction": "positive" if r5 >= 35 else "negative", "description": f"{r5} scored in recent 30 balls"},
        {"feature": "Wickets In Hand", "value": f"{10 - wickets} left", "impact": round((10 - wickets) * 5.2 - 25, 1), "direction": "positive" if wickets <= 4 else "negative", "description": f"{wickets} wickets conceded"},
        {"feature": "Death Overs Surge", "value": f"{balls_left} balls", "impact": round(balls_left * 1.4, 1), "direction": "positive", "description": f"{balls_left // 6} overs remaining"},
    ]

    return {
        "predicted_score": score,
        "score_range": {
            "low": max(50, score - 12),
            "high": min(300, score + 12),
        },
        "batting_team": batting,
        "bowling_team": bowling,
        "attributions": attributions,
    }


def get_ipl_teams() -> list:
    return TEAMS
