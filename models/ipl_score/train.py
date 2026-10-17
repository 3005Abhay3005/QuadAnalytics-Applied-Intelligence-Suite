"""
IPL 1st Innings Score Prediction Model Training
Dataset: ipl_new.csv
Model: ExtraTreesRegressor with engineered features
MAE: ~9.75 runs | R²: ~0.716
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import time

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "..", "data", "ipl_new.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

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


def load_and_engineer(path):
    df = pd.read_csv(path)
    df = df[(df['bat_team'].isin(TEAMS)) & (df['bowl_team'].isin(TEAMS))]
    df = df.rename(columns={"bat_team": "batting_team", "bowl_team": "bowling_team"})

    # Core derived features
    df['balls_bowled']      = (df['overs'].astype(int) * 6
                               + ((df['overs'] % 1) * 10).astype(int)).clip(1, 120)
    df['balls_remaining']   = (120 - df['balls_bowled']).clip(0, 120)
    df['wickets_in_hand']   = 10 - df['wickets']
    df['run_rate']          = df['runs'] / (df['balls_bowled'] / 6).replace(0, 1)
    df['proj_score']        = df['run_rate'] * 20
    df['rrr_last5']         = df['runs_last_5'] / 5
    df['momentum']          = df['runs_last_5'] - df['wickets_last_5'] * 10
    df['boundary_pressure'] = df['runs_last_5'] * df['wickets_in_hand'] / 10.0

    # One-hot team encodings
    for team in TEAMS:
        df[f"bat_{team}"]  = (df["batting_team"] == team).astype(int)
        df[f"bowl_{team}"] = (df["bowling_team"] == team).astype(int)

    df = df[df['overs'] >= 5.0]
    return df


def get_feature_cols():
    return (
        [f"bat_{t}"  for t in TEAMS]
        + [f"bowl_{t}" for t in TEAMS]
        + ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5",
           "balls_bowled", "balls_remaining", "wickets_in_hand",
           "run_rate", "proj_score", "rrr_last5", "momentum", "boundary_pressure"]
    )


def train():
    print("=" * 65)
    print("Training IPL Score Predictor (ExtraTrees + Feature Engineering)")
    print("=" * 65)

    df = load_and_engineer(DATA_PATH)
    print(f"Dataset: {df.shape[0]} rows  |  Features: {len(get_feature_cols())}")

    feature_cols = get_feature_cols()
    X = df[feature_cols].values
    y = df["total"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    t0 = time.time()
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    print("Training model...")
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2  = r2_score(y_test,  model.predict(X_test))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Training R²  : {train_r2:.4f}")
    print(f"Testing R²   : {test_r2:.4f}")
    print(f"Test MAE     : {test_mae:.2f} runs")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model,        os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(TEAMS,        os.path.join(ARTIFACTS_DIR, "teams.pkl"))
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_cols.pkl"))
    print(f"\n[OK] Artifacts saved -> {ARTIFACTS_DIR}")

    # Sanity checks
    print("\nSanity checks:")
    checks = [
        ("Delhi Daredevils",           "Chennai Super Kings",   68,  3, 10.2, 29, 1),
        ("Mumbai Indians",             "Kolkata Knight Riders", 100, 1, 12.0, 45, 0),
        ("Royal Challengers Bangalore","Sunrisers Hyderabad",   30,  5, 8.0,  12, 3),
    ]
    for bt, bwl, r, w, ov, rl5, wl5 in checks:
        balls_bowled     = int(ov) * 6 + int(round((ov % 1) * 10))
        balls_bowled     = max(1, balls_bowled)
        balls_remaining  = max(0, 120 - balls_bowled)
        wickets_in_hand  = 10 - w
        run_rate         = r / (balls_bowled / 6.0)
        proj_score       = run_rate * 20
        rrr_last5        = rl5 / 5.0
        momentum         = rl5 - wl5 * 10
        boundary_pressure = rl5 * wickets_in_hand / 10.0
        row = {}
        for team in TEAMS:
            row[f"bat_{team}"]  = 1 if bt == team else 0
            row[f"bowl_{team}"] = 1 if bwl == team else 0
        row.update({"runs": r, "wickets": w, "overs": ov, "runs_last_5": rl5,
                    "wickets_last_5": wl5, "balls_bowled": balls_bowled,
                    "balls_remaining": balls_remaining, "wickets_in_hand": wickets_in_hand,
                    "run_rate": run_rate, "proj_score": proj_score, "rrr_last5": rrr_last5,
                    "momentum": momentum, "boundary_pressure": boundary_pressure})
        X_s = np.array([[row[c] for c in feature_cols]])
        pred = int(round(model.predict(X_s)[0]))
        print(f"  {bt} vs {bwl}: {r}/{w} in {ov} ov -> Predicted={pred}")


if __name__ == "__main__":
    train()
