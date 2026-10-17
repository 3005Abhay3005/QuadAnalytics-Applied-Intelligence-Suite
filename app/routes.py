import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.predictor import (
    predict_diabetes,
    predict_breast_cancer,
    predict_car_price,
    predict_ipl_score,
    get_car_options,
    get_ipl_teams,
    TEAMS,
)
from app.utils import success_response

router = APIRouter(prefix="/api")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
_DATASET_CACHE = {}


def _load_dataset_info(model_id: str):
    mapping = {
        "diabetes": ("diabetes.csv", "PIMA Indians Diabetes", "Clinical Metabolic"),
        "breast-cancer": ("The_Cancer_data_1500_V2.csv", "Cancer1500 Lifestyle & Genetics", "Oncology Risk"),
        "car-price": ("used_cars_india.csv", "Indian Pre-Owned Cars Marketplace", "Automotive Secondary Market"),
        "ipl-score": ("ipl_new.csv", "IPL Ball-by-Ball Deliveries", "Sports Quantitative Analytics"),
    }
    # normalize key
    norm_id = model_id.lower().replace("_", "-")
    if norm_id not in mapping:
        raise HTTPException(404, f"Dataset for '{model_id}' not found.")

    fname, title, domain = mapping[norm_id]
    if norm_id not in _DATASET_CACHE:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise HTTPException(404, f"Dataset file {fname} not found.")

        # Read sample or full
        df = pd.read_csv(fpath)
        columns = [{"name": col, "type": str(df[col].dtype)} for col in df.columns]
        summary = {}
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            summary[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 2),
                "median": round(float(df[col].median()), 2),
            }

        sample_df = df.head(50).fillna("")
        sample = sample_df.to_dict(orient="records")

        _DATASET_CACHE[norm_id] = {
            "id": norm_id,
            "title": title,
            "domain": domain,
            "filename": fname,
            "total_rows": len(df),
            "columns": columns,
            "summary": summary,
            "sample": sample,
        }
    return _DATASET_CACHE[norm_id]


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {
        "status": "operational",
        "service": "Quad Analytics Inference Engine",
        "engine": "FastAPI + scikit-learn",
        "version": "3.0.0",
        "privacy_policy": "Zero User Tracking — Ephemeral In-Memory Inference",
    }


# ---------------------------------------------------------------------------
# Stats / Model Benchmark Metadata (Zero User Tracking)
# ---------------------------------------------------------------------------

@router.get("/stats")
def stats():
    """Returns benchmark metadata for all 4 models (zero personal user data)."""
    return {
        "models": [
            {
                "id": "diabetes",
                "name": "Clinical Diabetes Screening Engine",
                "short_name": "Diabetes Screening",
                "algorithm": "Support Vector Classifier (Linear)",
                "domain": "Metabolic Endocrinology",
                "type": "Classification",
                "dataset": "PIMA Indians Diabetes (NIH/NIDDK)",
                "samples": 768,
                "accuracy": 70.1,
                "roc_auc": 0.824,
                "latency_ms": 2.4,
                "features": 6,
                "status": "Operational",
            },
            {
                "id": "breast-cancer",
                "name": "Oncology Risk Stratification Engine",
                "short_name": "Oncology Profiler",
                "algorithm": "Random Forest Ensemble (100 Trees)",
                "domain": "Preventive Oncology",
                "type": "Classification",
                "dataset": "Cancer1500 Multi-Risk Cohort",
                "samples": 1500,
                "accuracy": 83.0,
                "roc_auc": 0.891,
                "latency_ms": 3.1,
                "features": 7,
                "status": "Operational",
            },
            {
                "id": "car-price",
                "name": "Automotive Asset Valuation Engine",
                "short_name": "Vehicle Valuation",
                "algorithm": "Gradient Boosting Regressor (Log Target Transform)",
                "domain": "Secondary Market Valuation",
                "type": "Regression",
                "dataset": "Indian Pre-Owned Cars Marketplace",
                "samples": 6019,
                "r2_score": 90.56,
                "mae": "INR 127,169 (~1.27 Lakh)",
                "latency_ms": 3.8,
                "features": 14,
                "status": "Operational",
            },
            {
                "id": "ipl-score",
                "name": "Sports Momentum Forecaster",
                "short_name": "IPL Score Forecaster",
                "algorithm": "Extra Trees Regressor (Feature Engineered)",
                "domain": "Sports Quantitative Analytics",
                "type": "Regression",
                "dataset": "IPL Ball-by-Ball Deliveries (2008-Present)",
                "samples": 108610,
                "r2_score": 71.57,
                "mae": "+/- 9.75 Runs",
                "latency_ms": 4.1,
                "features": 29,
                "status": "Operational",
            },
        ],
        "telemetry": {
            "total_models": 4,
            "active_instances": 4,
            "avg_latency_ms": 3.35,
            "architecture": "In-Memory Singleton Registry",
            "retention_policy": "Zero User Logging — Ephemeral Stateless Compute",
        },
    }


# ---------------------------------------------------------------------------
# Dataset Explorer Endpoint
# ---------------------------------------------------------------------------

@router.get("/dataset/{model_id}")
def dataset_endpoint(model_id: str):
    """Streams genuine dataset sample records, schema, and column statistics."""
    try:
        return _load_dataset_info(model_id)
    except HTTPException as h:
        raise h
    except Exception as e:
        raise HTTPException(500, f"Error processing dataset: {str(e)}")


# ---------------------------------------------------------------------------
# Diabetes
# ---------------------------------------------------------------------------

class DiabetesInput(BaseModel):
    pregnancies: float = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose (mg/dL)")
    blood_pressure: float = Field(..., ge=0, le=200, description="Diastolic BP (mm Hg)")
    insulin: float = Field(..., ge=0, le=900, description="2-hr serum insulin (μU/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body Mass Index")
    age: float = Field(..., ge=18, le=100, description="Age (years)")


@router.post("/predict/diabetes")
def predict_diabetes_route(data: DiabetesInput):
    try:
        result = predict_diabetes(data.model_dump())
        return success_response(result, "Diabetes Prediction (SVM)")
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run: python models/diabetes/train.py")
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Breast Cancer
# ---------------------------------------------------------------------------

class BreastCancerInput(BaseModel):
    age: float = Field(..., alias="Age", description="Age")
    bmi: float = Field(..., alias="BMI", description="BMI")
    smoking: float = Field(..., alias="Smoking", description="Smoking (0/1)")
    genetic_risk: float = Field(..., alias="GeneticRisk", description="Genetic Risk (0/1/2)")
    physical_activity: float = Field(..., alias="PhysicalActivity", description="Physical Activity")
    alcohol_intake: float = Field(..., alias="AlcoholIntake", description="Alcohol Intake")
    cancer_history: float = Field(..., alias="CancerHistory", description="Cancer History (0/1)")

    class Config:
        populate_by_name = True


@router.post("/predict/breast-cancer")
def predict_bc_route(data: BreastCancerInput):
    try:
        raw = {
            "Age": data.age,
            "BMI": data.bmi,
            "Smoking": data.smoking,
            "GeneticRisk": data.genetic_risk,
            "PhysicalActivity": data.physical_activity,
            "AlcoholIntake": data.alcohol_intake,
            "CancerHistory": data.cancer_history,
        }
        result = predict_breast_cancer(raw)
        return success_response(result, "Breast Cancer Classification (RF)")
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run: python models/breast_cancer/train.py")
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Car Price
# ---------------------------------------------------------------------------

class CarPriceInput(BaseModel):
    name: str
    company: str
    year: int = Field(..., ge=1990, le=2026)
    kms_driven: int = Field(..., ge=0, le=1000000)
    fuel_type: str
    transmission: Optional[str] = None
    location: Optional[str] = None
    owner_type: Optional[str] = None
    mileage: Optional[float] = None
    engine: Optional[float] = None
    power: Optional[float] = None
    seats: Optional[float] = None


@router.post("/predict/car-price")
def predict_car_route(data: CarPriceInput):
    try:
        result = predict_car_price(data.model_dump(exclude_unset=False))
        return success_response(result, "Car Price Prediction (Gradient Boosting)")
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run: python models/car_price/train.py")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/options/car")
def car_options():
    try:
        return get_car_options()
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet.")


# ---------------------------------------------------------------------------
# IPL Score
# ---------------------------------------------------------------------------

class IPLInput(BaseModel):
    batting_team: str
    bowling_team: str
    runs: int = Field(..., ge=0, le=300)
    wickets: int = Field(..., ge=0, le=10)
    overs: float = Field(..., ge=5.0, le=20.0)
    runs_last_5: int = Field(..., ge=0, le=120)
    wickets_last_5: int = Field(..., ge=0, le=5)


@router.post("/predict/ipl-score")
def predict_ipl_route(data: IPLInput):
    if data.batting_team not in TEAMS:
        raise HTTPException(422, f"Invalid batting team. Choose from: {TEAMS}")
    if data.bowling_team not in TEAMS:
        raise HTTPException(422, f"Invalid bowling team. Choose from: {TEAMS}")
    if data.batting_team == data.bowling_team:
        raise HTTPException(422, "Batting and bowling teams must be different.")
    try:
        result = predict_ipl_score(data.model_dump())
        return success_response(result, "IPL Score Prediction (RF)")
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run: python models/ipl_score/train.py")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/options/ipl")
def ipl_teams():
    return {"teams": get_ipl_teams()}
