# Quad Analytics — Applied Intelligence Suite

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-purple)

A production-ready, full-stack Machine Learning platform that seamlessly integrates four distinct ML models into a single, cohesive user experience. The application uses a high-performance **FastAPI** backend with memory pre-loading to serve predictions to a custom, responsive, privacy-first frontend.

![Quad Analytics](frontend/assets/logo.jpg)

> **Note:** Trained model `.pkl` artifacts are excluded from this repository due to file size constraints.  
> Run the training scripts once after cloning (see **How to Run** below) to regenerate them locally.

---


## Key Features

*   **Explainable AI (XAI):** Models do not just return a prediction. The API calculates feature attributions, and the frontend dynamically renders them via **Chart.js** to show users *exactly* which inputs influenced the decision (positive vs. negative impact).
*   **Zero Cold-Start Latency:** The FastAPI backend utilizes `lifespan` architecture to pre-load all `.pkl` machine learning artifacts (models, scalers, encoders) into server RAM at startup, ensuring near-instantaneous prediction responses.
*   **Strict Schema Validation:** Utilizes **Pydantic** for rigorous data validation. It features alias mapping to seamlessly convert readable JSON payloads from the frontend into the precise `snake_case` arrays expected by the scikit-learn models.
*   **Premium "Lavender Fields" UI/UX:** A completely custom, zero-dependency CSS design system mimicking high-end SaaS platforms. Includes automatic Light/Dark mode toggling (persisted via `localStorage`) and fully responsive grid layouts.
*   **Privacy-First Design:** Completely stateless inference. Zero user tracking, no session cookies, and no database logs of user inputs — ideal for handling sensitive mock healthcare data.
*   **Integrated EDA Explorers:** Model detail pages feature embedded Exploratory Data Analysis (EDA) visualizations, and a dedicated dataset explorer allows users to inspect the underlying training data schemas.

---

## Machine Learning Models

The platform demonstrates versatility by handling varying domains and prediction types.

### 1. Clinical Diagnostics (Classification)

| Model | Algorithm | Dataset | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Breast Cancer Risk | Random Forest (100 Trees) | Cancer1500 Multi-Risk Cohort | 83.0% | 0.891 |
| Diabetes Prediction | Support Vector Classifier (Linear) | PIMA Indians Diabetes (NIH/NIDDK) | 70.1% | 0.824 |

### 2. Market & Sports Valuation (Regression)

| Model | Algorithm | Dataset | R² | MAE |
|---|---|---|---|---|
| Vehicle Valuation | Gradient Boosting + Log Target Transform | Indian Pre-Owned Cars (6,019 rows) | 90.56% | INR 1.27 Lakh |
| IPL Score Forecaster | Extra Trees Regressor + Feature Engineering | IPL Ball-by-Ball 2008–Present (108,610 rows) | 71.57% | +/- 9.75 Runs |

---

## Model Version History

### Vehicle Valuation (Car Price)

| Version | Dataset | Algorithm | R² | MAE |
|---|---|---|---|---|
| v1 | Quikr India (892 rows) | Random Forest | 88.0% | ~INR 1.54 Lakh |
| **v2 (current)** | Indian Pre-Owned Cars (6,019 rows) | **Gradient Boosting + Log Target** | **90.56%** | **INR 1.27 Lakh** |

**v2 Improvements:**
- Migrated to a 6.7x larger, cleaner dataset.
- Implemented `TransformedTargetRegressor` with `log1p` scaling for long-tailed price distributions.
- Added feature engineering: `Model_Series` resolution, `Kms_Per_Year`, `Power_Per_Engine`.
- Added dynamic Transmission and Location inputs to the UI.
- Added precomputed spec lookup so the form auto-fills powertrain specs on car model selection.

---

### IPL Score Forecaster

| Version | Dataset | Algorithm | Features | R² | MAE |
|---|---|---|---|---|---|
| v1 | IPL Ball-by-Ball (76,014 rows) | Random Forest | 7 | ~93.5%* | +/- 12.2 Runs |
| v2 | IPL Ball-by-Ball 2008–Present (108,610 rows) | Random Forest | 7 | 72.15% | +/- 10.34 Runs |
| **v3 (current)** | IPL Ball-by-Ball 2008–Present (108,610 rows) | **Extra Trees Regressor** | **29** | **71.57%** | **+/- 9.75 Runs** |

*v1 R² was inflated due to a smaller, less varied test split.

**v3 Improvements:**
- Switched from `RandomForestRegressor` to `ExtraTreesRegressor` (faster training, lower MAE on rolling-window cricket data).
- Wrote a full data preprocessing pipeline (`scratch/preprocess_ipl.py`) to convert raw ball-by-ball logs into 30-ball rolling window state features.
- Added 8 engineered features based on feature importance analysis:
  - `balls_bowled`, `balls_remaining` — over-state context
  - `wickets_in_hand` — batting resources remaining
  - `run_rate` — current scoring rate
  - `proj_score` — simple linear score projection (run_rate × 20)
  - `rrr_last5` — recent run rate (runs_last_5 / 5 overs)
  - `momentum` — composite momentum index (runs_last_5 - wickets_last_5 × 10)
  - `boundary_pressure` — pressure index (runs_last_5 × wickets_in_hand / 10)
- `train.py` and `predictor.py` now share the exact same feature pipeline to eliminate training/inference drift.

---

## Technology Stack

**Frontend**
*   **HTML5 & CSS3:** Custom "Lavender Fields" CSS variables, CSS Grid/Flexbox layouts, and zero-emoji professional styling.
*   **JavaScript (Vanilla):** Handles async API requests, dynamic DOM updates, and custom form behaviors.
*   **Chart.js:** Renders dynamic EDA charts and real-time Explainable AI attribution waterfalls.

**Backend**
*   **FastAPI & Uvicorn:** High-performance, asynchronous REST API infrastructure.
*   **Pydantic:** Robust data typing and request validation.

**Machine Learning**
*   **Scikit-Learn:** Core algorithm training (Extra Trees, Gradient Boosting, Random Forests, SVC, Pipelines).
*   **Pandas & NumPy:** Data cleaning, feature engineering, and numerical transformations.
*   **Joblib:** Serialization of trained models and preprocessing artifacts (`.pkl`).

---

## Project Architecture

```text
ml-website/
├── run.py                 # Uvicorn entry point (starts server on port 8000)
├── app/
│   ├── main.py            # FastAPI configuration, static mounts, and lifespan loader
│   ├── routes.py          # API endpoints and Pydantic schemas
│   └── predictor.py       # ML inference logic and feature mapping
├── frontend/              # HTML/CSS/JS files served statically
│   ├── css/style.css      # Design system variables and layouts
│   ├── js/app.js          # API client, Chart.js renderers, UI logic
│   └── assets/            # Logos, favicons, etc.
├── models/                # Serialized .pkl artifacts grouped by model
│   ├── car_price/train.py
│   ├── ipl_score/train.py
│   ├── diabetes/train.py
│   └── breast_cancer/train.py
├── data/                  # CSV datasets (for the Data Explorer UI)
└── scratch/               # Data preprocessing scripts
    └── preprocess_ipl.py  # Converts raw ball-by-ball IPL data to rolling-window features
```

---

## How to Run

1.  **Clone the repository** (or navigate to the project directory).
2.  **Install dependencies:**
    ```bash
    pip install fastapi uvicorn scikit-learn pandas numpy pydantic joblib
    ```
3.  **Train models** (one-time, artifacts already committed):
    ```bash
    python models/car_price/train.py
    python models/ipl_score/train.py
    python models/diabetes/train.py
    python models/breast_cancer/train.py
    ```
4.  **Start the server:**
    ```bash
    python run.py
    ```
5.  **Access the platform:**
    Open your browser and navigate to `http://localhost:8000`

---

*Developed as a demonstration of end-to-end Machine Learning deployment, bridging the gap between raw data science and production software engineering.*
