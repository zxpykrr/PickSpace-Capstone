# PickSpace

**A decision-support web app for buying a house or condo in Khon Kaen, Thailand.**
It estimates a fair price with machine learning, shows whether a listing is cheap or expensive compared with similar properties, and recommends areas that match what the buyer cares about (price, schools, hospitals, flood risk, and more).

**Live system:** https://dss-project-fronend.vercel.app/

> Senior project, College of Computing, Khon Kaen University (Business Information Technology). Team of 2.
>
> This repository documents the project. The application code is maintained in the team's project repositories, so the folder names below describe that codebase.

---

## What it does

| Feature | What the user gets |
|---|---|
| **Search** | Browse new-build projects and second-hand listings with filters and a detail page for each unit |
| **Price prediction** | Enter a property's features and get today's estimated market price, a 1-year forecast, and areas with higher prices |
| **Price check** | For any listing in the system, see whether its price is *low / fair / high* compared with similar properties, using the model's MAPE as the tolerance band |
| **Recommendation** | A content-based recommender that matches a buyer's needs (budget, bedrooms, nearby facilities, distance) to listings, scored with a location index |
| **Location analysis** | Per-project analysis built from distances to facilities, flood risk and elevation |
| **Accounts** | Register / login (JWT, Google sign-in), password reset by email, profile and saved reports |
| **Admin** | Manage users, projects and images, view usage, and retrain the models from the dashboard |

## Results

Price model (regression), evaluated on **119 held-out new-build units** that the model never saw (split by project, not by row):

| Model | MAE (THB) | MAPE | R² |
|---|---:|---:|---:|
| **Gradient Boosting (selected)** | **496,828** | **13.27%** | **0.861** |
| Voting Ensemble | 509,790 | 13.42% | 0.851 |
| Random Forest | 527,430 | 13.86% | 0.844 |
| LightGBM | 542,422 | 14.05% | 0.825 |
| CatBoost | 556,680 | 14.57% | 0.836 |

Error by market (repeated project-based splits): **new-build 15.5% MAPE**, **second-hand 27.2% MAPE**. Both are reported so the second-hand estimates are not oversold.
Overfitting check: train R² 0.98 vs test R² 0.86.

Recommender (v10): 1,256 items, checked against 25 buyer personas (constraint satisfaction 100%, diversity@5 = 4.44).

## How the price model was built

1. **Data**: 1,058 listings = 628 new-build units + 430 second-hand listings. 959 rows came from the original project data; a further ~1,100 second-hand listings were scraped from NaYoo and cleaned (duplicates, missing values, outliers) down to 99 usable rows.
2. **Location features** (QGIS and open data): distance to malls, hospitals, schools, markets, roads and more; elevation; flood-risk distance, with flood points collected from local news; official land appraisal rates by sub-district requested from the Treasury Department; a location score from a **GIS expert survey** (AHP weights, TOPSIS ranking).
3. **Feature selection**: from 38 candidates down to **14 features** with cross-validation, then checked with permutation importance.
4. **Leakage control**: train/test split by project (`GroupShuffleSplit`), tuning with `GroupKFold`, and second-hand listings from the same project as any test unit are removed from training, so the score reflects unseen projects.
5. **Model selection**: 5 algorithms tuned with `RandomizedSearchCV`; the winner is chosen by MAE in baht because it is easy to explain.
6. **Pricing without price history**: current price comes from comparable listings in the system; the 1-year forecast = ML prediction × a growth-rate formula.

## Architecture

```
Next.js 16 / React 19 / Tailwind 4   (Vercel)
        │  REST + JWT
        ▼
Express 5 on Node 22                 (Render, Docker)
        │  spawns / calls
        ▼
Flask ML service (scikit-learn, LightGBM, CatBoost)
        │
        ▼
PostgreSQL
```

- `DSS_Project_Fronend/` – Next.js app (search, prediction, recommendation, analysis, profile, admin)
- `DSS_Project_Backend-/` – Express API (`routes/`, `controllers/`, `services/`) plus the ML code
  - `ml1/` – recommender (content-based + AHP / TOPSIS location score + evaluation scripts)
  - `ml2/` – price model (`trainmodel.py`, data preparation, scrapers, `metrics_report.json`)
  - `ml_service.py` – Flask service that loads both models once and serves predictions

## Run it locally (from the application codebase)

**Requirements:** Node 22+, Python 3.13, PostgreSQL.

```bash
# Backend
cd DSS_Project_Backend-
npm install
pip install -r requirements.txt
cp .env.example .env        # fill in the values below
npm start                   # http://localhost:3001 (also starts the Flask ML service)

# Frontend
cd DSS_Project_Fronend
npm install
cp .env.example .env.local  # set NEXT_PUBLIC_API_URL and NEXT_PUBLIC_GOOGLE_CLIENT_ID
npm run dev                 # http://localhost:3000
```

Backend environment variables: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `JWT_SECRET`, `FRONTEND_URL`, `GOOGLE_CLIENT_ID`, and `EMAILJS_*` for password-reset emails.

To retrain the price model: `python ml2/trainmodel.py` (writes `ml2/model.pkl` and `ml2/artifacts/metrics_report.json`).

## Tech stack

Python · scikit-learn · LightGBM · CatBoost · Flask · Node.js · Express · PostgreSQL · Next.js · React · Tailwind CSS · QGIS · Docker

## Team

- **Priyakorn Phaladon** – data cleaning and scraping, QGIS location features, flood and appraisal data, GIS expert survey and location scoring, model selection for price prediction and recommendation, pricing logic (including no-history case), and the front-end and back-end for prediction, recommendation, search and profile.
- **Teammate** – the other half of the two-person team; this README describes my own contribution.

## Limitations

- Second-hand estimates are less accurate (27% MAPE) than new-build estimates (15%) because there is less and noisier data.
- Prices are **estimates for decision support**, not appraisals.
- Coverage is Khon Kaen only.
