# UNG Storage and Supply Research

Research application for daily UNG price features and rolling one-to-five-session forecasts. The initial focus is EIA storage tightness and natural-gas supply context. Forecasts are experimental and are not trade recommendations.

## Model status

The experimental forecast set is CatBoost plus Bayesian dynamic regression with dynamic variable selection. CatBoost fits nonlinear boosted trees. The Bayesian model uses a sequential spike-and-slab coefficient filter: coefficients can drift over time and feature inclusion probabilities update with each training observation. It is a scalable approximate Bayesian filter, not a full MCMC posterior. Neither model is presumed superior; compare the separate chronological scorecards before relying on either.

## Current data

- **UNG daily OHLCV:** Nasdaq's public historical endpoint. Yahoo is not used.
- **EIA weekly working gas:** EIA API (with a free API key) or the official public workbook when no key is configured.
- **Storage features:** reported storage, week-over-week change, deviation from a trailing seasonal five-year comparison, and difference from the prior year's corresponding report week.
- **CFTC positioning:** optional secondary context, retained from the previous project.
- **Henry Hub spot:** shown as reference data; it is not currently included in model features.
- **Pipeline notices:** not connected yet. Public pipeline postings need a reviewed source list and format-specific collectors before they can be used as model inputs.

EIA estimates are subject to revision. A GitHub Actions collector saves normalized snapshots under `data/raw/eia_storage/` on weekdays after the normal report time, starting when this project is enabled. Earlier history from the API/workbook is the latest available vintage, not a complete archive of what was known at every past date. The forecasting run does not yet reconstruct its historical training features from archived vintages. Storage availability timestamps use the normal Thursday 10:30 a.m. Eastern release schedule; holiday changes can differ.

The current project does not yet ingest the complete NYMEX natural-gas futures curve or market expectations for each EIA storage release. A storage surprise feature cannot be built without a point-in-time expectation series. Do not treat a historical expectation collected later as if it were known before the report.

## Run locally

```powershell
python -m pip install -e .
streamlit run streamlit_app.py
```

Set `EIA_API_KEY` in the local environment for EIA's API. The key is optional; without it, the collector uses EIA's public workbook.

Run the scheduled-style collector with:

```powershell
$env:PYTHONPATH = "src"
python run_forecast.py
```

To archive storage manually without generating a forecast:

```powershell
$env:PYTHONPATH = "src"
python collect_storage.py
```

Forecast and outcome ledgers are stored under `data/ledger/`. Do not commit raw snapshots, local ledgers, or credentials.

## Planned data work

1. Compare the storage features against price-only candidates using chronological out-of-sample forecasts.
2. Add a pre-release storage expectation series only if a reliable free historical source is identified and archived point in time.
3. Evaluate a small set of relevant pipeline EBB notices, storing notice text and posting/effective times before attempting to convert notices into numerical supply-loss estimates.
4. Compare CatBoost and Bayesian dynamic selection on the same chronological splits, reviewing forecast error, calibration, inclusion stability, and sensitivity to feature availability.

## Models

- **CatBoost:** independent regressors estimate the expected return and 10th/90th percentile return ranges.
- **Bayesian dynamic selection:** time-varying regression coefficients are updated sequentially; per-feature inclusion probabilities adapt from predictive evidence. The current implementation is an approximate diagonal spike-and-slab filter, so correlated inputs can share or obscure importance. Review inclusion stability and chronological forecast performance before interpreting feature weights.
