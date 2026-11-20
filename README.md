# UNG Storage and Supply Research

Research application for daily UNG price features and rolling one-to-five-session forecasts. The initial focus is EIA storage tightness and natural-gas supply context. Forecasts are experimental and are not trade recommendations.

## Model status

Model choice is intentionally open. Ridge, histogram gradient boosting, random forest, and extra trees are retained as separate challenger candidates from the earlier project. The application reports each candidate separately and does not select a winner or create an adaptive ensemble. We should choose the final model only after agreeing on the targets, walk-forward evaluation, and scoring criteria.

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

Set `EIA_API_KEY` in the local environment for EIA's API. The key is optional; without it, the collector uses EIA's public workbook. Set `JULIANY_MODELS` to a comma-separated list of candidate model names to run a subset.

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
4. Decide on the final ML method only after reviewing benchmark performance, calibration, and forecast stability.
