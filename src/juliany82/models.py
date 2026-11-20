"""Forecast models: CatBoost and Bayesian dynamic regression with selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


CATBOOST = "catboost"
BAYESIAN_DYNAMIC = "bayesian_dynamic_selection"
MODEL_NAMES = (CATBOOST, BAYESIAN_DYNAMIC)


@dataclass(frozen=True, slots=True)
class ModelForecast:
    model_name: str
    horizon: int
    target_session: str
    predicted_low: float
    expected_close: float
    predicted_high: float
    probability_up: float
    uncertainty: float
    feature_contributions: list[dict[str, Any]]


def _clean_features(train: pd.DataFrame, predict: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute and standardize using training rows only."""
    values = train.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    latest = predict.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians)
    latest = np.where(np.isfinite(latest), latest, medians)
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return (values - means) / scales, (latest - means) / scales


def _catboost_predict(train_x: pd.DataFrame, train_y: np.ndarray, predict_x: pd.DataFrame,
                      loss: str, *, explain: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from catboost import CatBoostRegressor, Pool

    model = CatBoostRegressor(
        loss_function=loss, iterations=350, depth=5, learning_rate=0.035,
        l2_leaf_reg=5.0, random_seed=42, verbose=False, allow_writing_files=False,
        thread_count=4,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(predict_x)
    contributions: list[dict[str, Any]] = []
    if explain:
        shap_values = model.get_feature_importance(data=Pool(predict_x), type="ShapValues")
        values = predict_x.iloc[0]
        contributions = [
            {
                "feature": str(column),
                "feature_value": float(values[column]) if pd.notna(values[column]) else None,
                "contribution": float(shap_values[0, position]),
                "method": "CatBoost SHAP",
            }
            for position, column in enumerate(predict_x.columns)
        ]
        contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return predictions, model.predict(train_x), contributions


def _bayesian_dynamic_predict(
    train_x: pd.DataFrame, train_y: np.ndarray, predict_x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Diagonal spike-and-slab dynamic linear model with online inclusion odds.

    Coefficients follow a random walk. Each feature has an inclusion probability
    updated from its one-step predictive Bayes factor at each training date.
    This is an approximate, scalable Bayesian variable-selection filter, not MCMC.
    """
    x, future = _clean_features(train_x, predict_x)
    y = np.asarray(train_y, dtype=float)
    usable = np.isfinite(y)
    x, y = x[usable], y[usable]
    if not len(y):
        raise ValueError("Bayesian dynamic regression has no finite target rows")

    p = x.shape[1]
    variance = max(float(np.var(y)), 1e-8)
    observation_var = max(variance * 0.25, 1e-8)
    beta = np.zeros(p)
    beta_var = np.full(p, max(variance / max(p, 1), 1e-6))
    inclusion = np.full(p, 0.5)
    intercept = 0.0
    intercept_var = max(variance, 1e-6)
    residuals = []

    for row, target in zip(x, y):
        predicted = intercept + float(np.dot(inclusion * beta, row))
        residuals.append(target - predicted)

        # Dynamic intercept update.
        intercept_var += variance * 0.002
        intercept_gain = intercept_var / (intercept_var + observation_var)
        intercept += intercept_gain * (target - predicted)
        intercept_var *= 1.0 - intercept_gain

        for j in range(p):
            if abs(row[j]) < 1e-10:
                beta_var[j] += variance * 0.01
                continue
            partial = target - intercept - float(np.dot((inclusion * beta), row)) + inclusion[j] * beta[j] * row[j]
            prior_var = beta_var[j] + variance * 0.01
            spike_var = observation_var
            slab_var = observation_var + row[j] * row[j] * prior_var
            log_bf = 0.5 * (math.log(spike_var / slab_var)) + 0.5 * (
                partial * partial / spike_var - (partial - row[j] * beta[j]) ** 2 / slab_var
            )
            # Temper Bayes factors and retain a small amount of prior adaptation.
            prior_probability = float(np.clip(0.995 * inclusion[j] + 0.005 * 0.5, 1e-4, 1 - 1e-4))
            log_odds = math.log(prior_probability / (1 - prior_probability)) + float(np.clip(log_bf, -8, 8))
            inclusion[j] = 1.0 / (1.0 + math.exp(-float(np.clip(log_odds, -30, 30))))

            gain = prior_var * row[j] / slab_var
            beta[j] += gain * (partial - row[j] * beta[j])
            beta_var[j] = max((1.0 - gain * row[j]) * prior_var, 1e-10)

    forecast = intercept + future @ (inclusion * beta)
    raw_latest = predict_x.iloc[0]
    contributions = [
        {
            "feature": str(column),
            "feature_value": float(raw_latest[column]) if pd.notna(raw_latest[column]) else None,
            "selection_probability": float(inclusion[position]),
            "effective_coefficient": float(inclusion[position] * beta[position]),
            "contribution": float(inclusion[position] * beta[position] * future[0, position]),
            "method": "Bayesian inclusion-weighted coefficient",
        }
        for position, column in enumerate(train_x.columns)
    ]
    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return forecast, np.asarray(residuals, dtype=float), contributions


def _predict(name: str, x: pd.DataFrame, y: np.ndarray, latest: pd.DataFrame,
             key: str, *, explain: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if name == CATBOOST:
        loss = "RMSE" if key == "close" else f"Quantile:alpha={0.1 if key == 'low' else 0.9}"
        predictions, fitted, contributions = _catboost_predict(x, y, latest, loss, explain=explain)
        return np.asarray(predictions), np.asarray(y, dtype=float) - fitted, contributions
    if name == BAYESIAN_DYNAMIC:
        predictions, residuals, contributions = _bayesian_dynamic_predict(x, y, latest)
        return np.asarray(predictions), residuals, contributions if explain else []
    raise ValueError(f"unknown model: {name}; choose from {', '.join(MODEL_NAMES)}")


def fit_individual_forecast(
    model_name: str, features: pd.DataFrame, targets: pd.DataFrame,
    *, reference_close: float, target_session: str, horizon: int, min_rows: int = 100,
) -> ModelForecast:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model_name}; choose from {', '.join(MODEL_NAMES)}")
    joined = features.join(targets).dropna(
        subset=["forward_low_return", "forward_return", "forward_high_return"]
    )
    if len(joined) < min_rows:
        raise ValueError(f"{model_name}: need {min_rows} rows, have {len(joined)}")
    usable_columns = [column for column in features.columns if joined[column].notna().any()]
    x = joined[usable_columns]
    latest = features.iloc[[-1]][usable_columns]
    forecasts = {}
    residuals = None
    feature_contributions: list[dict[str, Any]] = []
    for key, target in (("low", "forward_low_return"), ("close", "forward_return"),
                        ("high", "forward_high_return")):
        predictions, target_residuals, contributions = _predict(
            model_name, x, joined[target].to_numpy(dtype=float), latest, key, explain=(key == "close")
        )
        forecasts[key] = float(predictions[0])
        if key == "close":
            residuals = target_residuals
            feature_contributions = contributions

    low = max(0.01, reference_close * (1 + forecasts["low"]))
    close = max(0.01, reference_close * (1 + forecasts["close"]))
    high = max(0.01, reference_close * (1 + forecasts["high"]))
    low, high = min(low, high), max(low, high)
    residual_scale = max(float(np.nanstd(residuals)) if residuals is not None else 0.0, 1e-6)
    probability_up = 0.5 * (1.0 + math.erf(forecasts["close"] / (residual_scale * math.sqrt(2.0))))
    probability_up = float(np.clip(probability_up, 0.01, 0.99))
    uncertainty = float(np.clip(residual_scale + (high - low) / max(close, 1e-6), 0.0, 1.0))
    return ModelForecast(
        model_name, horizon, target_session, low, close, high, probability_up, uncertainty,
        feature_contributions,
    )


def score_forecast(forecast: dict, actual_low: float, actual_close: float, actual_high: float) -> dict[str, float | bool]:
    """Score one matured forecast without changing the original forecast record."""
    low_error = float(forecast["predicted_low"]) - actual_low
    close_error = float(forecast["expected_close"]) - actual_close
    high_error = float(forecast["predicted_high"]) - actual_high
    return {
        "low_error": low_error,
        "close_error": close_error,
        "high_error": high_error,
        "predicted_range_width": float(forecast["predicted_high"]) - float(forecast["predicted_low"]),
        "actual_range_width": actual_high - actual_low,
        "range_contains_close": bool(forecast["predicted_low"] <= actual_close <= forecast["predicted_high"]),
        "range_contains_day": bool(forecast["predicted_low"] <= actual_low and actual_high <= forecast["predicted_high"]),
    }


def walk_forward_evaluate(
    model_name: str, features: pd.DataFrame, targets: pd.DataFrame, *,
    horizon: int, min_train_rows: int = 100, step: int = 5,
) -> list[dict[str, float | str | int | bool]]:
    """Evaluate an individual model using expanding-window, future-blind forecasts."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {model_name}; choose from {', '.join(MODEL_NAMES)}")
    joined = features.join(targets)
    rows: list[dict[str, float | str | int | bool]] = []
    for index in range(min_train_rows, len(joined), step):
        train_end = max(0, index - horizon)
        train_features = features.iloc[:train_end]
        train_targets = targets.iloc[:train_end]
        test = joined.iloc[index:index + step].dropna(
            subset=["forward_low_return", "forward_return", "forward_high_return"]
        )
        if test.empty:
            continue
        train = train_features.join(train_targets).dropna(
            subset=["forward_low_return", "forward_return", "forward_high_return"]
        )
        if len(train) < min_train_rows:
            continue
        usable = [column for column in features.columns if train[column].notna().any()]
        x = train[usable]
        test_x = features.loc[test.index, usable]
        predictions, _, _ = _predict(
            model_name, x, train["forward_return"].to_numpy(dtype=float), test_x, "close"
        )
        for position, timestamp in enumerate(test.index):
            actual = float(joined.loc[timestamp, "forward_return"])
            rows.append({
                "model_name": model_name,
                "horizon": horizon,
                "target_session": timestamp.isoformat(),
                "predicted_return": float(predictions[position]),
                "actual_return": actual,
                "absolute_error": abs(float(predictions[position]) - actual),
                "direction_correct": bool((predictions[position] > 0) == (actual > 0)),
            })
    return rows
