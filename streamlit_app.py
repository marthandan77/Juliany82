"""Streamlit shell for the rolling UNG forecast system."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from juliany82.ledger import DEFAULT_SUPABASE_URL, ForecastLedger
from juliany82.context_data import (
    combine_context_sources,
    fetch_cftc_positioning,
    fetch_eia_henry_hub_spot,
    fetch_eia_storage,
    archive_storage_snapshot,
)
from juliany82.feed_status import (
    describe_context_feed,
    describe_henry_hub_feed,
    describe_price_feed,
    unavailable_feed,
)
from juliany82.providers import fetch_nasdaq_ung
from juliany82.rolling import generate_rolling_forecasts
from juliany82.scorecards import build_scorecards, is_comparable_outcome


def make_ledger() -> ForecastLedger:
    """Use Supabase when its server-side secret is configured; otherwise use local files."""
    try:
        config = st.secrets.get("supabase", {})
        key = config.get("service_role_key") or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        url = config.get("url", DEFAULT_SUPABASE_URL)
    except Exception:
        key, url = "", DEFAULT_SUPABASE_URL
    return ForecastLedger(supabase_url=url, supabase_key=key)

st.set_page_config(page_title="UNG Storage & Supply Research", page_icon="NG", layout="wide")
st.title("UNG Storage & Supply Research")
st.caption("Daily and five-session experimental forecasts from two complementary models.")
ledger = make_ledger()
if ledger.use_supabase:
    st.caption("Forecast history: Supabase connected")
else:
    st.warning("Forecast history is using local files and may not persist on hosted Streamlit. Add the Supabase service role key to Streamlit secrets to enable shared storage.")

run_report_path = Path("data/ledger/latest-run.json")
persisted_report = {}
if run_report_path.exists():
    try:
        persisted_report = json.loads(run_report_path.read_text(encoding="utf-8"))
        st.caption(
            f"Latest scan: {persisted_report.get('run_at', 'unknown')} | "
            f"Status: {persisted_report.get('status', 'unknown')} | "
            f"Source: {persisted_report.get('trigger', 'scheduled run')}"
        )
    except (OSError, json.JSONDecodeError):
        st.warning("The latest run report exists but could not be read.")

st.info(
    "CatBoost and Bayesian dynamic regression with variable selection are experimental models. "
    "Their forecasts are research outputs, "
    "not trade recommendations."
)
if st.button("Scan", type="primary", use_container_width=True):
    with st.spinner("Scanning configured feeds, scoring completed forecasts, and updating models..."):
        try:
            fetched = fetch_nasdaq_ung()
            agreement = "UNG daily OHLCV fetched from Nasdaq; no second provider configured."
            provider_status = "NASDAQ"
            data_feeds = [describe_price_feed(
                "Nasdaq UNG", fetched, "UNG OHLCV target and price features"
            )]
            context_frames = []
            context_status = {}
            feed_specs = (
                (
                    "eia_storage", "EIA storage", fetch_eia_storage,
                    ("storage_bcf", "storage_change_bcf", "storage_vs_5y_bcf", "storage_vs_year_ago_bcf"),
                    "https://ir.eia.gov/ngs/wngsr.csv",
                    "Availability uses the usual Thursday 10:30 a.m. Eastern release time. Holiday schedule changes may differ. Retrieved data vintages are archived from collection start; older history may contain later EIA revisions.",
                ),
                (
                    "cftc_positioning", "CFTC natural-gas positioning", fetch_cftc_positioning,
                    ("managed_money_net_pct_oi", "producer_merchant_net_pct_oi"),
                    "https://www.cftc.gov/dea/futures/nat_gas_lf.htm",
                    "Availability uses a conservative report-date + 4 days, 21:00 UTC rule, not an observed publication timestamp.",
                ),
            )
            for key, name, fetcher, value_columns, source_url, availability_note in feed_specs:
                try:
                    frame = fetcher()
                    if key == "eia_storage":
                        archive_storage_snapshot(frame)
                    context_frames.append(frame)
                    context_status[key] = "AVAILABLE"
                    data_feeds.append(describe_context_feed(
                        name, frame, value_columns=value_columns, availability_note=availability_note,
                        source_url=source_url,
                    ))
                except Exception as context_exc:
                    context_status[key] = f"UNAVAILABLE: {type(context_exc).__name__}"
                    data_feeds.append(unavailable_feed(
                        name, f"Fetch or validation failed ({type(context_exc).__name__})."
                    ))
            try:
                data_feeds.append(describe_henry_hub_feed(fetch_eia_henry_hub_spot()))
            except Exception as henry_hub_exc:
                data_feeds.append(unavailable_feed(
                    "EIA Henry Hub spot", f"Fetch or validation failed ({type(henry_hub_exc).__name__})."
                ))
            data_feeds.extend([
                unavailable_feed(
                    "Market depth", "No depth feed is configured in the daily UNG collector.",
                    status="NOT_CONNECTED",
                ),
                unavailable_feed(
                    "Pipeline supply notices", "No pipeline notice pages are configured yet.",
                    status="NOT_CONFIGURED",
                ),
            ])
            ledger = make_ledger()
            scored = ledger.score_matured(fetched.frame)
            generated = generate_rolling_forecasts(
                fetched.frame, context=combine_context_sources(context_frames),
                model_names=("catboost", "bayesian_dynamic_selection"), ledger=ledger
            )
            report = {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "trigger": "manual_scan",
                "status": "READY" if context_status.get("eia_storage") == "AVAILABLE" else "PROVISIONAL",
                "provider": fetched.snapshot.provider,
                "provider_status": provider_status,
                "agreement": agreement,
                "context": context_status,
                "context_source_count": sum(value == "AVAILABLE" for value in context_status.values()),
                "data_feeds": data_feeds,
                "latest_session": str(fetched.frame.index[-1].date()),
                "history_rows": len(fetched.frame),
                "forecasts": len(generated),
                "scored": scored,
                "ledger_forecasts": len(ledger.forecasts()),
                "ledger_outcomes": len(ledger.outcomes()),
            }
            try:
                run_report_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_report_path = run_report_path.with_suffix(".json.tmp")
                temporary_report_path.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                temporary_report_path.replace(run_report_path)
            except OSError as report_exc:
                st.warning(
                    "Scan results are available in this app session; run report could not be saved "
                    f"({type(report_exc).__name__})."
                )
            st.session_state["latest_forecasts"] = generated
            st.session_state["latest_bars"] = fetched.frame.tail(120)
            st.session_state["run_quality"] = {
                "status": report["status"],
                "provider_status": provider_status,
                "agreement": agreement,
                "context_source_count": sum(value == "AVAILABLE" for value in context_status.values()),
                "data_feeds": data_feeds,
            }
            st.success(
                f"Scan complete: {len(data_feeds)} feed statuses refreshed, {scored} matured forecasts scored, "
                f"and {len(generated)} forecast rows updated."
            )
        except Exception as exc:
            st.error(f"Forecast blocked: {type(exc).__name__}: {exc}")

forecasts = st.session_state.get("latest_forecasts", [])
quality = st.session_state.get("run_quality")
feed_rows = quality.get("data_feeds") if quality else persisted_report.get("data_feeds", [])
st.subheader("Market input availability")
if feed_rows:
    feed_columns = {
        "source": "Source", "status": "Status", "latest_observation": "Latest observation",
        "latest_value": "Latest value", "available_at": "Availability timing",
        "retrieved_at": "Checked (UTC)", "source_url": "Source link", "role": "Use", "note": "Notes",
    }
    feed_table = pd.DataFrame(feed_rows)
    for column in feed_columns:
        if column not in feed_table:
            feed_table[column] = "â€”"
    st.dataframe(
        feed_table[list(feed_columns)].rename(columns=feed_columns),
        use_container_width=True,
        hide_index=True,
        column_config={"Source link": st.column_config.LinkColumn("Source link", display_text="Open")},
    )
    st.caption(
        "Feed availability means the source was fetched and validated. Storage signals include "
        "week-to-week change and seasonal/year-over-year differences."
    )
else:
    st.info("No feed inventory has been saved yet. Run the storage archive or scan to populate it.")
if quality:
    st.subheader("Latest run quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Status", quality["status"])
    q2.metric("UNG providers", quality.get("provider_status", "UNKNOWN"))
    q3.metric("Context sources", f"{quality.get('context_source_count', 0)}/2")
    st.caption(quality["agreement"])
if forecasts:
    st.subheader("Next five trading sessions")
    st.caption("Low and high show the model's expected trading range. Close is the expected closing price.")
    forecast_columns = {
        "target_session": "Session", "model_name": "Model", "horizon": "Days ahead",
        "predicted_low": "Forecast low", "expected_close": "Forecast close",
        "predicted_high": "Forecast high", "probability_up": "Up probability",
        "calibration_status": "Range calibration",
        "calibration_sample_count": "Calibration samples",
    }
    forecast_data = pd.DataFrame(forecasts)
    for column in forecast_columns:
        if column not in forecast_data:
            forecast_data[column] = "â€”"
    forecast_table = forecast_data[list(forecast_columns)].rename(columns=forecast_columns)
    forecast_table["Up probability"] = (forecast_table["Up probability"] * 100).round(0).astype(str) + "%"
    for column in ("Forecast low", "Forecast close", "Forecast high"):
        forecast_table[column] = forecast_table[column].round(2)
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
cards = build_scorecards(ledger.outcomes())
st.subheader("Individual model scorecards")
if cards:
    scorecard_columns = [
        "model_name", "horizon", "matured_forecasts", "mean_abs_close_error",
        "mean_abs_low_error", "mean_abs_high_error", "range_coverage",
        "average_range_width", "status",
    ]
    st.dataframe(
        [{key: row.get(key) for key in scorecard_columns} for row in cards],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Range coverage = percentage of sessions where the actual low and high stayed inside the forecast range.")
    st.caption("Close/low/high errors are mean absolute price errors. Lower is better. Narrower ranges are useful only when coverage remains high.")
    scorecard_frame = pd.DataFrame(cards)
    st.subheader("Range accuracy at a glance")
    chart_frame = scorecard_frame.pivot(index="model_name", columns="horizon", values="range_coverage")
    chart_frame.columns = [f"{int(value)}d" for value in chart_frame.columns]
    st.bar_chart(chart_frame, y_label="Range coverage (1.0 = 100%)")
else:
    st.info("No matured forecasts yet; scorecards will populate as target sessions complete.")

outcomes = [row for row in ledger.outcomes() if is_comparable_outcome(row)]
forecast_history = ledger.forecasts()
if forecast_history:
    st.subheader("Forecast history")
    scored_ids = {str(row["forecast_id"]) for row in outcomes}
    history = pd.DataFrame(forecast_history)
    history["result_status"] = history["forecast_id"].astype(str).map(
        lambda forecast_id: "SCORED" if forecast_id in scored_ids else "AWAITING TARGET BAR"
    )
    history = history.sort_values("issued_at", ascending=False)
    history_columns = [
        "issued_at", "origin_session", "target_session", "model_name", "horizon",
        "predicted_low", "expected_close", "predicted_high", "result_status",
    ]
    st.caption(f"All forecast vintages are retained in the ledger ({len(history)} total). Showing the latest 100 here.")
    st.dataframe(
        history[[column for column in history_columns if column in history]].head(100),
        use_container_width=True,
        hide_index=True,
    )
if outcomes:
    st.subheader("Latest forecast vs actual")
    comparisons = pd.DataFrame(outcomes).sort_values("scored_at", ascending=False).head(20)
    comparison_columns = [
        "target_session", "model_name", "horizon",
        "predicted_low", "actual_low", "low_error",
        "predicted_close", "actual_close", "close_error",
        "predicted_high", "actual_high", "high_error",
    ]
    comparisons = comparisons[[column for column in comparison_columns if column in comparisons]]
    st.dataframe(comparisons, use_container_width=True, hide_index=True)
    st.caption("Forecast errors are predicted price minus actual price. A row is added once when its target session has a completed Nasdaq bar.")

st.subheader("Planned output")
st.write("CatBoost captures nonlinear feature interactions. Bayesian dynamic selection estimates time-varying coefficients and updates each input's inclusion probability as new observations arrive. Compare both using their separate walk-forward scorecards; neither is assumed superior before out-of-sample evidence.")
