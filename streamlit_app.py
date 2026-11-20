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
from juliany82.news import fetch_ung_news


def make_ledger() -> ForecastLedger:
    """Use Supabase when its server-side secret is configured; otherwise use local files."""
    try:
        config = st.secrets.get("supabase", {})
        key = config.get("service_role_key") or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        url = config.get("url", DEFAULT_SUPABASE_URL)
    except Exception:
        key, url = "", DEFAULT_SUPABASE_URL
    return ForecastLedger(supabase_url=url, supabase_key=key)

st.set_page_config(page_title="UNG Range Desk", page_icon="NG", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def cached_news():
    return fetch_ung_news()

st.markdown("""<style>
.stApp {background:#0b1118;color:#e6edf3}
[data-testid="stMetric"] {
  background:#142230 !important;
  border:1px solid #526a80 !important;
  border-radius:8px !important;
  padding:14px 16px !important;
  min-height:112px;
  box-shadow:0 2px 8px rgba(0,0,0,.28);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetric"] label, [data-testid="stMetric"] label * {
  color:#c4d4e2 !important;
  opacity:1 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
  color:#ffffff !important;
  opacity:1 !important;
  font-weight:700 !important;
}
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
  color:#b6c8d8 !important;
  opacity:1 !important;
}
div[data-testid="stDataFrame"] {border:1px solid #263746}
h1,h2,h3 {color:#f4b942}
</style>""", unsafe_allow_html=True)
st.title("UNG RANGE DESK")
st.caption("Daily range research terminal · UNG price, natural-gas context, model scorecards and headlines")
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
            henry_hub_snapshot = None
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
                henry_hub_snapshot = fetch_eia_henry_hub_spot()
                henry_hub_history = getattr(henry_hub_snapshot, "history", None)
                henry_feed = describe_henry_hub_feed(henry_hub_snapshot)
                if henry_hub_history is None:
                    henry_feed["role"] = "Reference only; history unavailable in loaded module"
                    henry_feed["note"] = "Spot quote is available, but the loaded EIA parser does not expose history. Restart or refresh the app deployment to enable Henry Hub model features."
                data_feeds.append(henry_feed)
            except Exception as henry_hub_exc:
                henry_hub_history = None
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
                fetched.frame,
                henry_hub=henry_hub_history,
                context=combine_context_sources(context_frames),
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
            st.session_state["henry_hub"] = henry_hub_snapshot
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

forecasts = st.session_state.get("latest_forecasts") or ledger.forecasts()
quality = st.session_state.get("run_quality")
bars = st.session_state.get("latest_bars")
hh_snapshot = st.session_state.get("henry_hub")
cards = build_scorecards(ledger.outcomes())
if bars is not None and not bars.empty:
    last = bars.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("UNG last close", f"${float(last['close']):.2f}", str(bars.index[-1].date()))
    yz = None
    yz5 = yz20 = None
    if "open" in bars:
        from juliany82.features import build_features
        yz_frame = build_features(bars, hh_snapshot.history if hh_snapshot else None)
        yz = yz_frame.iloc[-1].get("yang_zhang_vol_10")
        yz5 = yz_frame.iloc[-1].get("yang_zhang_vol_5")
        yz20 = yz_frame.iloc[-1].get("yang_zhang_vol_20")
    if pd.notna(yz):
        daily_move_pct = float(yz) / (252 ** 0.5)
        daily_move_dollars = float(last["close"]) * daily_move_pct
        m2.metric(
            "Recent volatility · 10 days",
            f"{float(yz):.1%} per year",
            f"About ±${daily_move_dollars:.2f} daily move; either way",
            help="Yang–Zhang estimates how much UNG has been moving, including overnight gaps. The percentage is annualized. It describes move size, not whether price will rise or fall.",
        )
    else:
        m2.metric(
            "Recent volatility · 10 days", "Warming up", "Need more daily price history",
            help="Yang–Zhang needs open, high, low and close prices over the recent sessions.",
        )
    recent = bars.tail(20)
    if len(recent) >= 2 and float(recent["high"].max()) > float(recent["low"].min()):
        range_low = float(recent["low"].min())
        range_high = float(recent["high"].max())
        range_position = (float(last["close"]) - range_low) / (range_high - range_low)
        range_position = min(1.0, max(0.0, range_position))
        position_text = "Near the lower end" if range_position < 0.33 else "Near the upper end" if range_position > 0.67 else "Near the middle"
        m3.metric("Price in 20-day range", f"{range_position:.0%}", position_text)
    else:
        m3.metric("Price in 20-day range", "Warming up", "Need 2+ sessions")
    if forecasts:
        current_origin = max(row.get("issued_at", "") for row in forecasts)
        current = [row for row in forecasts if row.get("issued_at") == current_origin]
        models = sorted({row["model_name"] for row in current})
        a, b = st.columns([1, 1])
        selected_model = a.selectbox("Range model", models, key="desk_model")
        selected_horizon = b.selectbox("Trading horizon", sorted({int(row["horizon"]) for row in current}), format_func=lambda value: f"{value} session(s)", key="desk_horizon")
        selected = next(row for row in current if row["model_name"] == selected_model and int(row["horizon"]) == selected_horizon)
        matching = next((row for row in cards if row["model_name"] == selected_model and int(row["horizon"]) == selected_horizon), None)
        st.subheader("Suggested buy-low / sell-high range")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Buy zone · range low", f"${float(selected['predicted_low']):.2f}")
        r2.metric("Sell zone · range high", f"${float(selected['predicted_high']):.2f}")
        r3.metric("Expected close", f"${float(selected['expected_close']):.2f}")
        width = float(selected['predicted_high']) - float(selected['predicted_low'])
        r4.metric("Forecast width", f"${width:.2f}", f"{width / float(selected['reference_close']):.1%} of reference close")
        if matching and matching.get("matured_forecasts", 0) >= 30:
            st.info(f"Historical range coverage: {matching['range_coverage']:.0%} across {matching['matured_forecasts']} matured forecasts at this model and horizon. This is a past walk-forward coverage rate, not the probability this specific range will hold.")
        else:
            n = matching.get("matured_forecasts", 0) if matching else 0
            st.info(f"Confidence: preliminary · only {n} matured comparable forecasts; 30 are required before displaying historical coverage.")
        st.caption("Research levels, not an execution instruction. UNG can gap beyond either level; check spreads, fees and position limits.")

    st.subheader("Contango & futures-curve effect")
    st.markdown(
        "**Simple meaning:** contango is when next month's gas futures cost more than the front month. "
        "That curve shape can create a headwind for UNG over time as it rolls contracts; it does not predict tomorrow's direction. "
        "The quotes below are TradingView visual context and are not model inputs."
    )
    tv_config = {
        "colorTheme": "dark",
        "dateRange": "1D",
        "isTransparent": True,
        "showSymbolLogo": False,
        "showChart": False,
        "showFloatingTooltip": False,
        "width": "100%",
        "height": 220,
        "locale": "en",
        "tabs": [{
            "title": "Henry Hub futures",
            "symbols": [
                {"s": "NYMEX:NG1!", "d": "Front month"},
                {"s": "NYMEX:NG2!", "d": "Second month"},
            ],
        }],
    }
    st.components.v1.html(
        """<div class="tradingview-widget-container" style="height:220px;width:100%">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/NYMEX-NG1%21/forward-curve/" target="_blank" rel="noopener nofollow">Open full Henry Hub forward curve on TradingView</a></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>
  """ + json.dumps(tv_config) + """
  </script>
</div>""",
        height=260,
        scrolling=False,
    )
    st.caption("Compare front month with second month: if the second-month quote is higher, the near curve is in contango; if lower, it is in backwardation. TradingView may delay quotes. This visual is not ingested into CatBoost or Bayesian forecasts.")

    st.subheader("Useful additions for the desk")
    with st.expander("Range confidence — how often has this model's range held?"):
        st.write("The scorecard compares old forecast ranges with the actual low and high over the same number of sessions. It reports past coverage only after 30 forecasts mature. That percentage describes historical results; it is not the chance this new range will hold.")
    with st.expander("Price location — is UNG near its recent low or high?"):
        st.write(f"The card above locates today's close between the lowest low and highest high of the last 20 sessions ({f'${range_low:.2f}–${range_high:.2f}' if 'range_low' in locals() else 'range not available'}). Near the bottom can help frame a buy zone; it does not guarantee a bounce.")
    with st.expander("Volatility trend — are price swings growing or shrinking?"):
        if pd.notna(yz5) and pd.notna(yz20) and float(yz20) > 0:
            vol_ratio = float(yz5) / float(yz20)
            vol_description = "Recent swings are larger than the 20-session pace" if vol_ratio > 1.05 else "Recent swings are smaller than the 20-session pace" if vol_ratio < 0.95 else "Recent swings are close to the 20-session pace"
            st.write(f"The recent 5-session Yang–Zhang volatility is {vol_ratio:.2f}× the 20-session reading. {vol_description}. This helps describe possible range width, not direction.")
        else:
            st.write("This reading needs enough open, high, low and close history to calculate both volatility windows.")
    with st.expander("Next useful model — forecast range width separately"):
        st.write("A HAR-style volatility model can combine short-, medium- and longer-term volatility to estimate whether upcoming price ranges may widen or narrow. It forecasts move size, not direction. Yang–Zhang inputs are available; a separate walk-forward-tested HAR forecast is not yet on this page.")
    with st.expander("Which level gets touched first?"):
        st.write("This would estimate whether the buy level or sell level is reached first. Daily bars cannot tell the order if both levels trade during the same session. Intraday price history is needed to train and score this fairly.")
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
st.subheader("Individual model scorecards")
if cards:
    scorecard_columns = [
        "model_name", "horizon", "matured_forecasts", "mean_abs_close_error",
        "mean_abs_low_error", "mean_abs_high_error", "range_coverage",
        "average_range_width", "recent_20_abs_close_error",
        "recent_20_abs_low_error", "recent_20_abs_high_error",
        "recent_20_range_coverage", "status",
    ]
    st.dataframe(
        [{key: row.get(key) for key in scorecard_columns} for row in cards],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Range coverage = percentage of sessions where the actual low and high stayed inside the forecast range.")
    st.caption("Errors are mean absolute price errors. All-time columns include all matured forecasts; recent columns use the last 20 for that model and horizon. Lower error is better. Narrower ranges are useful only when coverage remains high.")
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
    explainable = [row for row in forecast_history if row.get("feature_contributions")]
    if explainable:
        st.subheader("Input influence on forecast")
        explainable.sort(key=lambda row: str(row.get("issued_at", "")), reverse=True)
        labels = {
            str(row["forecast_id"]): (
                f"{row['origin_session']} → {row['target_session']} | "
                f"{row['model_name']} | {row['horizon']} session(s)"
            )
            for row in explainable
        }
        selected_id = st.selectbox(
            "Choose a forecast vintage", list(labels), format_func=lambda value: labels[value]
        )
        selected_forecast = next(row for row in explainable if str(row["forecast_id"]) == selected_id)
        contribution_frame = pd.DataFrame(selected_forecast["feature_contributions"])
        display_columns = [
            column for column in (
                "feature", "feature_value", "contribution", "selection_probability",
                "effective_coefficient", "method",
            ) if column in contribution_frame
        ]
        st.dataframe(
            contribution_frame[display_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Contributions explain the model's expected-close return forecast. Positive values push the forecast up; negative values push it down. This measures influence on a prediction, not whether the input improved accuracy or caused the move.")
        st.caption("Accuracy benefit requires a separate chronological feature-removal evaluation. Historical EIA vintages are incomplete, so that comparison is not yet reliable for older periods.")
        impact_rows = []
        history_by_id = {str(row["forecast_id"]): row for row in forecast_history}
        recent_outcomes: dict[tuple[str, int], list[dict]] = {}
        for outcome in sorted(outcomes, key=lambda row: str(row.get("scored_at", ""))):
            group = (str(outcome["model_name"]), int(outcome["horizon"]))
            recent_outcomes.setdefault(group, []).append(outcome)
        for (model_name, horizon), group_outcomes in recent_outcomes.items():
            for outcome in group_outcomes[-20:]:
                forecast_row = history_by_id.get(str(outcome["forecast_id"]), {})
                for contribution in forecast_row.get("feature_contributions", []):
                    impact_rows.append({
                        "model_name": model_name,
                        "horizon": horizon,
                        "feature": contribution["feature"],
                        "contribution": float(contribution["contribution"]),
                        "selection_probability": contribution.get("selection_probability"),
                    })
        if impact_rows:
            impact = pd.DataFrame(impact_rows)
            summary = impact.groupby(["model_name", "horizon", "feature"], as_index=False).agg(
                scored_forecasts=("contribution", "count"),
                mean_absolute_contribution=("contribution", lambda values: values.abs().mean()),
                mean_signed_contribution=("contribution", "mean"),
                mean_selection_probability=("selection_probability", "mean"),
            )
            summary = summary.sort_values(
                ["model_name", "horizon", "mean_absolute_contribution"],
                ascending=[True, True, False],
            )
            top_inputs = summary.groupby(["model_name", "horizon"], sort=False).head(5)
            st.caption("Largest average absolute contributions among the latest 20 matured forecasts per model and horizon.")
            st.dataframe(top_inputs, use_container_width=True, hide_index=True)
if outcomes:
    st.subheader("Forecast vs actual history")
    comparisons = pd.DataFrame(outcomes).sort_values("scored_at", ascending=False).head(100)
    comparison_columns = [
        "target_session", "model_name", "horizon",
        "predicted_low", "actual_low", "low_error",
        "predicted_close", "actual_close", "close_error",
        "predicted_high", "actual_high", "high_error",
    ]
    comparisons = comparisons[[column for column in comparison_columns if column in comparisons]]
    st.dataframe(comparisons, use_container_width=True, hide_index=True)
    st.caption(f"Showing the latest {len(comparisons)} scored forecasts. All scored vintages remain in the ledger. Errors are forecast minus actual.")

st.subheader("UNG & natural-gas headlines")
st.caption("Recent news links · informational context only; headlines are not model inputs or a sentiment signal.")
if st.button("Refresh headlines"):
    cached_news.clear()
try:
    news = cached_news()
    if news.empty:
        st.info("No matching headlines were returned for the past seven days.")
    else:
        feed_label = str(news.iloc[0].get("feed", "GDELT"))
        st.caption(f"Feed: {feed_label} · cached for up to one hour")
        st.dataframe(
            news.rename(columns={"seen_utc": "Seen (UTC)", "headline": "Headline", "source": "Source", "language": "Language", "article": "Article"}),
            use_container_width=True,
            hide_index=True,
            column_config={"Article": st.column_config.LinkColumn("Article", display_text="Read")},
        )
except Exception as news_exc:
    st.warning(f"Headline feed unavailable ({type(news_exc).__name__}). Forecasting and scorecards remain available.")

st.subheader("Planned output")
st.write("CatBoost captures nonlinear feature interactions. Bayesian dynamic selection estimates time-varying coefficients and updates each input's inclusion probability as new observations arrive. Compare both using their separate walk-forward scorecards; neither is assumed superior before out-of-sample evidence.")
