"""Append-only forecast and outcome ledgers."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .models import score_forecast

DEFAULT_SUPABASE_URL = "https://majosloydqiwmsesgikg.supabase.co"


class ForecastLedger:
    def __init__(
        self, directory: str | Path = "data/ledger", *,
        supabase_url: str | None = None, supabase_key: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.forecasts_path = self.directory / "forecasts.jsonl"
        self.outcomes_path = self.directory / "outcomes.jsonl"
        self.supabase_url = (
            supabase_url or os.getenv("SUPABASE_URL", DEFAULT_SUPABASE_URL)
        ).rstrip("/")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.use_supabase = bool(self.supabase_url and self.supabase_key)
        if not self.use_supabase:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def forecasts(self) -> list[dict[str, Any]]:
        if self.use_supabase:
            return self._remote_read("forecasts")
        return self._read(self.forecasts_path)

    def outcomes(self) -> list[dict[str, Any]]:
        if self.use_supabase:
            return self._remote_read("outcomes")
        return self._read(self.outcomes_path)

    def _request(self, table: str, *, query: str = "", method: str = "GET",
                 body: list[dict[str, Any]] | None = None, prefer: str = "") -> bytes:
        url = f"{self.supabase_url}/rest/v1/{table}"
        if query:
            url += f"?{query}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        payload = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        request = Request(url, data=payload, headers=headers, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                return response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Supabase {table} request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Supabase {table} connection failed: {exc.reason}") from exc

    def _remote_read(self, table: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        offset = 0
        page_size = 1000
        while True:
            query = urlencode({"select": "record", "order": "created_at.asc", "limit": page_size,
                               "offset": offset})
            payload = self._request(table, query=query)
            rows = json.loads(payload or b"[]")
            records.extend(row["record"] for row in rows)
            if len(rows) < page_size:
                return records
            offset += page_size

    def append_forecast(self, record: dict[str, Any]) -> bool:
        """Append once by forecast_id; return False for an existing vintage."""
        if self.use_supabase:
            query = urlencode({"on_conflict": "forecast_id"})
            before = self._request(
                "forecasts", query=query, method="POST",
                body=[{"forecast_id": str(record["forecast_id"]), "record": record}],
                prefer="resolution=ignore-duplicates,return=representation",
            )
            return bool(json.loads(before or b"[]"))
        existing = {str(row["forecast_id"]) for row in self.forecasts()}
        if str(record["forecast_id"]) in existing:
            return False
        with self.forecasts_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return True

    def score_matured(self, bars: pd.DataFrame) -> int:
        """Score forecasts against their horizon range and target-session close."""
        old_outcomes = self.outcomes()
        existing = {
            str(row["forecast_id"]): row for row in old_outcomes
            if row.get("scoring_basis") == "horizon_window"
            or (not row.get("scoring_basis") and int(row.get("horizon", 0)) == 1)
        }
        bar_dates = pd.DatetimeIndex(pd.to_datetime(bars.index))
        if bar_dates.tz is not None:
            bar_dates = bar_dates.tz_localize(None)
        bar_dates = bar_dates.normalize()
        rows: list[dict[str, Any]] = []
        for forecast in self.forecasts():
            forecast_id = str(forecast["forecast_id"])
            target = pd.Timestamp(forecast["target_session"]).normalize()
            origin = pd.Timestamp(forecast["origin_session"]).normalize()
            target_positions = (bar_dates == target).nonzero()[0]
            if not len(target_positions):
                continue
            period_positions = ((bar_dates > origin) & (bar_dates <= target)).nonzero()[0]
            if not len(period_positions):
                continue
            target_bar = bars.iloc[int(target_positions[-1])]
            period_bars = bars.iloc[period_positions]
            if forecast_id in existing:
                continue
            actual_low = float(period_bars.low.min())
            actual_high = float(period_bars.high.max())
            actual_close = float(target_bar.close)
            result = score_forecast(forecast, actual_low, actual_close, actual_high)
            rows.append({
                "forecast_id": forecast_id,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "target_session": target.date().isoformat(),
                "model_name": forecast["model_name"],
                "horizon": forecast["horizon"],
                "predicted_low": float(forecast["predicted_low"]),
                "predicted_close": float(forecast["expected_close"]),
                "predicted_high": float(forecast["predicted_high"]),
                "actual_low": actual_low,
                "actual_close": actual_close,
                "actual_high": actual_high,
                "scoring_basis": "horizon_window",
                **result,
            })
        if rows:
            if self.use_supabase:
                payload = [{"forecast_id": row["forecast_id"], "record": row} for row in rows]
                self._request(
                    "outcomes", query=urlencode({"on_conflict": "forecast_id"}), method="POST",
                    body=payload, prefer="resolution=merge-duplicates,return=minimal",
                )
                return len(rows)
            merged = {str(row["forecast_id"]): row for row in old_outcomes}
            merged.update({str(row["forecast_id"]): row for row in rows})
            temporary = self.outcomes_path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                       for row in merged.values()),
                encoding="utf-8",
            )
            temporary.replace(self.outcomes_path)
        return len(rows)
