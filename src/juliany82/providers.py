"""Initial public-data adapters for UNG and Henry Hub."""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone

import pandas as pd

from .contracts import DataValidationError, FetchedFrame, SourceSnapshot, validate_ohlcv


def _get(url: str, *, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "UNG-Forecast/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise DataValidationError(f"HTTP {response.status}: {url}")
            body = response.read()
    except Exception as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"download failed: {url}: {exc}") from exc
    if not body:
        raise DataValidationError(f"empty response: {url}")
    return body


def _snapshot(provider: str, instrument: str, url: str, raw: bytes, published: date) -> SourceSnapshot:
    return SourceSnapshot(
        provider=provider,
        instrument=instrument,
        retrieved_at=datetime.now(timezone.utc),
        published_through=published,
        content_sha256=hashlib.sha256(raw).hexdigest(),
        source_url=url,
    )


def fetch_nasdaq_ung() -> FetchedFrame:
    """Fetch completed daily UNG bars from Nasdaq's public historical endpoint."""
    today = datetime.now(timezone.utc).date()
    query = urllib.parse.urlencode({
        "assetclass": "etf", "fromdate": "2007-04-18", "todate": today.isoformat(), "limit": "5000"
    })
    url = f"https://api.nasdaq.com/api/quote/UNG/historical?{query}"
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.nasdaq.com",
        "Referer": "https://www.nasdaq.com/",
    })
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
        payload = json.loads(raw)
        rows = payload["data"]["tradesTable"]["rows"]
        frame = pd.DataFrame(rows)
        aliases = {"".join(c for c in str(x).lower() if c.isalnum()): x for x in frame.columns}
        names = {
            "session": aliases.get("date"), "open": aliases.get("open"),
            "high": aliases.get("high"), "low": aliases.get("low"),
            "close": aliases.get("close") or aliases.get("closelast") or aliases.get("lastsaleprice"),
            "volume": aliases.get("volume"),
        }
        if any(value is None for value in names.values()):
            raise ValueError(f"missing Nasdaq columns: {list(frame.columns)}")
        frame = frame[[names[key] for key in names]]
        frame.columns = list(names)
        frame["session"] = pd.to_datetime(frame["session"], errors="raise").dt.normalize()
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="raise")
        frame = frame.set_index("session").sort_index()
    except Exception as exc:
        if isinstance(exc, DataValidationError):
            raise
        raise DataValidationError(f"Nasdaq response schema invalid: {exc}") from exc
    frame = validate_ohlcv(frame)
    frame = frame.loc[frame.index.date < today]
    if len(frame) < 500:
        raise DataValidationError(f"Nasdaq returned too little history: {len(frame)} rows")
    latest = frame.index[-1].date()
    if (today - latest).days > 5:
        raise DataValidationError(f"Nasdaq history is stale: {latest}")
    return FetchedFrame(frame, _snapshot("nasdaq", "UNG", url, raw, latest), raw)
