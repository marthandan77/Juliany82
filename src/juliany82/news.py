"""Free, keyless GDELT headlines for UNG and natural-gas research."""
from __future__ import annotations

import urllib.parse
import urllib.request

import pandas as pd


def fetch_ung_news(max_records: int = 25) -> pd.DataFrame:
    """Fetch recent GDELT article links; this feed is informational, not a model input."""
    query = '(UNG OR "United States Natural Gas Fund" OR "Henry Hub" OR "natural gas futures" OR "natural gas storage")'
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max(1, min(int(max_records), 50))),
        "timespan": "7d",
        "sort": "DateDesc",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "Juliany82-ResearchDesk/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read()
    import json
    data = json.loads(payload)
    articles = data.get("articles", []) if isinstance(data, dict) else []
    rows = []
    for article in articles:
        title, link = str(article.get("title", "")).strip(), str(article.get("url", "")).strip()
        if title and link:
            rows.append({
                "seen_utc": article.get("seendate", ""),
                "headline": title,
                "source": article.get("domain", article.get("sourceCountry", "")),
                "language": article.get("language", ""),
                "article": link,
            })
    frame = pd.DataFrame(rows, columns=["seen_utc", "headline", "source", "language", "article"])
    return frame.drop_duplicates("article").reset_index(drop=True)
