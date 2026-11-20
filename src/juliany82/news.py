"""Free, keyless GDELT headlines for UNG and natural-gas research."""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd


def fetch_ung_news(max_records: int = 25) -> pd.DataFrame:
    """Fetch recent news from GDELT, falling back to Google News RSS when rate limited."""
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
    try:
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
                    "feed": "GDELT",
                })
    except Exception as gdelt_error:
        rows = _fetch_google_news(max_records)
        if not rows:
            raise RuntimeError(f"GDELT failed ({type(gdelt_error).__name__}) and Google News RSS returned no headlines") from gdelt_error
    frame = pd.DataFrame(rows, columns=["seen_utc", "headline", "source", "language", "article"])
    if rows and "feed" in rows[0]:
        frame["feed"] = rows[0]["feed"]
    return frame.drop_duplicates("article").reset_index(drop=True)


def _fetch_google_news(max_records: int) -> list[dict[str, str]]:
    """Use Google's public news RSS search when GDELT is unavailable."""
    query = 'UNG OR "Henry Hub" OR "natural gas futures" OR "natural gas storage"'
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Juliany82 Research Desk"})
    with urllib.request.urlopen(request, timeout=20) as response:
        root = ET.fromstring(response.read())
    rows = []
    for item in root.findall("./channel/item")[:max_records]:
        title = item.findtext("title", default="").strip()
        link = item.findtext("link", default="").strip()
        source = item.findtext("source", default="").strip()
        if title and link:
            rows.append({
                "seen_utc": item.findtext("pubDate", default=""),
                "headline": title,
                "source": source,
                "language": "en",
                "article": link,
                "feed": "Google News RSS (GDELT rate limited/unavailable)",
            })
    return rows
