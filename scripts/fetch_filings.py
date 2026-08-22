#!/usr/bin/env python3
"""Download 10-K filings from SEC EDGAR and write them out as plain text.

    python scripts/fetch_filings.py --tickers AAPL MSFT JPM BAC

Stdlib only, so it runs before any dependency is installed. EDGAR asks that
automated clients identify themselves and stay under 10 requests/second; both
are honoured below. See https://www.sec.gov/os/accessing-edgar-data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
DEFAULT_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "docs-rag research project (contact: you@example.com)"
)
REQUEST_INTERVAL = 0.2  # seconds; EDGAR's ceiling is 10 requests/second

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "table", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "section", "hr",
}
_DROP_TAGS = {"script", "style", "ix:header"}


class _TextExtractor(HTMLParser):
    """Flatten filing HTML (including inline-XBRL markup) into readable text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._suppress = 0

    def handle_starttag(self, tag, attrs):
        if tag in _DROP_TAGS:
            self._suppress += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _DROP_TAGS and self._suppress:
            self._suppress -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._suppress:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t]+", " ", raw)
        # A run of blank lines is a paragraph break; anything more is noise.
        raw = re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        return raw.strip()


def fetch(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    time.sleep(REQUEST_INTERVAL)
    return payload


def ticker_to_cik(user_agent: str) -> dict[str, int]:
    index = json.loads(fetch(TICKER_INDEX_URL, user_agent))
    return {row["ticker"].upper(): int(row["cik_str"]) for row in index.values()}


def latest_10k(cik: int, user_agent: str) -> tuple[str, str, str]:
    """Return ``(accession, filing_date, primary_document)`` for the newest 10-K."""
    data = json.loads(fetch(SUBMISSIONS_URL.format(cik=cik), user_agent))
    recent = data["filings"]["recent"]
    for accession, form, date, document in zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["primaryDocument"],
    ):
        if form == "10-K":
            return accession, date, document
    raise LookupError(f"no 10-K found for CIK {cik}")


def decode(payload: bytes) -> str:
    """Decode filing bytes.

    EDGAR documents are a mix of UTF-8 and Windows-1252 and rarely declare it
    correctly. Decoding cp1252 as UTF-8 turns every curly apostrophe into a
    replacement character, which then rides all the way into the embeddings and
    the quoted answer, so the fallback matters.
    """
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def download_10k(ticker: str, out_dir: Path, user_agent: str, ciks: dict[str, int]) -> Path:
    ticker = ticker.upper()
    if ticker not in ciks:
        raise LookupError(f"unknown ticker: {ticker}")
    cik = ciks[ticker]

    accession, date, document = latest_10k(cik, user_agent)
    url = ARCHIVE_URL.format(
        cik=cik, accession=accession.replace("-", ""), document=document
    )
    html = decode(fetch(url, user_agent))

    parser = _TextExtractor()
    parser.feed(html)
    body = parser.text()

    out_path = out_dir / f"{ticker}_10-K_{date}.txt"
    # A header line so the ingested title and the citation say something useful.
    out_path.write_text(
        f"{ticker} Form 10-K filed {date} (accession {accession})\n"
        f"Source: {url}\n\n{body}\n",
        encoding="utf-8",
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM", "BAC"])
    parser.add_argument("--out", default="data/filings")
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="EDGAR requires a descriptive User-Agent with contact details.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ciks = ticker_to_cik(args.user_agent)
    for ticker in args.tickers:
        try:
            path = download_10k(ticker, out_dir, args.user_agent, ciks)
        except Exception as exc:  # noqa: BLE001 - one bad ticker should not stop the run
            print(f"{ticker}: FAILED ({exc})")
            continue
        size_kb = path.stat().st_size / 1024
        print(f"{ticker}: {path} ({size_kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
