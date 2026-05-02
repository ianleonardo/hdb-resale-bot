"""
9_download_hdb_rpi.py — Download the official HDB Resale Price Index from data.gov.sg.

Dataset : HDB Resale Price Index (base Q1 2009 = 100), published quarterly.
Source  : https://data.gov.sg/api/action/datastore_search?resource_id=d_f0c768860912d66940efdac9435dc046
Output  : data/hdb_rpi.csv  (columns: year, quarter, quarter_label, rpi)

The API returns one row in wide format with columns like "20251Q" (2025 Q1),
"20254Q" (2025 Q4), etc. This script pivots to long format and saves.
"""

import re
import time

import pandas as pd
import requests

from utils import DATAGOV_API_KEY, DATA_DIR

OUTPUT_PATH = DATA_DIR / "hdb_rpi.csv"

DATASET_ID  = "d_f0c768860912d66940efdac9435dc046"
CKAN_URL    = f"https://data.gov.sg/api/action/datastore_search?resource_id={DATASET_ID}&limit=1"
QUARTER_RE  = re.compile(r"^(\d{4})([1-4])Q$")


def fetch_rpi() -> dict:
    headers = {"x-api-key": DATAGOV_API_KEY} if DATAGOV_API_KEY else {}
    for attempt in range(1, 4):
        resp = requests.get(CKAN_URL, headers=headers, timeout=30)
        if resp.status_code == 429:
            wait = 15 * attempt
            print(f"  Rate-limited — waiting {wait}s (attempt {attempt}/3)...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"API error: {data}")
        return data["result"]["records"][0]
    raise RuntimeError("Failed after 3 attempts (rate limit).")


def parse_wide(record: dict) -> pd.DataFrame:
    """Pivot the single wide row into (year, quarter, quarter_label, rpi) long format."""
    rows = []
    for col, val in record.items():
        m = QUARTER_RE.match(col)
        if not m:
            continue
        year, qtr = int(m.group(1)), int(m.group(2))
        rows.append({
            "year":          year,
            "quarter":       qtr,
            "quarter_label": f"{year}-Q{qtr}",
            "rpi":           float(val),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values(["year", "quarter"])
        .reset_index(drop=True)
    )
    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching HDB Resale Price Index from data.gov.sg...")
    record = fetch_rpi()

    df = parse_wide(record)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} quarters → {OUTPUT_PATH}")
    print(f"Range : {df['quarter_label'].iloc[0]} – {df['quarter_label'].iloc[-1]}")
    print(f"RPI   : {df['rpi'].min():.1f} (min) – {df['rpi'].max():.1f} (max)")
    print()
    print("Most recent 8 quarters:")
    print(df.tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
