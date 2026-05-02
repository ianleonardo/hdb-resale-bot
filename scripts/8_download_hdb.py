import io
import time

import pandas as pd
from tqdm.auto import tqdm

from utils import (
    DATAGOV_API_KEY,
    RAW_DIR,
    datagov_poll_download,
    geocode_address,
    load_geocode_cache,
    read_datagov_file,
    save_geocode_cache,
)


def _normalize_address_columns(df: pd.DataFrame, block_col: str, street_col: str) -> pd.DataFrame:
    out = df.copy()
    out[block_col] = out[block_col].astype(str).str.strip().str.upper()
    out[street_col] = out[street_col].astype(str).str.strip().str.upper()
    out = out[(out[block_col] != "") & (out[street_col] != "")]
    return out


def main() -> None:
    # data.gov.sg dataset IDs
    hdb_property_id = "d_17f5382f26140b1fdae0ba2ef6239d2f"
    hdb_resale_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

    property_file = RAW_DIR / "hdb_property_info.csv"
    resale_file = RAW_DIR / "hdb_resale_2017_onwards.csv"
    output_file = RAW_DIR / "hdb_town_block_street_postal.csv"

    datagov_poll_download(hdb_property_id, property_file, api_key=DATAGOV_API_KEY)
    datagov_poll_download(hdb_resale_id, resale_file, api_key=DATAGOV_API_KEY)

    property_bytes = read_datagov_file(property_file)
    resale_bytes = read_datagov_file(resale_file)

    prop = pd.read_csv(io.BytesIO(property_bytes))
    resale = pd.read_csv(io.BytesIO(resale_bytes))

    prop = _normalize_address_columns(prop, block_col="blk_no", street_col="street")
    resale = _normalize_address_columns(resale, block_col="block", street_col="street_name")

    # Build a canonical town mapping from resale records.
    town_map = (
        resale[["town", "block", "street_name"]]
        .rename(columns={"street_name": "street"})
        .dropna(subset=["town"])
        .drop_duplicates()
    )
    town_map["town"] = town_map["town"].astype(str).str.strip().str.upper()

    hdb = (
        prop[["blk_no", "street"]]
        .rename(columns={"blk_no": "block"})
        .drop_duplicates()
        .merge(town_map, on=["block", "street"], how="left")
    )

    hdb["address"] = hdb["block"] + " " + hdb["street"]

    cache = load_geocode_cache()
    postals = []

    print("Geocoding HDB addresses for postal codes...")
    for address in tqdm(hdb["address"]):
        was_cached = address in cache
        result = geocode_address(address, cache)
        if not was_cached:
            # Respect OneMap rate limit when cache misses happen.
            time.sleep(0.3)
        postals.append(str(result.get("postal", "")).strip())

    save_geocode_cache(cache)

    hdb["postal"] = postals
    hdb = hdb[["town", "block", "street", "postal"]].drop_duplicates()

    hdb.to_csv(output_file, index=False)
    print(f"Saved -> {output_file}")
    print(f"Rows: {len(hdb):,}")


if __name__ == "__main__":
    main()
