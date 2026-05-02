"""
7_build_enriched_dataset.py
────────────────────────────
Downloads the latest HDB resale flat prices from data.gov.sg, enriches each
transaction with building info, geocoordinates (OneMap), and proximity features
to amenities (MRT, bus stops, malls, hawker centres, schools).

Outputs
-------
data/hdb_resale_complete.csv   – final enriched dataset (~65+ columns)

Run from the project root:
    python scripts/7_build_enriched_dataset.py
"""

import io
import os
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree
from tqdm.auto import tqdm
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / 'data'
RAW_DIR   = DATA_DIR / 'raw'
CACHE_DIR = DATA_DIR / 'cache'
for d in [RAW_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Credentials ───────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / '.env')
ONEMAP_EMAIL    = os.getenv('ONEMAP_EMAIL', '')
ONEMAP_PASSWORD = os.getenv('ONEMAP_PASSWORD', '')
DATAGOV_API_KEY = os.getenv('DATAGOV_API_KEY', '')

if not DATAGOV_API_KEY:
    sys.exit('ERROR: DATAGOV_API_KEY not set in .env')

EARTH_RADIUS_M = 6_371_000

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def datagov_poll_download(dataset_id, save_path, api_key=None, max_retries=60, sleep_sec=5):
    """Download a dataset from data.gov.sg v2 API, with polling until ready."""
    save_path = Path(save_path)
    if save_path.exists():
        print(f'  Cached: {save_path.name}')
        return save_path
    url     = f'https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download'
    headers = {'x-api-key': api_key} if api_key else {}
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        dl_url = resp.json().get('data', {}).get('url')
        if dl_url:
            print(f'  Downloading {dataset_id} ...')
            r = requests.get(dl_url, stream=True, timeout=300)
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65_536):
                    f.write(chunk)
            print(f'  Saved -> {save_path.name}')
            return save_path
        print(f'  Preparing ({attempt}/{max_retries}) ...')
        time.sleep(sleep_sec)
    raise TimeoutError(f'Dataset {dataset_id} not ready after {max_retries} retries.')


def read_datagov_file(path):
    """Read a data.gov.sg download (handles zip-wrapped CSV/GeoJSON)."""
    import zipfile
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            target = next((n for n in names if n.endswith(('.csv', '.geojson', '.json'))), names[0])
            return z.read(target)
    return path.read_bytes()


def get_onemap_token():
    if not ONEMAP_EMAIL or not ONEMAP_PASSWORD:
        return None
    try:
        resp = requests.post(
            'https://www.onemap.gov.sg/api/auth/post/getToken',
            json={'email': ONEMAP_EMAIL, 'password': ONEMAP_PASSWORD},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()['access_token']
    except Exception as e:
        print(f'  WARNING: Could not get OneMap token: {e}')
        return None


def geocode_address(address, cache):
    if address in cache:
        return cache[address]
    try:
        resp = requests.get(
            'https://www.onemap.gov.sg/api/common/elastic/search',
            params={'searchVal': address, 'returnGeom': 'Y', 'getAddrDetails': 'Y', 'pageNum': 1},
            timeout=10,
        )
        results = resp.json().get('results', [])
        if results:
            r = results[0]
            out = {
                'postal'   : r.get('POSTAL', ''),
                'Latitude' : float(r.get('LATITUDE', 0) or 0),
                'Longitude': float(r.get('LONGITUDE', 0) or 0),
            }
        else:
            out = {'postal': '', 'Latitude': 0.0, 'Longitude': 0.0}
    except Exception:
        out = {'postal': '', 'Latitude': 0.0, 'Longitude': 0.0}
    cache[address] = out
    return out


def get_planning_area(lat, lon, token, cache):
    key = f'{lat:.5f},{lon:.5f}'
    if key in cache:
        return cache[key]
    try:
        resp = requests.get(
            'https://www.onemap.gov.sg/api/private/popapi/getPlanningareaName',
            params={'token': token, 'lat': lat, 'lon': lon},
            timeout=10,
        )
        data = resp.json()
        area = data[0].get('pln_area_n', '') if isinstance(data, list) and data else ''
    except Exception:
        area = ''
    cache[key] = area
    return area


def build_balltree(df, lat_col='Latitude', lon_col='Longitude'):
    coords = np.radians(df[[lat_col, lon_col]].values.astype(float))
    return BallTree(coords, metric='haversine')


def nearest_features(flat_df, amenity_df, lat_col='Latitude', lon_col='Longitude', radii_m=None):
    """Return nearest-distance and optional within-radius counts for every flat row."""
    tree = build_balltree(amenity_df, lat_col, lon_col)
    flat_coords = np.radians(flat_df[['Latitude', 'Longitude']].values.astype(float))
    dists, idxs = tree.query(flat_coords, k=1)
    result = {
        'nearest_dist_m': dists[:, 0] * EARTH_RADIUS_M,
        'nearest_idx'   : idxs[:, 0],
    }
    if radii_m:
        for r in radii_m:
            counts = tree.query_radius(flat_coords, r=r / EARTH_RADIUS_M, count_only=True)
            result[f'count_{r}m'] = counts
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD LATEST HDB RESALE DATA
# ═══════════════════════════════════════════════════════════════════════════════

print('\n=== STEP 1: HDB Resale Transactions ===')
HDB_RESALE_ID  = 'd_8b84c4ee58e3cfc0ece0d773c8ca6abc'
HDB_RESALE_CSV = RAW_DIR / 'hdb_resale_2017_onwards.csv'

# Force re-download to get latest data
if HDB_RESALE_CSV.exists():
    HDB_RESALE_CSV.unlink()
    print('  Removed cached file — downloading fresh data...')

datagov_poll_download(HDB_RESALE_ID, HDB_RESALE_CSV, api_key=DATAGOV_API_KEY)

raw_bytes = read_datagov_file(HDB_RESALE_CSV)
df = pd.read_csv(io.BytesIO(raw_bytes))
print(f'  Rows: {len(df):,}  |  Columns: {df.shape[1]}')

# ── Feature engineering ───────────────────────────────────────────────────────
df = df.rename(columns={'month': 'Tranc_YearMonth'})
df['Tranc_Year']  = df['Tranc_YearMonth'].str[:4].astype(int)
df['Tranc_Month'] = df['Tranc_YearMonth'].str[5:7].astype(int)

storey_split = df['storey_range'].str.split(' TO ', expand=True).astype(int)
df['lower']      = storey_split[0]
df['upper']      = storey_split[1]
df['mid']        = (df['lower'] + df['upper']) / 2
df['mid_storey'] = np.ceil(df['mid']).astype(int)

df['full_flat_type']  = df['flat_type'].str.strip() + ' ' + df['flat_model'].str.strip()
df['address']         = df['block'].astype(str).str.strip() + ' ' + df['street_name'].str.strip()
df['floor_area_sqft'] = (df['floor_area_sqm'] * 10.7639).round(2)
df['hdb_age']         = df['Tranc_Year'] - df['lease_commence_date'].astype(int)

# Normalise join keys
df['block']       = df['block'].astype(str).str.strip().str.upper()
df['street_name'] = df['street_name'].str.strip().str.upper()
df['address']     = df['block'] + ' ' + df['street_name']

print(f'  Shape after feature engineering: {df.shape}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: JOIN HDB PROPERTY INFO
# ═══════════════════════════════════════════════════════════════════════════════

print('\n=== STEP 2: HDB Property Info ===')
prop = pd.read_csv(RAW_DIR / 'hdb_property_info.csv')
print(f'  Property info rows: {len(prop):,}')

blk_col    = next(c for c in prop.columns if 'blk' in c.lower())
street_col = next(c for c in prop.columns if 'street' in c.lower())

prop[blk_col]    = prop[blk_col].astype(str).str.strip().str.upper()
prop[street_col] = prop[street_col].astype(str).str.strip().str.upper()

PROP_COLS = [
    blk_col, street_col,
    'max_floor_lvl', 'year_completed',
    'residential', 'commercial', 'market_hawker',
    'multistorey_carpark', 'precinct_pavilion',
    'total_dwelling_units',
    '1room_sold', '2room_sold', '3room_sold', '4room_sold', '5room_sold',
    'exec_sold', 'multigen_sold', 'studio_apartment_sold',
    '1room_rental', '2room_rental', '3room_rental', 'other_room_rental',
]
PROP_COLS = [c for c in PROP_COLS if c in prop.columns]

df = df.merge(
    prop[PROP_COLS],
    left_on  = ['block', 'street_name'],
    right_on = [blk_col, street_col],
    how='left',
).drop(columns=[blk_col, street_col], errors='ignore')

match_pct = df['max_floor_lvl'].notna().mean()
print(f'  Shape after merge: {df.shape}  |  Match rate: {match_pct:.1%}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: GEOCODING (lat/lon + planning area)
# ═══════════════════════════════════════════════════════════════════════════════

print('\n=== STEP 3: Geocoding ===')
GEOCODE_CACHE_FILE  = CACHE_DIR / 'geocode_cache.json'
PLANAREA_CACHE_FILE = CACHE_DIR / 'planning_area_cache.json'

geocode_cache = json.loads(GEOCODE_CACHE_FILE.read_text()) if GEOCODE_CACHE_FILE.exists() else {}
planarea_cache = json.loads(PLANAREA_CACHE_FILE.read_text()) if PLANAREA_CACHE_FILE.exists() else {}

unique_addresses = df['address'].unique()
uncached = [a for a in unique_addresses if a not in geocode_cache]
print(f'  Unique addresses: {len(unique_addresses):,}  |  To geocode: {len(uncached):,}')

for addr in tqdm(uncached, desc='  Geocoding'):
    geocode_address(addr, geocode_cache)
    time.sleep(0.3)

GEOCODE_CACHE_FILE.write_text(json.dumps(geocode_cache, indent=2, ensure_ascii=False))

geo_df = pd.DataFrame.from_dict(geocode_cache, orient='index').reset_index()
geo_df = geo_df.rename(columns={'index': 'address'})
df = df.merge(geo_df, on='address', how='left')
valid = df['Latitude'].notna() & (df['Latitude'] != 0)
print(f'  Valid coordinates: {valid.sum():,} / {len(df):,} ({valid.mean():.1%})')

# Planning area
onemap_token = get_onemap_token()
if onemap_token:
    unique_coords = df[df['Latitude'] != 0][['Latitude', 'Longitude']].drop_duplicates()
    uncached_coords = [
        (r['Latitude'], r['Longitude'])
        for _, r in unique_coords.iterrows()
        if f"{r['Latitude']:.5f},{r['Longitude']:.5f}" not in planarea_cache
    ]
    print(f'  Coordinates to resolve for planning area: {len(uncached_coords):,}')
    for lat, lon in tqdm(uncached_coords, desc='  Planning area'):
        get_planning_area(lat, lon, onemap_token, planarea_cache)
        time.sleep(0.3)
    PLANAREA_CACHE_FILE.write_text(json.dumps(planarea_cache, indent=2))
    df['planning_area'] = df.apply(
        lambda row: planarea_cache.get(f"{row['Latitude']:.5f},{row['Longitude']:.5f}", ''),
        axis=1,
    )
    print('  planning_area column added.')
else:
    df['planning_area'] = ''
    print('  WARNING: No OneMap token — planning_area will be empty.')

# Checkpoint
df.to_csv(RAW_DIR / 'hdb_geocoded.csv', index=False)
print(f'  Checkpoint saved -> hdb_geocoded.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: PROXIMITY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print('\n=== STEP 4: Proximity Feature Engineering ===')

# Work only on rows with valid coordinates
df_geo = df[df['Latitude'].notna() & (df['Latitude'] != 0)].copy()
print(f'  Flats with valid coordinates: {len(df_geo):,} / {len(df):,}')

# ── Hawker Centres ────────────────────────────────────────────────────────────
print('  Loading hawker centres...')
hawker_df = pd.read_csv(RAW_DIR / 'hawker_centres_processed.csv')
hawker_df = hawker_df.dropna(subset=['Latitude', 'Longitude'])
hawker_df = hawker_df[(hawker_df['Latitude'] != 0) & (hawker_df['Longitude'] != 0)]
print(f'  Hawker centres: {len(hawker_df)}')

hf = nearest_features(df_geo, hawker_df, radii_m=[500, 1000, 2000])
df_geo['Hawker_Nearest_Distance'] = hf['nearest_dist_m']
df_geo['Hawker_Within_500m']      = hf['count_500m']
df_geo['Hawker_Within_1km']       = hf['count_1000m']
df_geo['Hawker_Within_2km']       = hf['count_2000m']
df_geo['hawker_food_stalls']      = hawker_df['food_stalls'].iloc[hf['nearest_idx']].values
df_geo['hawker_market_stalls']    = hawker_df['market_stalls'].iloc[hf['nearest_idx']].values
print('  Hawker features done.')

# ── MRT Stations ──────────────────────────────────────────────────────────────
print('  Loading MRT stations...')
mrt_df = pd.read_csv(RAW_DIR / 'mrt_stations.csv')
mrt_df = mrt_df.dropna(subset=['Latitude', 'Longitude'])

mf = nearest_features(df_geo, mrt_df)
df_geo['mrt_nearest_distance'] = mf['nearest_dist_m']
df_geo['mrt_name']             = mrt_df['mrt_name'].iloc[mf['nearest_idx']].values
df_geo['mrt_interchange']      = mrt_df['mrt_interchange'].iloc[mf['nearest_idx']].values
df_geo['bus_interchange']      = mrt_df['bus_interchange'].iloc[mf['nearest_idx']].values
df_geo['mrt_latitude']         = mrt_df['Latitude'].iloc[mf['nearest_idx']].values
df_geo['mrt_longitude']        = mrt_df['Longitude'].iloc[mf['nearest_idx']].values
print('  MRT features done.')

# ── Bus Stops ─────────────────────────────────────────────────────────────────
print('  Loading bus stops...')
bus_df = pd.read_csv(RAW_DIR / 'bus_stops.csv')
bus_df = bus_df.dropna(subset=['Latitude', 'Longitude'])
bus_df = bus_df[(bus_df['Latitude'] != 0) & (bus_df['Longitude'] != 0)]
print(f'  Bus stops: {len(bus_df):,}')

bf = nearest_features(df_geo, bus_df)
df_geo['bus_stop_nearest_distance'] = bf['nearest_dist_m']
df_geo['bus_stop_name']             = bus_df['bus_stop_name'].iloc[bf['nearest_idx']].values
df_geo['bus_stop_latitude']         = bus_df['Latitude'].iloc[bf['nearest_idx']].values
df_geo['bus_stop_longitude']        = bus_df['Longitude'].iloc[bf['nearest_idx']].values
print('  Bus stop features done.')

# ── Shopping Malls ────────────────────────────────────────────────────────────
print('  Loading shopping malls...')
mall_df = pd.read_csv(RAW_DIR / 'shopping_malls.csv')
mall_df = mall_df.dropna(subset=['Latitude', 'Longitude'])
mall_df = mall_df[(mall_df['Latitude'] != 0) & (mall_df['Longitude'] != 0)]
print(f'  Malls: {len(mall_df)}')

mlf = nearest_features(df_geo, mall_df, radii_m=[500, 1000, 2000])
df_geo['Mall_Nearest_Distance'] = mlf['nearest_dist_m']
df_geo['Mall_Within_500m']      = mlf['count_500m']
df_geo['Mall_Within_1km']       = mlf['count_1000m']
df_geo['Mall_Within_2km']       = mlf['count_2000m']
print('  Mall features done.')

# ── Primary Schools ───────────────────────────────────────────────────────────
print('  Loading primary schools...')
pri_sch_df = pd.read_csv(RAW_DIR / 'primary_schools.csv')
pri_sch_df = pri_sch_df.dropna(subset=['Latitude', 'Longitude'])
pri_sch_df = pri_sch_df[pri_sch_df['Latitude'] != 0].copy()
name_col_pri = next((c for c in pri_sch_df.columns if 'school_name' in c.lower()), None)
print(f'  Primary schools: {len(pri_sch_df)}')

if len(pri_sch_df) > 0:
    pf = nearest_features(df_geo, pri_sch_df)
    df_geo['pri_sch_nearest_distance'] = pf['nearest_dist_m']
    df_geo['pri_sch_name']             = pri_sch_df[name_col_pri].iloc[pf['nearest_idx']].values if name_col_pri else ''
    df_geo['pri_sch_affiliation']      = pri_sch_df['pri_sch_affiliation'].iloc[pf['nearest_idx']].values
    df_geo['pri_sch_latitude']         = pri_sch_df['Latitude'].iloc[pf['nearest_idx']].values
    df_geo['pri_sch_longitude']        = pri_sch_df['Longitude'].iloc[pf['nearest_idx']].values
    print('  Primary school features done.')

# ── Secondary Schools ─────────────────────────────────────────────────────────
print('  Loading secondary schools...')
sec_sch_df = pd.read_csv(RAW_DIR / 'secondary_schools.csv')
sec_sch_df = sec_sch_df.dropna(subset=['Latitude', 'Longitude'])
sec_sch_df = sec_sch_df[sec_sch_df['Latitude'] != 0].copy()
name_col_sec = next((c for c in sec_sch_df.columns if 'school_name' in c.lower()), None)
print(f'  Secondary schools: {len(sec_sch_df)}')

if len(sec_sch_df) > 0:
    sf = nearest_features(df_geo, sec_sch_df)
    df_geo['sec_sch_nearest_dist'] = sf['nearest_dist_m']
    df_geo['sec_sch_name']         = sec_sch_df[name_col_sec].iloc[sf['nearest_idx']].values if name_col_sec else ''
    df_geo['affiliation']          = sec_sch_df['affiliation'].iloc[sf['nearest_idx']].values
    df_geo['cutoff_point']         = sec_sch_df['cutoff_point'].iloc[sf['nearest_idx']].values if 'cutoff_point' in sec_sch_df.columns else np.nan
    df_geo['sec_sch_latitude']     = sec_sch_df['Latitude'].iloc[sf['nearest_idx']].values
    df_geo['sec_sch_longitude']    = sec_sch_df['Longitude'].iloc[sf['nearest_idx']].values
    print('  Secondary school features done.')

print(f'\n  Final shape before column selection: {df_geo.shape}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SELECT FINAL COLUMNS AND EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

print('\n=== STEP 5: Export ===')

FINAL_COLS = [
    'Tranc_YearMonth', 'Tranc_Year', 'Tranc_Month',
    'town', 'flat_type', 'block', 'street_name',
    'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date',
    'resale_price', 'mid_storey',
    'max_floor_lvl', 'year_completed',
    'residential', 'commercial', 'market_hawker',
    'multistorey_carpark', 'precinct_pavilion', 'total_dwelling_units',
    '1room_sold', '2room_sold', '3room_sold', '4room_sold', '5room_sold',
    'exec_sold', 'multigen_sold', 'studio_apartment_sold',
    '1room_rental', '2room_rental', '3room_rental', 'other_room_rental',
    'postal', 'Latitude', 'Longitude', 'planning_area',
    'Mall_Nearest_Distance', 'Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
    'Hawker_Nearest_Distance', 'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km',
    'hawker_food_stalls', 'hawker_market_stalls',
    'mrt_nearest_distance', 'mrt_name', 'bus_interchange', 'mrt_interchange',
    'mrt_latitude', 'mrt_longitude',
    'bus_stop_nearest_distance', 'bus_stop_name', 'bus_stop_latitude', 'bus_stop_longitude',
    'pri_sch_nearest_distance', 'pri_sch_name', 'pri_sch_affiliation',
    'pri_sch_latitude', 'pri_sch_longitude',
    'sec_sch_nearest_dist', 'sec_sch_name', 'cutoff_point', 'affiliation',
    'sec_sch_latitude', 'sec_sch_longitude',
]

# Keep only columns that exist in this run
final_cols = [c for c in FINAL_COLS if c in df_geo.columns]
missing    = [c for c in FINAL_COLS if c not in df_geo.columns]
if missing:
    print(f'  WARNING: missing columns (will be omitted): {missing}')

df_final = df_geo[final_cols].copy()

OUTPUT_PATH = DATA_DIR / 'hdb_resale_complete.csv'
df_final.to_csv(OUTPUT_PATH, index=False)

print(f'  Saved -> {OUTPUT_PATH}')
print(f'  Rows   : {len(df_final):,}')
print(f'  Columns: {df_final.shape[1]}')

# ── Coverage report ───────────────────────────────────────────────────────────
print('\n=== Column coverage (non-null %) ===')
coverage = df_final.notna().mean().sort_values(ascending=False)
print(coverage.to_string())

print('\nDone.')
