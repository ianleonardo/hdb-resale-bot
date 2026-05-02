import os
import json
import time
import zipfile
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
CACHE_DIR = DATA_DIR / 'cache'

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / '.env')
DATAGOV_API_KEY = os.getenv('DATAGOV_API_KEY', '')

def datagov_poll_download(dataset_id, save_path, api_key=None, max_retries=60, sleep_sec=5):
    save_path = Path(save_path)
    if save_path.exists():
        print(f'  Cached: {save_path.name}')
        return save_path

    url = f'https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download'
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
            print(f'  Saved -> {save_path}')
            return save_path
        print(f'  Preparing ({attempt}/{max_retries}) ...')
        time.sleep(sleep_sec)

    raise TimeoutError(f'Dataset {dataset_id} not ready after {max_retries} retries.')

def read_datagov_file(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            target = next((n for n in names if n.endswith(('.csv', '.geojson', '.json'))), names[0])
            return z.read(target)
    return path.read_bytes()

GEOCODE_CACHE_FILE = CACHE_DIR / 'geocode_cache.json'

def load_geocode_cache():
    return json.loads(GEOCODE_CACHE_FILE.read_text()) if GEOCODE_CACHE_FILE.exists() else {}

def save_geocode_cache(cache):
    GEOCODE_CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))

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
                'postal': r.get('POSTAL', ''),
                'Latitude': float(r.get('LATITUDE', 0) or 0),
                'Longitude': float(r.get('LONGITUDE', 0) or 0),
            }
        else:
            out = {'postal': '', 'Latitude': 0.0, 'Longitude': 0.0}
    except Exception:
        out = {'postal': '', 'Latitude': 0.0, 'Longitude': 0.0}
    cache[address] = out
    return out
