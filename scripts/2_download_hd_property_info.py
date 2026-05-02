import io
import time
import pandas as pd
from tqdm.auto import tqdm
from utils import RAW_DIR, DATAGOV_API_KEY, datagov_poll_download, read_datagov_file, load_geocode_cache, save_geocode_cache, geocode_address

def main():
    HDB_PROP_ID  = 'd_17f5382f26140b1fdae0ba2ef6239d2f'
    HDB_PROP_CSV = RAW_DIR / 'hdb_property_info.csv'

    datagov_poll_download(HDB_PROP_ID, HDB_PROP_CSV, api_key=DATAGOV_API_KEY)

    prop_bytes = read_datagov_file(HDB_PROP_CSV)
    prop = pd.read_csv(io.BytesIO(prop_bytes))
    print(f'Property info rows: {len(prop):,}  |  Columns: {prop.shape[1]}')

    blk_col    = next(c for c in prop.columns if 'blk' in c.lower())
    street_col = next(c for c in prop.columns if 'street' in c.lower())

    prop[blk_col]    = prop[blk_col].astype(str).str.strip().str.upper()
    prop[street_col] = prop[street_col].astype(str).str.strip().str.upper()
    
    prop['address'] = prop[blk_col] + ' ' + prop[street_col]
    
    cache = load_geocode_cache()
    
    latitudes = []
    longitudes = []
    postals = []
    
    print('Geocoding HDB Property Info...')
    for addr in tqdm(prop['address']):
        res = geocode_address(addr, cache)
        if addr not in cache:
            time.sleep(0.3)
        latitudes.append(res.get('Latitude', 0.0))
        longitudes.append(res.get('Longitude', 0.0))
        postals.append(res.get('postal', ''))
        
    save_geocode_cache(cache)
    
    prop['Latitude'] = latitudes
    prop['Longitude'] = longitudes
    prop['postal'] = postals
    
    prop.to_csv(RAW_DIR / 'hdb_property_info_geocoded.csv', index=False)
    print('Saved hdb_property_info_geocoded.csv')

if __name__ == '__main__':
    main()
