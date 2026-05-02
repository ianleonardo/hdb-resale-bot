import pandas as pd
import requests
from utils import RAW_DIR

def main():
    MALL_FILE = RAW_DIR / 'shopping_malls.csv'

    if not MALL_FILE.exists():
        STATIC_MALL_URL = 'https://raw.githubusercontent.com/ValaryLim/Mall-Coordinates-Web-Scraper/master/mall_coordinates_updated.csv'
        print('Querying OpenStreetMap for mall locations (BBox optimized) ...')
        
        overpass_query = """
        [out:json][timeout:30];
        (
          node["shop"="mall"](1.13,103.59,1.48,104.04);
          way["shop"="mall"](1.13,103.59,1.48,104.04);
          relation["shop"="mall"](1.13,103.59,1.48,104.04);
        );
        out center;
        """
        
        malls = []
        success = False
        
        for api_url in ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter']:
            try:
                print(f'  Trying {api_url} ...')
                resp = requests.post(api_url, data={'data': overpass_query}, timeout=45)
                if resp.status_code == 200:
                    elements = resp.json().get('elements', [])
                    for e in elements:
                        lat = e.get('lat') or (e.get('center') or {}).get('lat')
                        lon = e.get('lon') or (e.get('center') or {}).get('lon')
                        if lat and lon:
                            malls.append({
                                'mall_name': e.get('tags', {}).get('name', 'Unknown'),
                                'Latitude' : float(lat),
                                'Longitude': float(lon),
                            })
                    success = True
                    break
                else:
                    print(f'  Failed ({resp.status_code})')
            except Exception as e:
                print(f'  Error: {e}')
        
        if not success:
            print('OSM Overpass timed out or failed. Falling back to static community list ...')
            try:
                fallback_df = pd.read_csv(STATIC_MALL_URL)
                for _, row in fallback_df.iterrows():
                    malls.append({
                        'mall_name': row.get('name', 'Unknown'),
                        'Latitude' : float(row.get('latitude', 0)),
                        'Longitude': float(row.get('longitude', 0)),
                    })
                success = True
            except Exception as e:
                print(f'  Fallback failed: {e}')
        
        mall_df = pd.DataFrame(malls).drop_duplicates(subset=['mall_name'])
        mall_df.to_csv(MALL_FILE, index=False)
        print(f'Malls found: {len(mall_df)}')
    else:
        mall_df = pd.read_csv(MALL_FILE)
        print(f'Loaded {len(mall_df)} malls from cache.')

if __name__ == '__main__':
    main()
