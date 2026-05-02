import json
import numpy as np
import pandas as pd
from utils import RAW_DIR, DATAGOV_API_KEY, datagov_poll_download, read_datagov_file

def main():
    HAWKER_ID   = 'd_4a086da0a5553be1d89383cd90d07ecd'
    HAWKER_FILE = RAW_DIR / 'hawker_centres.geojson'

    datagov_poll_download(HAWKER_ID, HAWKER_FILE, api_key=DATAGOV_API_KEY)

    raw_bytes = read_datagov_file(HAWKER_FILE)
    hawker_geo = json.loads(raw_bytes)

    hawker_rows = []
    for feat in hawker_geo.get('features', []):
        props = feat.get('properties', {})
        geom  = feat.get('geometry', {})
        if not geom:
            continue
        gtype = geom.get('type', '')
        coords = geom.get('coordinates', [])
        
        if gtype == 'Point':
            lon, lat = coords[0], coords[1]
        elif gtype == 'Polygon':
            pts = coords[0]
            lon = np.mean([p[0] for p in pts])
            lat = np.mean([p[1] for p in pts])
        elif gtype == 'MultiPolygon':
            pts = coords[0][0]
            lon = np.mean([p[0] for p in pts])
            lat = np.mean([p[1] for p in pts])
        else:
            continue
            
        hawker_rows.append({
            'name'         : props.get('name', ''),
            'Latitude'     : float(lat),
            'Longitude'    : float(lon),
            'food_stalls'  : int(props.get('no_of_food_stalls', 0) or 0),
            'market_stalls': int(props.get('no_of_market_stalls', 0) or 0),
        })

    hawker_df = pd.DataFrame(hawker_rows)
    hawker_df.to_csv(RAW_DIR / 'hawker_centres_processed.csv', index=False)
    print(f'Hawker centres: {len(hawker_df)}')

if __name__ == '__main__':
    main()
