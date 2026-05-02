import json
import pandas as pd
from utils import RAW_DIR, DATAGOV_API_KEY, datagov_poll_download, read_datagov_file

def main():
    BUS_FILE = RAW_DIR / 'bus_stops.geojson'
    BUS_CSV  = RAW_DIR / 'bus_stops.csv'
    BUS_STOP_ID = 'd_3f172c6feb3f4f92a2f47d93eed2908a'

    if not BUS_FILE.exists():
        datagov_poll_download(BUS_STOP_ID, BUS_FILE, api_key=DATAGOV_API_KEY)

    raw_bytes = read_datagov_file(BUS_FILE)
    data = json.loads(raw_bytes)
    rows = []
    for f in data.get('features', []):
        props = f.get('properties', {})
        geom  = f.get('geometry', {})
        if geom.get('type') == 'Point':
            lon, lat = geom.get('coordinates')
            rows.append({
                'bus_stop_code': props.get('BUS_STOP_NUM', ''),
                'bus_stop_name': props.get('BUS_STOP_NUM', ''),
                'Latitude'     : lat,
                'Longitude'    : lon
            })
    bus_df = pd.DataFrame(rows)
    bus_df = bus_df.dropna(subset=['Latitude', 'Longitude'])
    bus_df = bus_df[(bus_df['Latitude'] != 0) & (bus_df['Longitude'] != 0)]
    bus_df.to_csv(BUS_CSV, index=False)
    print(f'Bus stops with valid coordinates: {len(bus_df):,}')

if __name__ == '__main__':
    main()
