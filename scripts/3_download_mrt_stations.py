import json
import pandas as pd
from utils import RAW_DIR, DATAGOV_API_KEY, datagov_poll_download, read_datagov_file

def main():
    MRT_FILE   = RAW_DIR / 'mrt_stations.csv'
    EXITS_FILE = RAW_DIR / 'train_station_exits.geojson'
    MRT_EXITS_ID = 'd_b39d3a0871985372d7e1637193335da5'

    if not EXITS_FILE.exists():
        datagov_poll_download(MRT_EXITS_ID, EXITS_FILE, api_key=DATAGOV_API_KEY)

    raw_bytes = read_datagov_file(EXITS_FILE)
    data = json.loads(raw_bytes)
    rows = []
    for f in data.get('features', []):
        props = f.get('properties', {})
        geom  = f.get('geometry', {})
        if geom.get('type') == 'Point':
            lon, lat = geom.get('coordinates')
            rows.append({
                'StationName': props.get('STATION_NA', ''),
                'ExitCode'   : props.get('EXIT_CODE', ''),
                'Latitude'   : lat,
                'Longitude'  : lon
            })
    exits = pd.DataFrame(rows)

    mrt_df = (
        exits
        .groupby('StationName', as_index=False)
        .agg(mrt_name=('StationName', 'first'),
             Latitude =('Latitude',    'mean'),
             Longitude=('Longitude',   'mean'))
    )

    mrt_df['mrt_interchange'] = 0
    mrt_df.loc[mrt_df['mrt_name'].str.contains('/'), 'mrt_interchange'] = 1

    BUS_INTERCHANGE_STATIONS = {
        'Ang Mo Kio', 'Bedok', 'Bishan', 'Boon Lay', 'Bukit Merah',
        'Choa Chu Kang', 'Clementi', 'Eunos', 'Hougang', 'Joo Koon',
        'Jurong East', 'Pasir Ris', 'Punggol', 'Serangoon', 'Sembawang',
        'Sengkang', 'Tampines', 'Toa Payoh', 'Woodlands', 'Yishun',
        'Buona Vista',
    }
    mrt_df['bus_interchange'] = mrt_df['mrt_name'].apply(
        lambda n: any(b.lower() in n.lower() for b in BUS_INTERCHANGE_STATIONS)
    ).astype(int)

    mrt_df.to_csv(MRT_FILE, index=False)
    print(f'MRT stations: {len(mrt_df)}  |  '
          f'Interchanges: {mrt_df["mrt_interchange"].sum()}  |  '
          f'Bus interchanges: {mrt_df["bus_interchange"].sum()}')

if __name__ == '__main__':
    main()
