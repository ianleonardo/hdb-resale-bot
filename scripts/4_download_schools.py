import io
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import RAW_DIR, DATAGOV_API_KEY, datagov_poll_download, read_datagov_file, load_geocode_cache, save_geocode_cache, geocode_address

def main():
    SCHOOL_ID   = 'd_688b934f82c1059ed0a6993d2a829089'
    SCHOOL_FILE = RAW_DIR / 'schools.csv'

    datagov_poll_download(SCHOOL_ID, SCHOOL_FILE, api_key=DATAGOV_API_KEY)

    sch_raw = pd.read_csv(io.BytesIO(read_datagov_file(SCHOOL_FILE)))
    print(f'Schools loaded: {len(sch_raw)}')

    cache = load_geocode_cache()
    postal_col = next((c for c in sch_raw.columns if 'postal' in c.lower() or 'zip' in c.lower()), None)

    if postal_col:
        latitudes = []
        longitudes = []
        for postal in tqdm(sch_raw[postal_col], desc='Geocoding schools'):
            postal_str = str(postal).strip().zfill(6)
            if postal_str and postal_str != '000000':
                res = geocode_address(postal_str, cache)
                if postal_str not in cache:
                    time.sleep(0.3)
                latitudes.append(res.get('Latitude', 0.0))
                longitudes.append(res.get('Longitude', 0.0))
            else:
                latitudes.append(0.0)
                longitudes.append(0.0)
                
        save_geocode_cache(cache)
        sch_raw['Latitude'] = latitudes
        sch_raw['Longitude'] = longitudes
    else:
        print('WARNING: No postal code column found. Schools will not be geocoded.')
        sch_raw['Latitude']  = 0.0
        sch_raw['Longitude'] = 0.0

    sch_raw.to_csv(RAW_DIR / 'schools_geocoded.csv', index=False)

    level_col = next((c for c in sch_raw.columns if 'mainlevel' in c.lower()), None)
    name_col  = next((c for c in sch_raw.columns if 'school_name' in c.lower() or c.lower() == 'name'), None)

    if level_col:
        pri_sch_df = sch_raw[sch_raw[level_col].str.contains('PRIMARY',   na=False, case=False)].copy()
        sec_sch_df = sch_raw[sch_raw[level_col].str.contains('SECONDARY', na=False, case=False)].copy()
    else:
        pri_sch_df = sch_raw.copy()
        sec_sch_df = sch_raw.copy()

    sap_col = 'sap_ind' if 'sap_ind' in sch_raw.columns else None
    pri_sch_df['pri_sch_affiliation'] = (pri_sch_df[sap_col] == 'Y').astype(int) if sap_col else 0
    sec_sch_df['affiliation']         = (sec_sch_df[sap_col] == 'Y').astype(int) if sap_col else 0

    pri_sch_df['vacancy']     = np.nan
    sec_sch_df['cutoff_point'] = np.nan

    pri_sch_df = pri_sch_df[pri_sch_df['Latitude'] != 0].copy()
    sec_sch_df = sec_sch_df[sec_sch_df['Latitude'] != 0].copy()

    pri_sch_df.to_csv(RAW_DIR / 'primary_schools.csv', index=False)
    sec_sch_df.to_csv(RAW_DIR / 'secondary_schools.csv', index=False)
    print(f'Primary schools (geocoded): {len(pri_sch_df)}')
    print(f'Secondary schools (geocoded): {len(sec_sch_df)}')

if __name__ == '__main__':
    main()
