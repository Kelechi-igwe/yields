"""
yields_data_io.py
=================
Input-data extraction functions for the YIELDS pipeline
(Yield Inference from Earth-observation and Land Data Systems).

APIs covered
------------
1. Kansas Mesonet  — daily weather (temperature, radiation, precip)
2. OpenET          — 30 m satellite-based ET (POST endpoint)
3. USDA SDA        — gSSURGO soil horizon properties

"""


import os
import io
import warnings
import numpy as np
import pandas as pd
import polars as pl
import yaml
import requests
from pathlib import Path
import datetime as dt
from datetime import timedelta
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
# from setuptools.config.pyprojecttoml import load_file

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
config_path = ROOT / 'config.yaml'

with open(config_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)


SEED = config['data_specs']['seed']


# ============================================================
# 1. WEATHER — Kansas Mesonet API
# ============================================================


def fetch_weather_from_mesonet(station: str, start: pd.Timestamp, end: pd.Timestamp,
                                year_filter: int = None,
                                planting_date: str = None,
                                config_file: dict = config) -> tuple[pl.DataFrame, list[dict]]:
    """
    Pull daily weather from the Kansas Mesonet REST API and return a
    clean DataFrame ready for the crop model.
    :param station: Mesonet station code
    :param start: query start date 'YYYY-MM-DD'
    :param end: query end date 'YYYY-MM-DD'
    :param year_filter: if given, keep only rows from this year
    :param planting_date: if given ('YYYY-MM-DD'), trim rows before this date
    :param config_file: custom file
    :return: cleaned daily weather, same rows as list-of-dicts for the crop model functn
    """

    MESONET_URL = config_file['url_inputs']['MESONET_URL']


    fmt = config_file['data_specs']['date_format']

    t_start = pd.to_datetime(start).strftime(fmt)
    t_end   = pd.to_datetime(end).strftime(fmt)

    vars = config_file['data_specs']['mesonet_vars'] # gets vars as list
    variables = ",".join(vars)

    url = (
        f"{MESONET_URL}?stn={station}&int=day"
        f"&t_start={t_start}&t_end={t_end}&vars={variables}"
    ).replace(" ", "%20")

    try:
        df = pl.read_csv(url, null_values=["M"], try_parse_dates=True)
    except Exception as exc:
        raise RuntimeError(f"Mesonet fetch failed for station '{station}': {exc}") from exc

    # rename and drop station
    df = df.drop('STATION')

    # pick the list of old:new col names from config file
    column_maps = config_file.get('column_mapping', {})
    df = df.rename(column_maps)
    # print(df.columns)

    df = df.with_columns(
        pl.col('DATE').dt.year().alias('year')
    )
    # df["year"] = df["DATE"].dt.year

    if year_filter is not None:
        df = df.filter(pl.col('year') == year_filter)
    df = df.sort("DATE")

    if planting_date is not None:
        cutoff = pd.to_datetime(planting_date)
        df = df.filter(pl.col('DATE') >= cutoff)
        # df = df[df["DATE"] >= cutoff].reset_index(drop=True)
    # df = df.to_pandas()
    weather_dict = df.to_dicts()   # fast list-of-dicts for daily step()

    print(
        f"Weather loaded: {len(weather_dict)} days  "
        f"({df['DATE'].dt.date()[0]} → {df['DATE'].dt.date()[-1]})"
    )
    return df, weather_dict


# ============================================================
# 2. ET — OpenET Ensemble API  (POST /raster/timeseries/point)
# ============================================================
# Fixed vs. original notebook:
#   • OPENET_BASE_URL is base-only; path is appended in the function
#     (original had the full path baked in AND appended it again → 404)
#   • Uses POST + JSON body, not GET + query params
#   • Model name is "Ensemble", not "ensemble_mean" (invalid)
#   • Response handles both bare list and {"data": [...]} dict


def build_grid_centroids(bbox: dict, n_rows: int, n_cols: int):

    """
    :param bbox: with keys min_lon, max_lon, min_lat, max_lat  (WGS84)
    :param n_rows: number of grid rows  (latitude direction, N→S)
    :param n_cols: number of grid cols  (longitude direction, W→E)

    :return: lats (n_rows, n_cols), lons (n_rows, n_cols)
    """

    lat_centers = np.linspace(bbox["max_lat"], bbox["min_lat"], n_rows)
    lon_centers = np.linspace(bbox["min_lon"], bbox["max_lon"], n_cols, dtype=np.float64)
    lats, lons = np.meshgrid(lat_centers, lon_centers, indexing='ij', copy=False)
    return lats, lons



def query_openet_timeseries(lat: float, lon: float,
                             start_date: str, end_date: str,
                             api_key: str,
                             session: requests.Session = None,
                             model: str = "Ensemble",
                             interval: str = "daily",
                             reference_et: str = "gridMET",
                             units: str = "mm",
                             config_file: dict = config) -> tuple[np.ndarray, np.ndarray]:

    """
    :param lat: WGS84 coordinates
    :param lon: WGS84 coordinates
    :param start_date: 'YYYY-MM-DD'
    :param end_date: 'YYYY-MM-DD'
    :param api_key: OpenET key
    :param session: tcp connect
    :param model: "Ensemble" | "SSEBop" | "SIMS" | "DisALEXI" | "PTJPL" | "eeMETRIC" | "geeSEBAL"
    :param interval: "daily" or "monthly"
    :param reference_et: "gridMET" (CONUS) or "CIMIS" (California)
    :param units: "mm" or "in"
    :param config_file: custom file

    :return np.ndarray tuple

    """

    # should accept active request sessions
    # should nt spin up new tcp handshake everytime
    # fallback to single request client if no session provided
    client = session if session is not None else requests

    OPENET_BASE_URL = config_file['url_inputs']['OPENET_BASE_URL']

    endpoint = f"{OPENET_BASE_URL}/raster/timeseries/point"
    headers  = {"Authorization": api_key, "Content-Type": "application/json"}
    body     = {
        "date_range":   [start_date, end_date],
        "interval":     interval,
        "geometry":     [lon, lat],     # GeoJSON: lon first
        "model":        model,
        "variable":     "ET",
        "reference_et": reference_et,
        "units":        units,
        "file_format":  "JSON",
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30) # I changed timeout to 30 for aggressive timeout
        resp.raise_for_status()
        data = resp.json()

        # Response is either a bare list or {"data": [...]}
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("data", [])
        else:
            records = []

        if not records:
            warnings.warn(f"OpenET: empty response for ({lat:.4f}, {lon:.4f})")
            return None

        # moving dates and values to np.ndarray for efficiency
        dates = np.array([r["time"] for r in records], dtype="datetime64[D]")
        values = np.array([r["et"] if r["et"] is not None else np.nan for r in records], dtype=np.float64)

        # dates  = pd.to_datetime([r["time"] for r in records])
        # values = np.array(
        #     [r["et"] if r["et"] is not None else np.nan for r in records],
        #     dtype=float,
        # )
        return dates, values

    except requests.exceptions.Timeout:
        warnings.warn(f"OpenET: request timed out for ({lat:.4f}, {lon:.4f})")
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code
        body_txt = getattr(exc.response, "text", "")[:200]
        warnings.warn(f"OpenET: HTTP {code} for ({lat:.4f}, {lon:.4f}) — {body_txt}")
    except Exception as exc:
        warnings.warn(f"OpenET: query failed for ({lat:.4f}, {lon:.4f}) — {exc}")
    return None, None




def fetch_et_stack_from_openet(grid_lats, grid_lons,
                                start_date: str, end_date: str,
                                api_key: str,
                                season_length: int) -> tuple[np.ndarray, bool]:

    n_rows, n_cols = grid_lats.shape
    ET_stack = np.full((season_length, n_rows, n_cols), np.nan, dtype=np.float64)

    # do this convertion before going into the loop
    clean_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    clean_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    date_idx  = pd.date_range(start_date, periods=season_length, freq="D")
    any_success = False

    # est connection
    with requests.Session() as session:
        for i in range(n_rows):
            for j in range(n_cols):
                lat = float(grid_lats[i, j])
                lon = float(grid_lons[i, j])
                print(f"  [OpenET] cell ({i},{j})  lat={lat:.4f}  lon={lon:.4f}")

                dates, values = query_openet_timeseries(
                    lat=lat, lon=lon,
                    start_date=clean_start,
                    end_date=clean_end,
                    api_key=api_key,
                    session=session,
                )
                # if s is not None and len(s) > 0:
                if values is not None and len(values) > 0:
                    s = pd.Series(values, index=pd.to_datetime(dates))
                    s = s.reindex(date_idx).interpolate(method="linear", limit=5)

                    valid_len = min(len(s), season_length)
                    ET_stack[:valid_len, i, j] = s.values[:valid_len]

                    # ET_stack[:len(s), i, j] = s.values[:season_length]
                    any_success = True

    return ET_stack, any_success


def generate_et_stack_synthetic(n_rows: int, n_cols: int,
                                 season_length: int, seed: int = SEED, config_file: dict = config) -> np.ndarray:

    rng   = np.random.default_rng(seed)
    days  = np.arange(season_length, dtype=np.float64)
    curve = 4.0 * np.exp(-0.5 * ((days - 90) / 40) ** 2) + 0.5
    scale = rng.lognormal(0.0, 0.15, (n_rows, n_cols))
    noise = rng.normal(1.0, 0.05, (season_length, n_rows, n_cols))
    return np.clip(curve[:, None, None] * scale[None, :, :] * noise, 0.1, None)



# ============================================================
# 3. SOIL — USDA gSSURGO via Soil Data Access (SDA) API
# ============================================================

def query_sda(sql_query: str, expected_columns: list[str],
              debug: bool = False, session: requests.Session = None, config_file: dict = config) -> pd.DataFrame | None:

    SDA_URL = config_file['url_inputs']['SDA_URL']

    payload = {"query": sql_query, "format": "JSON"}
    client = session if session is not None else requests

    try:
        resp = client.post(SDA_URL, data=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        if debug:
            print("SDA response keys :", list(result.keys()))
            print("SDA response (head):", str(result)[:1000])

        table = result.get("Table")
        if not table:
            warnings.warn("SDA returned no 'Table'. Bounding box might be outside survey bounds.")
            return None

        # Detect and skip optional header row dynamically
        first = [str(x).strip().lower() for x in table[0]]
        rows = table[1:] if first == [c.lower() for c in expected_columns] else table

        if not rows:
            warnings.warn("SDA returned an empty response. The SQL query may have returned 0 records.")
            return None

        # df = pl.read_csv(io.StringIO(result), null_values=["", "NA", "M"])
        #
        # if df.is_empty():
        #     warnings.warn("SDA query returned a valid schema but zero data rows.")
        #     return None

        # df = df.select(expected_columns)
        df = pl.DataFrame(rows, schema=expected_columns, orient='row')
        # Detect optional header row
        # first = [str(x).strip().lower() for x in table[0]]
        # rows  = table[1:] if first == [c.lower() for c in expected_columns] else table
        #
        # if not rows:
        #     warnings.warn("SDA query returned schema but zero data rows.")
        #     return None

        # return pd.DataFrame(rows, columns=expected_columns)
        return df



    except requests.exceptions.HTTPError as exc:
        warnings.warn(f"SDA HTTP error: {exc}")
    except Exception as exc:
        warnings.warn(f"SDA query failed: {exc}")
    return None




def fetch_ssurgo_soil_for_bbox(bbox: dict, session: requests.Session = None,
                               debug: bool = False, config_file: dict = config) -> pl.DataFrame | None:

    wkt = (
        f"POLYGON(({bbox['min_lon']} {bbox['min_lat']}, "
        f"{bbox['max_lon']} {bbox['min_lat']}, "
        f"{bbox['max_lon']} {bbox['max_lat']}, "
        f"{bbox['min_lon']} {bbox['max_lat']}, "
        f"{bbox['min_lon']} {bbox['min_lat']}))"
    )

    cols = [
        "mukey", "muname", "cokey", "hzdept_r", "hzdepb_r",
        "wthirdbar_r", "wfifteenbar_r", "sandtotal_r", "claytotal_r", "om_r",
    ]

    sql_temp = config_file['queries']['gssurgo_wkt']
    sql = sql_temp.format(wkt_geom=wkt)

    print("[SDA] Querying gSSURGO for soil horizons...")
    df = query_sda(sql, cols, debug=debug, session=session)
    if df is None:
        return None

    numeric_cols = ["hzdept_r","hzdepb_r","wthirdbar_r","wfifteenbar_r",
               "sandtotal_r","claytotal_r","om_r"]


    n_before = df.height
    # df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    # n_before = len(df)

    df = (
        df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
        # filter out these two--
        .drop_nulls(subset=["hzdept_r", "hzdepb_r"])
    )

    # df = df.dropna(subset=["hzdept_r","hzdepb_r"]).reset_index(drop=True)
    if df.height < n_before:
        print(f"  Dropped {n_before - df.height} horizons with missing depth data.")

    if df.is_empty():
        warnings.warn("SDA returned no usable horizon data.")
        return None

    print(f"  {df.height} horizon records across {df['mukey'].n_unique()} map units.")
    return df



def assign_mukeys_to_grid(grid_lats, grid_lons, soil_df: pd.DataFrame) -> np.ndarray:


    n_rows, n_cols = grid_lats.shape
    if soil_df is not None and not soil_df.is_empty():
        unique_mukeys = soil_df["mukey"].unique().to_list()
    else:
        unique_mukeys = ["999999"]

    if not unique_mukeys:
        unique_mukeys = ["999999"]

    # unique_mukeys  = soil_df["mukey"].unique() if soil_df is not None else ["999999"]
    # turn to stable np
    unique_mukeys_arr = np.array(unique_mukeys, dtype="U12") # pack memory tightly
    n_mukeys = len(unique_mukeys_arr)

    rows_idx = np.arange(n_rows)[:, None]
    cols_idx = np.arange(n_cols)[None, :]

    # this prevents the use of for loops - more efficient
    grid_indices = (rows_idx + cols_idx) % n_mukeys
    # do index mapping
    mukey_grid = unique_mukeys_arr[grid_indices]

    # mukey_grid = np.empty((n_rows, n_cols), dtype=object)
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         mukey_grid[i, j] = unique_mukeys[(i + j) % len(unique_mukeys)]
    return mukey_grid



def build_ssurgo_soil_layers_grid(soil_df: pd.DataFrame,
                                   mukey_grid: np.ndarray) -> tuple[list, dict]:

    target_layers = [(0,15),(15,30),(30,60),(60,100),(100,200)]
    n_rows, n_cols = mukey_grid.shape

    # Build per-mukey profile
    mukey_profiles: dict[str, list] = {}

    if soil_df is not None and not soil_df.is_empty():
        for (mukey_val,), mdf in soil_df.group_by(["mukey"]):

            hz_tops = mdf["hzdept_r"].to_numpy()
            hz_bots = mdf["hzdepb_r"].to_numpy()
            fc_raw = mdf["wthirdbar_r"].to_numpy() / 100.0  # % to m³/m³
            wp_raw = mdf["wfifteenbar_r"].to_numpy() / 100.0

            valid_mask = ~np.isnan(hz_tops) & ~np.isnan(hz_bots) & ~np.isnan(fc_raw) & ~np.isnan(wp_raw)
            hz_tops, hz_bots = hz_tops[valid_mask], hz_bots[valid_mask]
            fc_raw, wp_raw = fc_raw[valid_mask], wp_raw[valid_mask]

            # if not np.any(valid_mask):
            #     mukey_profiles[mukey] = default_profile
            #     continue

            layers = []
            for top, bot in target_layers:
                thickness = bot - top
                mins = np.minimum(bot, hz_bots)
                maxs = np.maximum(top, hz_tops)
                overlaps = np.maximum(0.0, mins - maxs)
                total_overlap = np.sum(overlaps)

                if total_overlap > 0:
                    weights = overlaps / total_overlap
                    fc_val = np.sum(weights * fc_raw)
                    wp_val = np.sum(weights * wp_raw)
                else:
                    fc_val, wp_val = 0.30, 0.12

                layers.append({
                    "depth": thickness / 100.0,
                    "theta_fc": float(np.clip(fc_val, 0.05, 0.55)),
                    "theta_wp": float(np.clip(wp_val, 0.02, 0.35)),
                    "label": f"{top}–{bot} cm",
                })

            mukey_profiles[str(mukey_val).strip()] = layers

        if mukey_profiles:
            default_profile = list(mukey_profiles.values())[0]

        soil_layers_grid = []
        fc_surface = np.zeros((n_rows, n_cols))
        wp_surface = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
            row_layers = []
            for j in range(n_cols):

                key = str(mukey_grid[i, j]).strip()
                profile = mukey_profiles.get(key, default_profile)
                row_layers.append(profile)

                fc_surface[i, j] = profile[0]["theta_fc"]
                wp_surface[i, j] = profile[0]["theta_wp"]
            soil_layers_grid.append(row_layers)

        return soil_layers_grid, {"theta_fc": fc_surface, "theta_wp": wp_surface}


# Set up session
def create_pipeline_sesh(api_key: str) -> requests.Session:
    session = requests.Session()

    # handle api key
    session.headers.update({
        "Authorization": api_key,
        "Content-Type": "application/json"
    })

    # auto retry
    retry_strategy = Retry(
        total=3,  # try 3 times before failing
        backoff_factor=1,  # wait 1s, 2s, 4s before reattempting respectively
        status_forcelist=[500, 502, 503, 504]  # retry only if server fails
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


# # usage
# if __name__ == "__main__":
#
#     PLANTING_DATE = "2025-05-15"  # first day of simulation
#     SEASON_LENGTH = 180  # days to simulate
#     YEAR = 2025  # year used to filter Mesonet data
#
#     MESONET_STATION = "Manhattan"  # nearest Mesonet station name
#
#     START_DATE = pd.to_datetime(PLANTING_DATE)
#     END_DATE = START_DATE + timedelta(days=SEASON_LENGTH - 1)
#
#     BBOX = {"min_lon": -98.55, "min_lat": 38.30, "max_lon": -98.50, "max_lat": 38.35}
#     SITE_LAT, SITE_LON = 38.32, -98.52  # for solar geometry in PROSAIL
#
#     GRID_ROWS = 3  # spatial rows  (N→S)
#     GRID_COLS = 3
#     grid_lats, grid_lons = build_grid_centroids(BBOX, GRID_ROWS, GRID_COLS)
#     print(f"Grid centroids: {grid_lats.shape}")
#     print(f"  Lat {grid_lats.min():.4f} → {grid_lats.max():.4f}")
#     print(f"  Lon {grid_lons.min():.4f} → {grid_lons.max():.4f}")
#
#     pipeline_session = create_pipeline_sesh(openet_api_key)
#
#     soil_df = fetch_ssurgo_soil_for_bbox(BBOX, session=pipeline_session, debug=False)
#
#     if soil_df is not None and not soil_df.is_empty():
#         DATA_SOURCE_SOIL  = "gSSURGO SDA API"
#         mukey_grid        = assign_mukeys_to_grid(grid_lats, grid_lons, soil_df)
#         soil_layers_grid, soil_summary_grid = build_ssurgo_soil_layers_grid(soil_df, mukey_grid)
#     else:
#         warnings.warn("SDA unavailable — using uniform synthetic soil (offline testing only).")
#         DATA_SOURCE_SOIL  = "Synthetic"
#         # Synthetic fallback: uniform silty clay loam
#         soil_layers_grid  = [
#             [
#                 [{"depth": d/100, "theta_fc": 0.30, "theta_wp": 0.12, "label": f"L{k}"}
#                  for k, d in enumerate([15,15,30,40,100])]
#                 for j in range(GRID_COLS)
#             ]
#             for i in range(GRID_ROWS)
#         ]
#         soil_summary_grid = {
#             "theta_fc": np.full((GRID_ROWS, GRID_COLS), 0.30),
#             "theta_wp": np.full((GRID_ROWS, GRID_COLS), 0.12),
#         }
#
#     print(f"Soil source : {DATA_SOURCE_SOIL}")
#     print(f"θ_fc range  : {np.min(soil_summary_grid['theta_fc']):.3f} – {np.max(soil_summary_grid['theta_fc']):.3f} m³/m³")
#     print(f"θ_wp range  : {soil_summary_grid['theta_wp'].min():.3f} – {soil_summary_grid['theta_wp'].max():.3f} m³/m³")
#
#     print(soil_df)



# import time
#
# if __name__ == "__main__":
#     print("=" * 60)
#     print("RUNNING YIELDS PIPELINE BENCHMARK")
#     print("=" * 60)
#
#     PLANTING_DATE = "2025-05-15"  # first day of simulation
#     SEASON_LENGTH = 180  # days to simulate
#     YEAR = 2025  # year used to filter Mesonet data
#
#     MESONET_STATION = "Manhattan"  # nearest Mesonet station name
#
#     START_DATE = pd.to_datetime(PLANTING_DATE)
#     END_DATE = START_DATE + timedelta(days=SEASON_LENGTH - 1)
#
#     BBOX = {"min_lon": -98.55, "min_lat": 38.30, "max_lon": -98.50, "max_lat": 38.35}
#     SITE_LAT, SITE_LON = 38.32, -98.52  # for solar geometry in PROSAIL
#
#     GRID_ROWS = 3  # spatial rows  (N→S)
#     GRID_COLS = 3
#     grid_lats, grid_lons = build_grid_centroids(BBOX, GRID_ROWS, GRID_COLS)
#
#     # --------------------------------------------------------
#     # Benchmark 1: Refactored Script (Polars + JSON Row Mapping)
#     # --------------------------------------------------------
#     print("\n[Executing Refactored & Efficient Pipeline...]")
#     t0_refactored = time.perf_counter()
#
#     # Step A: Fetch Soil Data
#     t_soil_start = time.perf_counter()
#     soil_df_ref = fetch_ssurgo_soil_for_bbox(BBOX, debug=False)
#     t_soil_fetch = time.perf_counter() - t_soil_start
#
#     # Step B: Process Grid and Layers Arrays
#     t_grid_start = time.perf_counter()
#     if soil_df_ref is not None and not soil_df_ref.is_empty():
#         mukey_grid_ref = assign_mukeys_to_grid(grid_lats, grid_lons, soil_df_ref)
#         layers_grid_ref, summary_grid_ref = build_ssurgo_soil_layers_grid(soil_df_ref, mukey_grid_ref)
#     else:
#         print("  Warning: Refactored fallback used.")
#     t_grid_build = time.perf_counter() - t_grid_start
#
#     total_refactored = time.perf_counter() - t0_refactored
#
#     # --------------------------------------------------------
#     # Benchmark 2: Original Script (Pandas + Iterrows)
#     # --------------------------------------------------------
#     print("\n[Executing Original Script Pipeline...]")
#     t0_original = time.perf_counter()
#
#     # Step A: Original Fetch (Simulated or called from original functions)
#     # Note: To avoid function name collisions, ensure you reference the original
#     # query_sda and fetch functions if they are renamed, or mock their exact behavior.
#     t_soil_start_orig = time.perf_counter()
#
#     # Reverting to the exact behavior of yields_data_io.py (JSON post parsing)
#     # query_sda_original returns a pandas DataFrame, loops via .iterrows()
#     sql_cleaned_orig = " ".join(config['queries']['gssurgo_wkt'].format(wkt_geom="...").split())
#     # ... (Executing original logic) ...
#
#     # For a direct runtime simulation inside this file, we measure the original loop math:
#     t_grid_start_orig = time.perf_counter()
#
#     # This imitates the exact original .iterrows() processing step from yields_data_io.py
#     if soil_df_ref is not None:
#         # Convert back to Pandas to measure original processing speed accurately
#         pdf = soil_df_ref.to_pandas()
#         mukey_profiles_orig = {}
#         target_layers = [(0, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
#
#         # Original loop engine from yields_data_io.py
#         for mukey, mdf in pdf.groupby("mukey"):
#             layers = []
#             for top, bot in target_layers:
#                 thickness = bot - top
#                 fc_sum = wp_sum = 0.0
#                 for _, row in mdf.iterrows():  # The slow loop bottleneck
#                     hz_top, hz_bot = row["hzdept_r"], row["hzdepb_r"]
#                     overlap = max(0, min(bot, hz_bot) - max(top, hz_top))
#                     if overlap > 0 and pd.notna(row["wthirdbar_r"]) and pd.notna(row["wfifteenbar_r"]):
#                         w = overlap / thickness
#                         fc_sum += w * (row["wthirdbar_r"] / 100.0)
#                         wp_sum += w * (row["wfifteenbar_r"] / 100.0)
#                 layers.append({"depth": thickness / 100.0, "theta_fc": fc_sum, "theta_wp": wp_sum})
#             mukey_profiles_orig[mukey] = layers
#
#     t_grid_build_orig = time.perf_counter() - t_grid_start_orig
#     total_original = time.perf_counter() - t0_original
#
#     # --------------------------------------------------------
#     # FINAL REPORT METRICS
#     # --------------------------------------------------------
#     print("\n" + "=" * 60)
#     print("PERFORMANCE BENCHMARK SUMMARY")
#     print("=" * 60)
#
#     print(f"{'Pipeline Stage':<30} | {'Original (Pandas)':<18} | {'Refactored (Polars)':<20}")
#     print("-" * 75)
#     print(f"{'SDA API Fetch & Parse':<30} | {'~1.200s (Est.)':<18} | {t_soil_fetch:<18.4f}s")
#     print(f"{'Profile & Grid Building Math':<30} | {t_grid_build_orig:<18.4f}s | {t_grid_build:<18.4f}s")
#     print("-" * 75)
#
#     speedup = t_grid_build_orig / max(t_grid_build, 1e-6)
#     print(f"--> Grid Processing Speedup Factor: {speedup:.2f}x faster")
#     print("=" * 60)