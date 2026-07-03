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
from setuptools.config.pyprojecttoml import load_file

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
config_path = ROOT / 'config.yaml'

openet_api_key = os.getenv('OPENET_API_KEY')


# ============================================================
# 1. WEATHER — Kansas Mesonet API
# ============================================================


def fetch_weather_from_mesonet(station: str, start: str, end: str,
                                year_filter: int = None,
                                planting_date: str = None,
                                config_path: str = config_path) -> tuple[pd.DataFrame, list[dict]]:

    """
    Pull daily weather from the Kansas Mesonet REST API and return a
    clean DataFrame ready for the crop model.

    Parameters
    ----------
    station       : str  — Mesonet station code, e.g. "Manhattan"
    start         : str  — query start date 'YYYY-MM-DD'
    end           : str  — query end date   'YYYY-MM-DD'
    year_filter   : int  — if given, keep only rows from this year
    planting_date : str  — if given ('YYYY-MM-DD'), trim rows before this date
    config        : str  - path to config file

    Returns
    -------
    weather_df   : pd.DataFrame  — cleaned daily weather with columns:
                   DATE, T_mean, TMIN, TMAX, solar_rad, rain, year
    weather_data : list[dict]    — same rows as list-of-dicts for the
                   daily crop model step() function

    Notes
    -----
    Weather is treated as spatially uniform at the 30 m neighbourhood
    scale (one pull per site/county centroid), matching the YIELDS
    proposal design intent.  The Mesonet API returns 'M' for missing
    values; these are converted to NaN automatically.
    """
    with open(config_path) as f:
        config_file = yaml.safe_load(f)

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

    # ── optional filters ─────────────────────────────────────────────────────
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



# usage
PLANTING_DATE = "2025-05-15"                 # first day of simulation
SEASON_LENGTH = 180                          # days to simulate
YEAR = 2025                         # year used to filter Mesonet data

MESONET_STATION = "Manhattan"               # nearest Mesonet station name


# START_DATE = pd.to_datetime(PLANTING_DATE)
# END_DATE   = START_DATE + timedelta(days=SEASON_LENGTH - 1)

weather_df, weather_data = fetch_weather_from_mesonet(
    station      = MESONET_STATION,
    start        = "2025-01-01",        # wide pull; filtered to YEAR below
    end          = "2025-12-31",
    year_filter  = YEAR,
    planting_date= PLANTING_DATE,
)
print(weather_df)

#
# # ============================================================
# # 2. ET — OpenET Ensemble API  (POST /raster/timeseries/point)
# # ============================================================
# # Fixed vs. original notebook:
# #   • OPENET_BASE_URL is base-only; path is appended in the function
# #     (original had the full path baked in AND appended it again → 404)
# #   • Uses POST + JSON body, not GET + query params
# #   • Model name is "Ensemble", not "ensemble_mean" (invalid)
# #   • Response handles both bare list and {"data": [...]} dict
#
# OPENET_BASE_URL = "https://openet-api.org"
#
#
# def build_grid_centroids(bbox: dict, n_rows: int, n_cols: int):
#     """
#     Generate (lat, lon) centroid arrays for a regular grid over a bounding box.
#
#     Parameters
#     ----------
#     bbox   : dict with keys min_lon, max_lon, min_lat, max_lat  (WGS84)
#     n_rows : int — number of grid rows  (latitude direction, N→S)
#     n_cols : int — number of grid cols  (longitude direction, W→E)
#
#     Returns
#     -------
#     lats : np.ndarray, shape (n_rows, n_cols)
#     lons : np.ndarray, shape (n_rows, n_cols)
#     """
#     lat_centers = np.linspace(bbox["max_lat"], bbox["min_lat"], n_rows)
#     lon_centers = np.linspace(bbox["min_lon"], bbox["max_lon"], n_cols)
#     lons, lats  = np.meshgrid(lon_centers, lat_centers)
#     return lats, lons
#
#
# def query_openet_timeseries(lat: float, lon: float,
#                              start_date: str, end_date: str,
#                              api_key: str,
#                              model: str = "Ensemble",
#                              interval: str = "daily",
#                              reference_et: str = "gridMET",
#                              units: str = "mm") -> pd.Series | None:
#     """
#     Pull a daily ET time series for a single point from the OpenET API.
#
#     Parameters
#     ----------
#     lat, lon      : float — WGS84 coordinates (lat first for readability;
#                             lon is sent first in the POST body per GeoJSON)
#     start_date    : str   — 'YYYY-MM-DD'
#     end_date      : str   — 'YYYY-MM-DD'
#     api_key       : str   — OpenET key (Authorization header, no "Bearer")
#     model         : str   — "Ensemble" | "SSEBop" | "SIMS" | "DisALEXI" |
#                             "PTJPL" | "eeMETRIC" | "geeSEBAL"
#     interval      : str   — "daily" or "monthly"
#     reference_et  : str   — "gridMET" (CONUS) or "CIMIS" (California)
#     units         : str   — "mm" or "in"
#
#     Returns
#     -------
#     pd.Series  — date-indexed, values in mm/day; or None on failure.
#
#     Quota note
#     ----------
#     Free tier: 100 queries/month, 50 000 acres/query.  For a full 30 m
#     field raster use fetch_et_raster_for_polygon() (one query returns
#     the whole field) instead of looping over every pixel here.
#     """
#     endpoint = f"{OPENET_BASE_URL}/raster/timeseries/point"
#     headers  = {"Authorization": api_key, "Content-Type": "application/json"}
#     body     = {
#         "date_range":   [start_date, end_date],
#         "interval":     interval,
#         "geometry":     [lon, lat],     # GeoJSON: lon first
#         "model":        model,
#         "variable":     "ET",
#         "reference_et": reference_et,
#         "units":        units,
#         "file_format":  "JSON",
#     }
#
#     try:
#         resp = requests.post(endpoint, headers=headers, json=body, timeout=45)
#         resp.raise_for_status()
#         data = resp.json()
#
#         # Response is either a bare list or {"data": [...]}
#         if isinstance(data, list):
#             records = data
#         elif isinstance(data, dict):
#             records = data.get("data", [])
#         else:
#             records = []
#
#         if not records:
#             warnings.warn(f"OpenET: empty response for ({lat:.4f}, {lon:.4f})")
#             return None
#
#         dates  = pd.to_datetime([r["time"] for r in records])
#         values = np.array(
#             [r["et"] if r["et"] is not None else np.nan for r in records],
#             dtype=float,
#         )
#         return pd.Series(values, index=dates, name="ET_openet")
#
#     except requests.exceptions.Timeout:
#         warnings.warn(f"OpenET: request timed out for ({lat:.4f}, {lon:.4f})")
#     except requests.exceptions.HTTPError as exc:
#         code = exc.response.status_code
#         body_txt = getattr(exc.response, "text", "")[:200]
#         warnings.warn(f"OpenET: HTTP {code} for ({lat:.4f}, {lon:.4f}) — {body_txt}")
#     except Exception as exc:
#         warnings.warn(f"OpenET: query failed for ({lat:.4f}, {lon:.4f}) — {exc}")
#     return None
#
#
# def fetch_et_stack_from_openet(grid_lats, grid_lons,
#                                 start_date: str, end_date: str,
#                                 api_key: str,
#                                 season_length: int) -> tuple[np.ndarray, bool]:
#     """
#     Build a 3-D ET array (season_length × n_rows × n_cols) by querying
#     the OpenET point endpoint once per grid cell.
#
#     For production-scale use (full field at 30 m) switch to the polygon
#     raster-export endpoint to stay within the free-tier query quota.
#
#     Parameters
#     ----------
#     grid_lats, grid_lons : 2D np.ndarray from build_grid_centroids()
#     start_date, end_date : str, 'YYYY-MM-DD'
#     api_key              : str — OpenET API key
#     season_length        : int — number of days to fill
#
#     Returns
#     -------
#     ET_stack    : np.ndarray, shape (season_length, n_rows, n_cols), mm/day
#                   NaN where the API returned no data.
#     any_success : bool — True if at least one cell returned valid data.
#     """
#     n_rows, n_cols = grid_lats.shape
#     ET_stack  = np.full((season_length, n_rows, n_cols), np.nan)
#     date_idx  = pd.date_range(start_date, periods=season_length, freq="D")
#     any_success = False
#
#     for i in range(n_rows):
#         for j in range(n_cols):
#             lat = float(grid_lats[i, j])
#             lon = float(grid_lons[i, j])
#             print(f"  [OpenET] cell ({i},{j})  lat={lat:.4f}  lon={lon:.4f}")
#
#             s = query_openet_timeseries(
#                 lat=lat, lon=lon,
#                 start_date=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
#                 end_date=pd.to_datetime(end_date).strftime("%Y-%m-%d"),
#                 api_key=api_key,
#             )
#             if s is not None and len(s) > 0:
#                 s = s.reindex(date_idx).interpolate(method="linear", limit=5)
#                 ET_stack[:len(s), i, j] = s.values[:season_length]
#                 any_success = True
#
#     return ET_stack, any_success
#
#
# def generate_et_stack_synthetic(n_rows: int, n_cols: int,
#                                  season_length: int, seed: int = 42) -> np.ndarray:
#     """
#     Synthetic ET fallback when the API is unavailable.
#
#     Models a Gaussian seasonal bell (peak ~day 90) with per-cell
#     lognormal spatial multipliers (CV ≈ 15%) and small daily noise.
#     Use only for offline testing — not for real analysis.
#
#     Returns
#     -------
#     np.ndarray, shape (season_length, n_rows, n_cols), mm/day
#     """
#     rng   = np.random.default_rng(seed)
#     days  = np.arange(season_length)
#     curve = 4.0 * np.exp(-0.5 * ((days - 90) / 40) ** 2) + 0.5
#     scale = rng.lognormal(0.0, 0.15, (n_rows, n_cols))
#     noise = rng.normal(1.0, 0.05, (season_length, n_rows, n_cols))
#     return np.clip(curve[:, None, None] * scale[None, :, :] * noise, 0.1, None)
#
#
# # ============================================================
# # 3. SOIL — USDA gSSURGO via Soil Data Access (SDA) API
# # ============================================================
#
# SDA_URL = "https://SDMDataAccess.nrcs.usda.gov/Tabular/post.rest"
#
#
# def query_sda(sql_query: str, expected_columns: list[str],
#               debug: bool = False) -> pd.DataFrame | None:
#     """
#     POST a SQL query to the USDA Soil Data Access REST endpoint.
#
#     SDA's JSON response does not reliably include a header row, so the
#     caller must supply the column names in the same order as the SELECT.
#
#     Parameters
#     ----------
#     sql_query        : str       — SQL string
#     expected_columns : list[str] — column names matching the SELECT order
#     debug            : bool      — if True, print raw response for QA
#
#     Returns
#     -------
#     pd.DataFrame or None
#     """
#     payload = {"query": sql_query, "format": "JSON"}
#     try:
#         resp = requests.post(SDA_URL, data=payload, timeout=30)
#         resp.raise_for_status()
#         result = resp.json()
#
#         if debug:
#             print("SDA response keys :", list(result.keys()))
#             print("SDA response (head):", str(result)[:1500])
#
#         table = result.get("Table")
#         if not table:
#             warnings.warn(
#                 "SDA returned no 'Table'. The bounding box may not overlap "
#                 "any soil map units, or the SQL may have an error."
#             )
#             return None
#
#         # Detect optional header row
#         first = [str(x).strip().lower() for x in table[0]]
#         rows  = table[1:] if first == [c.lower() for c in expected_columns] else table
#
#         if not rows:
#             warnings.warn("SDA query returned schema but zero data rows.")
#             return None
#
#         return pd.DataFrame(rows, columns=expected_columns)
#
#     except requests.exceptions.HTTPError as exc:
#         warnings.warn(f"SDA HTTP error: {exc}")
#     except ValueError as exc:
#         warnings.warn(f"SDA response was not valid JSON: {exc}")
#     except Exception as exc:
#         warnings.warn(f"SDA query failed: {exc}")
#     return None
#
#
# def fetch_ssurgo_soil_for_bbox(bbox: dict, debug: bool = False) -> pd.DataFrame | None:
#     """
#     Query gSSURGO via SDA for soil horizon properties within a bounding box.
#
#     Columns returned
#     ----------------
#     mukey, muname, cokey, hzdept_r, hzdepb_r,
#     wthirdbar_r (field capacity %), wfifteenbar_r (wilting point %),
#     sandtotal_r, claytotal_r, om_r
#
#     Parameters
#     ----------
#     bbox  : dict — {min_lon, min_lat, max_lon, max_lat} in WGS84
#     debug : bool — pass to query_sda for raw-response inspection
#
#     Returns
#     -------
#     pd.DataFrame or None
#     """
#     wkt = (
#         f"POLYGON(({bbox['min_lon']} {bbox['min_lat']}, "
#         f"{bbox['max_lon']} {bbox['min_lat']}, "
#         f"{bbox['max_lon']} {bbox['max_lat']}, "
#         f"{bbox['min_lon']} {bbox['max_lat']}, "
#         f"{bbox['min_lon']} {bbox['min_lat']}))"
#     )
#
#     cols = [
#         "mukey", "muname", "cokey", "hzdept_r", "hzdepb_r",
#         "wthirdbar_r", "wfifteenbar_r", "sandtotal_r", "claytotal_r", "om_r",
#     ]
#
#     sql = f"""
#         SELECT
#             mu.mukey, mu.muname, co.cokey,
#             ch.hzdept_r, ch.hzdepb_r,
#             ch.wthirdbar_r, ch.wfifteenbar_r,
#             ch.sandtotal_r, ch.claytotal_r, ch.om_r
#         FROM
#             mapunit AS mu
#             INNER JOIN SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}') AS s
#                 ON mu.mukey = s.mukey
#             INNER JOIN component AS co ON mu.mukey = co.mukey
#             INNER JOIN chorizon  AS ch ON co.cokey = ch.cokey
#         WHERE co.majcompflag = 'Yes'
#         ORDER BY mu.mukey, ch.hzdept_r
#     """
#
#     print("[SDA] Querying gSSURGO for soil horizons...")
#     df = query_sda(sql, cols, debug=debug)
#     if df is None:
#         return None
#
#     numeric = ["hzdept_r","hzdepb_r","wthirdbar_r","wfifteenbar_r",
#                "sandtotal_r","claytotal_r","om_r"]
#     df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
#
#     n_before = len(df)
#     df = df.dropna(subset=["hzdept_r","hzdepb_r"]).reset_index(drop=True)
#     if len(df) < n_before:
#         print(f"  Dropped {n_before - len(df)} horizons with missing depth data.")
#
#     if df.empty:
#         warnings.warn("SDA returned no usable horizon data.")
#         return None
#
#     print(f"  {len(df)} horizon records across {df['mukey'].nunique()} map units.")
#     return df
#
#
# def assign_mukeys_to_grid(grid_lats, grid_lons, soil_df: pd.DataFrame) -> np.ndarray:
#     """
#     Assign a gSSURGO mukey to each grid cell.
#
#     Currently distributes available mukeys deterministically across the grid
#     (round-robin by (row+col) index).  Replace with a spatial intersection
#     (e.g. geopandas sjoin against the mapunit polygon shapefile) for a
#     production implementation.
#
#     Parameters
#     ----------
#     grid_lats, grid_lons : 2D np.ndarray from build_grid_centroids()
#     soil_df              : pd.DataFrame from fetch_ssurgo_soil_for_bbox()
#
#     Returns
#     -------
#     mukey_grid : 2D np.ndarray (dtype object), shape (n_rows, n_cols)
#     """
#     n_rows, n_cols = grid_lats.shape
#     unique_mukeys  = soil_df["mukey"].unique() if soil_df is not None else ["999999"]
#     if len(unique_mukeys) == 0:
#         unique_mukeys = ["999999"]
#
#     mukey_grid = np.empty((n_rows, n_cols), dtype=object)
#     for i in range(n_rows):
#         for j in range(n_cols):
#             mukey_grid[i, j] = unique_mukeys[(i + j) % len(unique_mukeys)]
#     return mukey_grid
#
#
# def build_ssurgo_soil_layers_grid(soil_df: pd.DataFrame,
#                                    mukey_grid: np.ndarray) -> tuple[list, dict]:
#     """
#     Convert SSURGO horizon data into per-cell layered soil profiles.
#
#     Horizons are aggregated into 5 standard depth intervals, weighted by
#     overlap.  wthirdbar_r and wfifteenbar_r are in % → converted to
#     volumetric fraction (÷ 100) so they are consistent with the model's
#     θ_fc / θ_wp convention (cm³ cm⁻³).
#
#     Parameters
#     ----------
#     soil_df    : pd.DataFrame — from fetch_ssurgo_soil_for_bbox()
#     mukey_grid : 2D np.ndarray — from assign_mukeys_to_grid()
#
#     Returns
#     -------
#     soil_layers_grid : list[list[list[dict]]]
#                        [row][col] → list of 5 layer dicts, each with:
#                        depth (m), theta_fc, theta_wp, label (str)
#     soil_summary_grid : dict of 2D np.ndarray
#                         keys: "theta_fc", "theta_wp"  (surface layer)
#     """
#     target_layers = [(0,15),(15,30),(30,60),(60,100),(100,200)]
#     n_rows, n_cols = mukey_grid.shape
#
#     # Build per-mukey profile
#     mukey_profiles: dict[str, list] = {}
#     for mukey, mdf in soil_df.groupby("mukey"):
#         layers = []
#         for top, bot in target_layers:
#             thickness = bot - top
#             fc_sum = wp_sum = 0.0
#             for _, row in mdf.iterrows():
#                 hz_top   = row["hzdept_r"]
#                 hz_bot   = row["hzdepb_r"]
#                 overlap  = max(0, min(bot, hz_bot) - max(top, hz_top))
#                 if overlap > 0 and pd.notna(row["wthirdbar_r"]) and pd.notna(row["wfifteenbar_r"]):
#                     w       = overlap / thickness
#                     fc_sum += w * (row["wthirdbar_r"]   / 100.0)  # % → fraction
#                     wp_sum += w * (row["wfifteenbar_r"] / 100.0)
#             layers.append({
#                 "depth":    (bot - top) / 100.0,                  # cm → m
#                 "theta_fc": float(np.clip(fc_sum if fc_sum > 0 else 0.30, 0.05, 0.55)),
#                 "theta_wp": float(np.clip(wp_sum if wp_sum > 0 else 0.12, 0.02, 0.35)),
#                 "label":    f"{top}–{bot} cm",
#             })
#         mukey_profiles[mukey] = layers
#
#     # Assign profiles to grid cells
#     default_profile = list(mukey_profiles.values())[0]
#     soil_layers_grid = []
#     fc_surface = np.zeros((n_rows, n_cols))
#     wp_surface = np.zeros((n_rows, n_cols))
#
#     for i in range(n_rows):
#         row_layers = []
#         for j in range(n_cols):
#             profile = mukey_profiles.get(mukey_grid[i, j], default_profile)
#             row_layers.append(profile)
#             fc_surface[i, j] = profile[0]["theta_fc"]
#             wp_surface[i, j] = profile[0]["theta_wp"]
#         soil_layers_grid.append(row_layers)
#
#     return soil_layers_grid, {"theta_fc": fc_surface, "theta_wp": wp_surface}
