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
4. Planetary Computer STAC — real Landsat Collection 2 surface reflectance
5. USDA NASS CDL   — Cropland Data Layer crop-type classification

"""


import os
import io
import re
import warnings
import xml.etree.ElementTree as ET
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
# 1. Weather data: Kansas Mesonet API
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
# 2. Evapotranspiration data: OpenET Ensemble API
# ============================================================
# The endpoint expects a POST request with a JSON body. Responses may be
# returned either as a list or inside a top-level "data" field.


def build_grid_centroids(bbox: dict, n_rows: int, n_cols: int):

    """
    :param bbox: with keys min_lon, max_lon, min_lat, max_lat  (WGS84)
    :param n_rows: number of grid rows  (latitude direction, N→S)
    :param n_cols: number of grid cols  (longitude direction, W→E)

    :return: lats (n_rows, n_cols), lons (n_rows, n_cols)

    NOTE: this spaces n_rows x n_cols centroids evenly in degrees across
    the bbox -- it does NOT align to real Landsat 30 m pixel boundaries.
    For output that corresponds to actual USGS Landsat pixels, use
    build_landsat_aligned_grid() below instead. This function is kept for
    quick tests/arbitrary grids where exact pixel alignment doesn't matter.
    """

    lat_centers = np.linspace(bbox["max_lat"], bbox["min_lat"], n_rows)
    lon_centers = np.linspace(bbox["min_lon"], bbox["max_lon"], n_cols, dtype=np.float64)
    lats, lons = np.meshgrid(lat_centers, lon_centers, indexing='ij', copy=False)
    return lats, lons


# Above this many cells, the per-pixel loops elsewhere in this pipeline
# (fetch_et_stack_from_openet's one-sequential-HTTP-request-per-pixel loop,
# and the crop model's pure-Python per-cell-per-day loop in
# cm_sim_engine.run_grid_simulation) get slow enough to be worth a warning.
# This is a heads-up, not a hard limit -- raise it yourself if you're
# prepared for the runtime.
_LARGE_GRID_WARNING_THRESHOLD = 2000


def build_landsat_aligned_grid(bbox: dict,
                                platforms: tuple = ("landsat-8", "landsat-9"),
                                lookback_days: int = 365,
                                subscription_key: str | None = None,
                                return_metadata: bool = False):
    """
    Build the operational pixel grid directly from a REAL Landsat scene's
    own 30 m pixel grid over the bbox, instead of inventing an independent
    lat/lon grid and hoping it lines up (see build_grid_centroids's
    caveat). Every (i, j) cell this returns is the centroid of an actual
    USGS Landsat Collection 2 pixel, not an approximation of one.

    HOW: finds any recent Landsat scene (Collection 2 Level-2) whose
    footprint covers the bbox via Planetary Computer's STAC catalog. The
    scene's specific date/cloud-cover don't matter here -- every
    Collection 2 Level-2 product for a given WRS-2 path/row shares the
    exact same pixel grid (same CRS, same affine transform), so this
    scene is used purely as a reference for that grid's geometry, not for
    its imagery. Reads that grid's transform, computes the integer pixel
    window covering the bbox, and returns the WGS84 lat/lon of every
    pixel centroid in that window -- these coordinates are what get
    passed to fetch_et_stack_from_openet, fetch_ssurgo_soil_for_bbox's
    mukey assignment, fetch_cdl_crop_mask_for_grid, and (later)
    fetch_landsat_bands_for_grid, so every step of the pipeline operates
    on the same real pixel grid.

    SCOPE: this assumes the whole bbox is covered by ONE Landsat scene
    (i.e. fits within a single WRS-2 tile / UTM zone, true for anything
    up to roughly field/farm/county scale). It is NOT the mechanism for
    tiling a multi-scene, multi-UTM-zone area like an entire state --
    that needs per-scene tiling and a mosaicking step. If the computed grid
    comes out larger than
    _LARGE_GRID_WARNING_THRESHOLD cells, a warning is printed pointing at
    the specific downstream loops that will be slow at that size.

    :param bbox: dict with min_lon, max_lon, min_lat, max_lat (WGS84)
    :param platforms: which Landsat platforms to search for a reference scene
    :param lookback_days: how far back from today to search (any cloud
        cover is fine -- only the grid geometry is used, not the imagery)
    :param subscription_key: optional Planetary Computer key (see
        fetch_landsat_bands_for_grid's docstring for where this goes)
    :return: (grid_lats, grid_lons), each shape (n_rows, n_cols) -- a
        drop-in replacement for build_grid_centroids()'s return value. If
        return_metadata is true, a third dict contains the reference scene
        CRS and affine transform for georeferenced raster outputs.
    """
    try:
        import pystac_client
        import planetary_computer as pc
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "build_landsat_aligned_grid needs pystac-client, planetary-computer, "
            "rasterio, and pyproj. Install with:\n"
            "    pip install pystac-client planetary-computer rasterio pyproj"
        ) from exc

    key = subscription_key or os.getenv("PC_SDK_SUBSCRIPTION_KEY")
    if key:
        pc.set_subscription_key(key)

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

    pad = 0.001  # degrees, tiny buffer so the search bbox isn't degenerate
    search_bbox = [bbox["min_lon"] - pad, bbox["min_lat"] - pad,
                    bbox["max_lon"] + pad, bbox["max_lat"] + pad]

    today = pd.Timestamp.utcnow().normalize()
    start = (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=search_bbox,
        datetime=f"{start}/{end}",
        query={"platform": {"in": list(platforms)}},
        limit=1,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Landsat scene found covering this bbox in the last "
            f"{lookback_days} days -- cannot determine the real pixel grid "
            f"without a reference scene. Widen lookback_days or double-check "
            f"the bbox is over land Landsat actually images."
        )
    item = items[0]

    with rasterio.open(item.assets["qa_pixel"].href) as ds:
        crs = ds.crs
        to_native = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x0, y0 = to_native.transform(bbox["min_lon"], bbox["min_lat"])
        x1, y1 = to_native.transform(bbox["max_lon"], bbox["max_lat"])
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        window = from_bounds(xmin, ymin, xmax, ymax, transform=ds.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise RuntimeError(
                "Computed a zero-size pixel window for this bbox -- it may "
                "be smaller than one 30 m Landsat pixel. Widen the bbox."
            )
        win_transform = ds.window_transform(window)
        n_rows_grid, n_cols_grid = int(round(window.height)), int(round(window.width))

    n_cells = n_rows_grid * n_cols_grid
    if n_cells > _LARGE_GRID_WARNING_THRESHOLD:
        warnings.warn(
            f"build_landsat_aligned_grid: this bbox covers {n_rows_grid} x "
            f"{n_cols_grid} = {n_cells} real Landsat pixels. "
            f"fetch_et_stack_from_openet queries one point at a time "
            f"({n_cells} sequential HTTP requests), and "
            f"cm_sim_engine.run_grid_simulation's crop-model loop is pure "
            f"Python per cell per day -- both will be slow at this size. "
            f"Fine to proceed, just expect a longer run; shrink the bbox "
            f"if you want faster iteration."
        )

    # Pixel-center coordinates in the scene's native CRS, vectorized.
    # Assumes an axis-aligned (non-rotated) raster -- true for all
    # standard Landsat Collection 2 products (delivered north-up in UTM).
    rows_idx, cols_idx = np.meshgrid(np.arange(n_rows_grid), np.arange(n_cols_grid), indexing="ij")
    xs = win_transform.c + (cols_idx + 0.5) * win_transform.a
    ys = win_transform.f + (rows_idx + 0.5) * win_transform.e

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = to_wgs84.transform(xs.ravel(), ys.ravel())
    grid_lons = np.asarray(lons, dtype=np.float64).reshape(n_rows_grid, n_cols_grid)
    grid_lats = np.asarray(lats, dtype=np.float64).reshape(n_rows_grid, n_cols_grid)

    print(f"[Landsat grid] reference scene {item.id} ({item.datetime.strftime('%Y-%m-%d')}), "
          f"CRS {crs}: bbox covers {n_rows_grid} x {n_cols_grid} = {n_cells} real 30 m Landsat pixels.")

    if return_metadata:
        return grid_lats, grid_lons, {
            "crs": crs,
            "transform": win_transform,
            "width": n_cols_grid,
            "height": n_rows_grid,
            "scene_id": item.id,
        }
    return grid_lats, grid_lons



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

    # Reuse an active session when supplied to avoid repeated connections.
    # The requests module remains the fallback for one-off queries.
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
        resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
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


# def generate_et_stack_synthetic(n_rows: int, n_cols: int,
#                                  season_length: int, seed: int = SEED, config_file: dict = config) -> np.ndarray:

#     rng   = np.random.default_rng(seed)
#     days  = np.arange(season_length, dtype=np.float64)
#     curve = 4.0 * np.exp(-0.5 * ((days - 90) / 40) ** 2) + 0.5
#     scale = rng.lognormal(0.0, 0.15, (n_rows, n_cols))
#     noise = rng.normal(1.0, 0.05, (season_length, n_rows, n_cols))
#     return np.clip(curve[:, None, None] * scale[None, :, :] * noise, 0.1, None)



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



def assign_mukeys_to_grid(grid_lats: np.ndarray, grid_lons: np.ndarray,
                           soil_df: pl.DataFrame,
                           session: requests.Session = None,
                           config_file: dict = config,
                           batch_size: int = 250,
                           debug: bool = False) -> np.ndarray:
    """
    Assign each grid pixel its REAL SSURGO map unit (mukey) via a true
    point-in-polygon spatial query against USDA Soil Data Access -- NOT an
    arbitrary index formula. (A previous version of this function assigned
    mukeys via `(i + j) % n_mukeys`, a round-robin pattern with no relation
    to where each pixel actually sits on the ground. That has been removed.)

    Sends every pixel centroid as a WKT point through SQL Server's
    CROSS APPLY against SDA_Get_Mukey_from_intersection_with_WktWgs84 (the
    same function fetch_ssurgo_soil_for_bbox already uses for the bbox
    itself), batched `batch_size` points per HTTP request rather than one
    request per pixel -- e.g. a 544-pixel grid becomes ~3 requests, not 544.

    Pixels whose point fails to intersect any surveyed mapunit (SDA error,
    edge of survey coverage, etc.) fall back to the single most common
    mukey already present in `soil_df` -- itself real SSURGO data pulled
    for this bbox, just not specific to that exact point. This is reported,
    never silently assumed correct.

    :param grid_lats, grid_lons: 2D centroid arrays (as from build_grid_centroids)
    :param soil_df: output of fetch_ssurgo_soil_for_bbox (used for the
        fallback mukey pool)
    :param session: optional requests.Session (SDA works fine with the
        shared pipeline session -- unlike the legacy CDL point service)
    :param batch_size: pixels per SDA request
    :param debug: print the generated SQL and raw SDA response for the
        first batch
    :return: (n_rows, n_cols) array of mukey strings
    """
    n_rows, n_cols = grid_lats.shape
    n_total = n_rows * n_cols

    if soil_df is None or soil_df.is_empty():
        raise ValueError(
            "assign_mukeys_to_grid needs a non-empty soil_df (output of "
            "fetch_ssurgo_soil_for_bbox) to use as the fallback mukey pool."
        )

    fallback_mukey = str(soil_df["mukey"].mode()[0])

    points = [
        (i * n_cols + j, float(grid_lats[i, j]), float(grid_lons[i, j]))
        for i in range(n_rows) for j in range(n_cols)
    ]

    parts = []
    for start in range(0, n_total, batch_size):
        batch = points[start:start + batch_size]
        values_sql = ",".join(f"({idx}, 'POINT({lon} {lat})')" for idx, lat, lon in batch)
        sql = (
            "SELECT p.pixel_idx, m.mukey\n"
            f"FROM (VALUES {values_sql}) AS p(pixel_idx, wkt)\n"
            "CROSS APPLY SDA_Get_Mukey_from_intersection_with_WktWgs84(p.wkt) AS m"
        )
        part = query_sda(sql, ["pixel_idx", "mukey"], debug=(debug and start == 0),
                          session=session, config_file=config_file)
        if part is not None and not part.is_empty():
            parts.append(part)

    mukey_grid = np.full((n_rows, n_cols), fallback_mukey, dtype="U12")
    n_assigned = 0
    if parts:
        result = pl.concat(parts).with_columns(pl.col("pixel_idx").cast(pl.Int64, strict=False))
        result = result.unique(subset=["pixel_idx"], keep="first")  # a point straddling two polygons returns >1 row
        for row in result.iter_rows(named=True):
            idx = row["pixel_idx"]
            if idx is None or row["mukey"] is None:
                continue
            i, j = divmod(int(idx), n_cols)
            if 0 <= i < n_rows and 0 <= j < n_cols:
                mukey_grid[i, j] = str(row["mukey"]).strip()
                n_assigned += 1

    print(f"[SSURGO] {n_assigned}/{n_total} grid pixel(s) matched to a real mukey by point-in-polygon "
          f"intersection; {n_total - n_assigned} fell back to the bbox's most common mukey "
          f"({fallback_mukey}).")
    if n_assigned == 0:
        warnings.warn(
            "No grid pixel's point-in-polygon SDA query succeeded -- every "
            "cell is using the bbox's single most common mukey as a "
            "real-but-not-point-specific fallback. Rerun with debug=True to "
            "inspect the generated SQL and SDA's raw response."
        )
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


# ============================================================
# 4. REAL LANDSAT RETRIEVAL (Microsoft Planetary Computer STAC)
# ============================================================
# Added to support src/scym_model.py, which needs real Landsat surface
# reflectance (not PROSAIL-simulated reflectance) to compute the GCVI that
# Lobell et al. (2015)'s SCYM regression is calibrated on.
#
# Uses Microsoft's Planetary Computer STAC catalog + Landsat Collection 2
# Level-2 (surface reflectance). This is a free, public catalog — no paid
# account needed. An optional free subscription key raises your per-IP rate
# limit; get one at https://planetarycomputer.microsoft.com/account/request
# and put it in your .env file as:
#     PC_SDK_SUBSCRIPTION_KEY=your_key_here
# (same .env file that already holds OPENET_API_KEY). It is picked up
# automatically below via os.getenv — no code changes needed if you add it.
#
# Needs three extra packages not otherwise used in this pipeline:
#     pip install pystac-client planetary-computer rasterio pyproj

def fetch_landsat_bands_for_grid(grid_lats: np.ndarray, grid_lons: np.ndarray,
                                  start_date, end_date,
                                  max_cloud_cover: float = 70.0,
                                  platforms: tuple = ("landsat-8", "landsat-9"),
                                  subscription_key: str | None = None) -> pl.DataFrame:
    """
    Query the Planetary Computer STAC catalog for Landsat Collection 2
    Level-2 (surface reflectance) scenes covering the grid's bounding box
    and date range, then sample the bands needed for NDVI/SAVI/GCVI at
    every grid-cell centroid, masking cloud/cloud-shadow/dilated-cloud
    pixels using the QA_PIXEL band.

    PERFORMANCE NOTE: this reads one small *windowed* array per band per
    scene (clipped tightly to the grid's extent) and then vectorized-
    indexes every grid point into that in-memory array. It does NOT open
    one network connection per pixel. An earlier version of this function
    called `dataset.sample([(x, y)])` once per point per band per scene --
    fine for a 3x3 demo grid, but at a few hundred real Landsat-pixel grid
    cells x tens of scenes x 6 bands, that was tens of thousands of
    individual remote reads and silently dropped most of the grid to
    timeouts/transient failures. This version issues roughly
    (n_scenes x n_bands) windowed reads total, regardless of grid size.

    This deliberately returns nothing (an empty DataFrame, with a warning)
    rather than any fabricated stand-in when no clear observations are
    found — callers (e.g. scym_model.run_scym_pipeline) should treat that
    as a hard failure rather than fall back to synthetic values.

    :param grid_lats: 2D array of grid-cell centroid latitudes (as returned
        by build_grid_centroids)
    :param grid_lons: 2D array of grid-cell centroid longitudes
    :param start_date, end_date: season start/end (anything pd.to_datetime accepts)
    :param max_cloud_cover: maximum *scene-level* cloud cover (%) to consider;
        per-pixel cloud/shadow masking is applied separately via QA_PIXEL
    :param platforms: which Landsat platforms to include
    :param subscription_key: optional Planetary Computer key; if omitted,
        reads the PC_SDK_SUBSCRIPTION_KEY environment variable (see above)
    :return: Polars DataFrame, one row per (scene date, pixel) that survived
        cloud masking, with columns:
        DATE, pixel_id, i, j, blue, green, red, nir, swir16
    """
    try:
        import pystac_client
        import planetary_computer as pc
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.transform import rowcol
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "Real Landsat retrieval needs pystac-client, planetary-computer, "
            "rasterio, and pyproj. Install with:\n"
            "    pip install pystac-client planetary-computer rasterio pyproj"
        ) from exc

    key = subscription_key or os.getenv("PC_SDK_SUBSCRIPTION_KEY")
    if key:
        pc.set_subscription_key(key)

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

    min_lat, max_lat = float(np.min(grid_lats)), float(np.max(grid_lats))
    min_lon, max_lon = float(np.min(grid_lons)), float(np.max(grid_lons))
    pad = 0.01  # degrees, small buffer around the grid extent
    bbox = [min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad]

    t_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    t_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{t_start}/{t_end}",
        query={
            "eo:cloud_cover": {"lt": max_cloud_cover},
            "platform": {"in": list(platforms)},
        },
    )
    items = list(search.items())
    print(f"[Landsat] {len(items)} scene(s) found over the grid "
          f"({t_start} to {t_end}, scene cloud cover < {max_cloud_cover}%)")

    empty_schema = {"DATE": pl.String, "pixel_id": pl.String, "i": pl.Int32, "j": pl.Int32,
                     "blue": pl.Float64, "green": pl.Float64, "red": pl.Float64,
                     "nir": pl.Float64, "swir16": pl.Float64}
    if not items:
        warnings.warn("No Landsat scenes matched the query -- widen the date "
                       "range or raise max_cloud_cover. Returning an empty "
                       "DataFrame (no synthetic fallback is generated).")
        return pl.DataFrame(schema=empty_schema)

    n_rows, n_cols = grid_lats.shape
    flat_lats = grid_lats.ravel()
    flat_lons = grid_lons.ravel()
    ij_pairs = [(i, j) for i in range(n_rows) for j in range(n_cols)]

    band_assets = {"blue": "blue", "green": "green", "red": "red",
                    "nir": "nir08", "swir16": "swir16"}
    SR_SCALE, SR_OFFSET = 0.0000275, -0.2  # USGS Collection 2 Level-2 SR scale factors
    WINDOW_PAD_M = 90.0  # ~3 Landsat pixels of buffer around the grid extent

    rows = []
    n_scenes_used = 0
    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        try:
            with rasterio.open(item.assets["qa_pixel"].href) as qa_ds:
                transformer = Transformer.from_crs("EPSG:4326", qa_ds.crs, always_xy=True)
                xs, ys = transformer.transform(flat_lons, flat_lats)
                xmin, xmax = float(np.min(xs)), float(np.max(xs))
                ymin, ymax = float(np.min(ys)), float(np.max(ys))

                window = from_bounds(xmin - WINDOW_PAD_M, ymin - WINDOW_PAD_M,
                                      xmax + WINDOW_PAD_M, ymax + WINDOW_PAD_M,
                                      transform=qa_ds.transform).round_offsets().round_lengths()
                if window.width <= 0 or window.height <= 0:
                    warnings.warn(f"[Landsat] scene {item.id}: computed window is empty, skipping.")
                    continue

                qa_arr = qa_ds.read(1, window=window)
                win_transform = qa_ds.window_transform(window)

            band_arrays = {}
            for band, asset_key in band_assets.items():
                with rasterio.open(item.assets[asset_key].href) as ds:
                    band_arrays[band] = ds.read(1, window=window)

            rows_arr, cols_arr = rowcol(win_transform, xs, ys)
            rows_arr = np.asarray(rows_arr)
            cols_arr = np.asarray(cols_arr)
            in_bounds = ((rows_arr >= 0) & (rows_arr < qa_arr.shape[0]) &
                         (cols_arr >= 0) & (cols_arr < qa_arr.shape[1]))

            n_added_this_scene = 0
            for pt_idx, (i, j) in enumerate(ij_pairs):
                if not in_bounds[pt_idx]:
                    continue
                r, c = int(rows_arr[pt_idx]), int(cols_arr[pt_idx])

                qa_val = int(qa_arr[r, c])
                # QA_PIXEL bit flags (Collection 2 Level-2): bit1 dilated
                # cloud, bit3 cloud, bit4 cloud shadow -- mask if set.
                cloudy = bool((qa_val >> 1) & 1) or bool((qa_val >> 3) & 1) or bool((qa_val >> 4) & 1)
                if cloudy:
                    continue

                vals, ok = {}, True
                for band, arr in band_arrays.items():
                    dn = float(arr[r, c])
                    if dn <= 0:  # nodata / fill value
                        ok = False
                        break
                    vals[band] = float(np.clip(dn * SR_SCALE + SR_OFFSET, 0.0, 1.0))
                if not ok:
                    continue

                rows.append({"DATE": date_str, "pixel_id": f"{i}_{j}", "i": i, "j": j, **vals})
                n_added_this_scene += 1

            if n_added_this_scene > 0:
                n_scenes_used += 1
        except Exception as exc:
            warnings.warn(f"[Landsat] skipped scene {item.id}: {exc}")

    print(f"[Landsat] {n_scenes_used}/{len(items)} scene(s) contributed at least one "
          f"cloud-free pixel observation")

    if not rows:
        warnings.warn("Every candidate Landsat scene was fully cloud/shadow "
                       "masked at all grid pixels -- no usable observations "
                       "for this season. Returning an empty DataFrame.")
        return pl.DataFrame(schema=empty_schema)

    return pl.DataFrame(rows)


# ============================================================
# 5. CROP TYPE — USDA NASS Cropland Data Layer (CDL)
# ============================================================
# Added so SCYM (src/scym_model.py) only runs on pixels the USDA actually
# classifies as the target crop, matching Lobell et al. (2015) Sec. 2.2's
# CDL masking step: "For each pixel classified as maize and with at least
# one non-cloud contaminated observation within each observation window,
# ... these coefficients were then used to estimate yield for the pixel."
#
# Uses USDA NASS's free, public CDL point-value REST service -- no API key.
#
# IMPORTANT AVAILABILITY CAVEAT: CDL for a given growing season is not
# published until after that season's harvest (typically ~January-February
# of the following year). If you are running this pipeline *during* the
# season being modeled, the current year's CDL will not exist yet -- you
# will need to pass the most recently published year (commonly year - 1)
# as a proxy for current-season crop type, accepting the small risk that a
# pixel was rotated to a different crop since then. This function does not
# guess or interpolate a crop class in that situation; it simply queries
# whatever `year` you give it and reports what CDL says for that year.

# A convenience subset of CDL class codes for reference/extension -- NOT
# the full ~130-class legend. Full legend:
# https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/meta.php
CDL_LEGEND = {
    1: "Corn", 5: "Soybeans", 24: "Winter Wheat", 21: "Barley", 23: "Spring Wheat",
    27: "Rye", 28: "Oats", 36: "Alfalfa", 37: "Other Hay/Non Alfalfa",
    61: "Fallow/Idle Cropland", 111: "Open Water", 121: "Developed/Open Space",
    122: "Developed/Low Intensity", 123: "Developed/Med Intensity",
    124: "Developed/High Intensity", 131: "Barren", 141: "Deciduous Forest",
    142: "Evergreen Forest", 143: "Mixed Forest", 152: "Shrubland",
    176: "Grassland/Pasture", 190: "Woody Wetlands", 195: "Herbaceous Wetlands",
}

CDL_QUERY_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue"


def _normalize_cdl_target_crop(target_crop: str) -> tuple[str, ...]:
    """Return accepted CDL category tokens for the requested crop."""
    key = str(target_crop or "").strip().lower()
    aliases = {
        "corn": ("corn", "maize"),
        "maize": ("corn", "maize"),
        "soybean": ("soybean", "soybeans"),
        "soybeans": ("soybean", "soybeans"),
    }
    return aliases.get(key, (key,))


def _fetch_cdl_crop_mask_raster(grid_lats: np.ndarray, grid_lons: np.ndarray,
                                 year: int, target_crop: str,
                                 debug: bool = False) -> pl.DataFrame:
    """
    Primary CDL path: pull ONE small windowed raster clip via Planetary
    Computer's 'usda-cdl' STAC collection and vectorized-sample every grid
    pixel from it in memory -- one HTTP round trip total, regardless of
    grid size, using the same STAC + rasterio pattern already proven for
    Landsat above (fetch_landsat_bands_for_grid).

    UNVERIFIED IN A LIVE ENVIRONMENT: this repo's sandbox has no network
    access, so the exact collection id ('usda-cdl') and asset key for the
    classification band could not be confirmed against a live STAC call.
    This function inspects `item.assets` and tries several likely key
    names; if none match, it raises with the full list of available keys
    printed so you can hardcode the right one in `candidate_keys` below.
    Pass debug=True to also print them on a successful run.
    """
    import pystac_client
    import planetary_computer as pc
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import rowcol
    from pyproj import Transformer

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)

    min_lat, max_lat = float(np.min(grid_lats)), float(np.max(grid_lats))
    min_lon, max_lon = float(np.min(grid_lons)), float(np.max(grid_lons))
    pad = 0.01
    bbox = [min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad]

    search = catalog.search(collections=["usda-cdl"], bbox=bbox,
                             datetime=f"{year}-01-01/{year}-12-31")
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No item found in Planetary Computer's 'usda-cdl' collection for "
            f"{year} over this AOI -- the collection id or year coverage may "
            f"differ from what this function assumes."
        )
    item = items[0]

    candidate_keys = ["cdls", "data", "classification", "cropland", "cdl"]
    asset_key = next((k for k in candidate_keys if k in item.assets), None)
    if asset_key is None:
        asset_key = next(
            (k for k, a in item.assets.items()
             if "tiff" in (a.media_type or "").lower() or "image" in (a.media_type or "").lower()),
            None,
        )
    if asset_key is None:
        raise RuntimeError(
            f"Could not identify the CDL classification asset on STAC item "
            f"'{item.id}' -- available assets: {list(item.assets.keys())}. "
            f"Add the correct key to `candidate_keys` in _fetch_cdl_crop_mask_raster."
        )
    if debug:
        print(f"[CDL] item '{item.id}', using asset '{asset_key}'; "
              f"all assets: {list(item.assets.keys())}")

    n_rows, n_cols = grid_lats.shape
    flat_lats, flat_lons = grid_lats.ravel(), grid_lons.ravel()

    with rasterio.open(item.assets[asset_key].href) as ds:
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        xs, ys = transformer.transform(flat_lons, flat_lats)
        pad_m = 60.0
        window = from_bounds(float(np.min(xs)) - pad_m, float(np.min(ys)) - pad_m,
                              float(np.max(xs)) + pad_m, float(np.max(ys)) + pad_m,
                              transform=ds.transform).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise RuntimeError("Computed CDL raster window is empty -- check the AOI bbox.")
        arr = ds.read(1, window=window)
        win_transform = ds.window_transform(window)
        nodata = ds.nodata

    rows_arr, cols_arr = rowcol(win_transform, xs, ys)
    rows_arr = np.clip(np.asarray(rows_arr), 0, arr.shape[0] - 1)
    cols_arr = np.clip(np.asarray(cols_arr), 0, arr.shape[1] - 1)
    codes = arr[rows_arr, cols_arr]

    match_tokens = _normalize_cdl_target_crop(target_crop)
    target_codes = {code for code, name in CDL_LEGEND.items() if name.strip().lower() in match_tokens}
    if not target_codes:
        raise ValueError(
            f"target_crop='{target_crop}' doesn't match any entry in CDL_LEGEND "
            f"-- add its code there first (full legend link in this module's docstring)."
        )

    rows = []
    for idx, (i, j) in enumerate((i, j) for i in range(n_rows) for j in range(n_cols)):
        code = int(codes[idx])
        valid = nodata is None or code != nodata
        class_name = CDL_LEGEND.get(code, f"CDL code {code}")
        is_target = valid and code in target_codes
        rows.append({"pixel_id": f"{i}_{j}", "i": i, "j": j,
                     "cdl_class": class_name, "is_target_crop": is_target})

    result = pl.DataFrame(rows)
    n_match = int(result["is_target_crop"].sum())
    print(f"[CDL] {n_match}/{result.height} grid pixel(s) classified as '{target_crop}' "
          f"(CDL {year}, Planetary Computer 'usda-cdl', asset '{asset_key}')")
    return result


def _query_cdl_point(x: float, y: float, year: int,
                      session: requests.Session = None, debug: bool = False) -> str | None:
    """
    Legacy fallback helper: query USDA NASS's CDL point-value service for
    the crop-class name at one point. `x, y` must already be in the CDL's
    native projection -- USA Contiguous Albers Equal Area Conic (EPSG:5070),
    meters. Only used by _fetch_cdl_crop_mask_pointwise as a fallback when
    the raster-based path above fails; calling this once per pixel does
    NOT scale past a small handful of points -- see that function's docstring.

    Returns the category name string (e.g. 'Corn'), or None if the query
    failed or returned nothing usable. Parses defensively (proper XML
    first, then a regex fallback) because the exact tag names returned by
    this legacy service have varied across versions and were not
    verifiable against a live call in the environment this was written in.
    """
    client = session if session is not None else requests
    try:
        resp = client.get(CDL_QUERY_URL, params={"year": year, "x": x, "y": y}, timeout=20)
        resp.raise_for_status()
        if debug:
            print(f"[CDL] raw response for ({x:.0f}, {y:.0f}), {year}: {resp.text[:300]}")

        text = None
        try:
            root = ET.fromstring(resp.text)
            result_el = root.find(".//{*}Result")
            if result_el is not None and result_el.text:
                text = result_el.text
        except ET.ParseError:
            pass
        if not text:
            match = re.search(r"<Result>(.*?)</Result>", resp.text, re.IGNORECASE | re.DOTALL)
            text = match.group(1) if match else None

        category_match = re.search(
            r'category\s*:\s*["\']([^"\']+)["\']',
            text or "",
            re.IGNORECASE,
        )
        if category_match:
            return category_match.group(1).strip()
        return text.strip() if text else None
    except Exception as exc:
        warnings.warn(f"CDL: point query failed for ({x:.0f}, {y:.0f}), year {year} -- {exc}")
        return None


def _fetch_cdl_crop_mask_pointwise(grid_lats: np.ndarray, grid_lons: np.ndarray,
                                    year: int, target_crop: str,
                                    session: requests.Session = None,
                                    debug: bool = False) -> pl.DataFrame:
    """
    Fallback CDL path, used only if _fetch_cdl_crop_mask_raster fails:
    one legacy HTTP point-value request per grid pixel. This does NOT
    scale -- fine for a handful of points, unreliable (timeouts, transient
    failures silently skipped) for grids of a few hundred pixels or more,
    which is exactly the failure mode this pipeline hit before the
    raster-based path was added. Kept only as a documented last resort.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    n_rows, n_cols = grid_lats.shape
    match_tokens = _normalize_cdl_target_crop(target_crop)
    rows = []
    for i in range(n_rows):
        for j in range(n_cols):
            lat, lon = float(grid_lats[i, j]), float(grid_lons[i, j])
            x, y = transformer.transform(lon, lat)
            cdl_class = _query_cdl_point(x, y, year, session=session,
                                          debug=(debug and len(rows) < 3))
            class_text = cdl_class.strip().lower() if cdl_class else ""
            is_target = bool(class_text) and any(token in class_text for token in match_tokens)
            rows.append({"pixel_id": f"{i}_{j}", "i": i, "j": j,
                         "cdl_class": cdl_class, "is_target_crop": is_target})

    result = pl.DataFrame(rows)
    n_match = int(result["is_target_crop"].sum())
    print(f"[CDL] (legacy point-wise path) {n_match}/{result.height} grid pixel(s) "
          f"classified as '{target_crop}' (CDL {year})")
    return result


def fetch_cdl_crop_mask_for_grid(grid_lats: np.ndarray, grid_lons: np.ndarray,
                                  year: int, target_crop: str = "corn",
                                  session: requests.Session = None,
                                  prefer_raster: bool = True,
                                  debug: bool = False) -> pl.DataFrame:
    """
    Classify every grid pixel using the USDA NASS Cropland Data Layer (CDL)
    for the given year, and flag which ones match `target_crop`.

    Tries the fast, scalable raster-based path first (one windowed read
    from Planetary Computer's 'usda-cdl' STAC collection, vectorized-sampled
    for every pixel at once -- see _fetch_cdl_crop_mask_raster). If that
    path raises for any reason (collection/asset-key mismatch, network
    issue), falls back to the slower legacy per-pixel point-query service
    with a clear warning -- both paths use only real USDA CDL data, never
    a fabricated crop class.

    IMPORTANT AVAILABILITY CAVEAT: CDL for a given growing season is not
    published until after that season's harvest (typically ~January-
    February of the following year). If you are running this pipeline
    *during* the season being modeled, pass the most recently published
    year instead (commonly year - 1).

    :param grid_lats, grid_lons: 2D centroid arrays (as from build_grid_centroids)
    :param year: CDL year to query
    :param target_crop: crop name to match (e.g. 'corn', 'maize', 'soybeans')
    :param session: optional requests.Session, used only by the legacy fallback
    :param prefer_raster: set False to force the legacy per-pixel path
        (mainly for testing/comparison)
    :param debug: print diagnostic details for whichever path runs
    :return: Polars DataFrame with columns pixel_id, i, j, cdl_class,
        is_target_crop (bool)
    """
    if prefer_raster:
        try:
            result = _fetch_cdl_crop_mask_raster(grid_lats, grid_lons, year, target_crop, debug=debug)
        except Exception as exc:
            warnings.warn(
                f"[CDL] Raster-based fetch failed ({exc}); falling back to the "
                f"slower legacy per-pixel CDL service. Fix _fetch_cdl_crop_mask_raster "
                f"if this keeps happening (see its docstring)."
            )
            result = _fetch_cdl_crop_mask_pointwise(grid_lats, grid_lons, year, target_crop,
                                                      session=session, debug=debug)
    else:
        result = _fetch_cdl_crop_mask_pointwise(grid_lats, grid_lons, year, target_crop,
                                                  session=session, debug=debug)

    n_match = int(result["is_target_crop"].sum())
    if n_match == 0:
        warnings.warn(
            f"No grid pixel was classified as '{target_crop}' in CDL {year}. "
            f"Check that `year` is a published CDL year, that the grid is "
            f"actually over cropland, and (if debugging) rerun with debug=True."
        )
    return result


# ============================================================
# 6. SCYM SUPPORT — weather covariates & synthetic management ensemble
# ============================================================
# Added to support src/scym_model.py, which implements the "Scalable
# satellite-based Crop Yield Mapper" of Lobell, Thau, Seifert, Engle &
# Little (2015, Remote Sensing of Environment 164:324-333).
#
# NOTE ON THE WORD "SYNTHETIC" HERE: generate_scym_ensemble_specs() below
# perturbs *management assumptions* (sowing date, cultivar duration,
# planting density, starting soil moisture) that drive the crop model --
# it does NOT fabricate weather, soil, or satellite data. This is the SCYM
# method's calibration mechanism (paper Table 1, "step 1: crop model
# simulations") and is why SCYM needs no real yield-satellite pairs to
# calibrate. Each ensemble replicate is still run on the site's REAL
# weather and REAL soil profile; only the management scenario is varied.
# This is a different thing from, and should not be confused with, using
# a fabricated stand-in for weather/soil/imagery when a real data source
# is unavailable (see fetch_landsat_bands_for_grid() above, which fails
# loudly with an empty result + warning rather than inventing reflectance).
#
# The specific ranges below are reasonable placeholder values, loosely
# shaped by Table 2 of Lobell et al. (2015) -- they are NOT independently
# sourced from Kansas-specific agronomic literature, and the crop model
# here has no explicit density/cultivar parameters the way APSIM does, so
# "cultivar_scalar" and "density_scalar" are proxy knobs (thermal-time and
# SLA multipliers) invented to approximate those effects. Treat these as a
# starting point to refine with site-specific sources (e.g. K-State
# Research & Extension planting-date guidance, RMA final-plant-date data,
# published KS maize population trials) before relying on this for
# anything beyond a working prototype.


def compute_seasonal_weather_covariates(weather_dta: list[dict],
                                         window: tuple[int, int] | None = None) -> dict:
    """
    Aggregate the daily weather record into season- (or window-) level
    covariates for use as regression predictors in the SCYM yield model.

    Lobell et al. (2015, Eq. 1) use four such covariates for their Midwest
    maize application: June-August total rainfall, June-August mean solar
    radiation, July mean vapor-pressure deficit, and August mean daily
    maximum temperature. The Kansas Mesonet feed used by this pipeline does
    not currently carry a VPD field, so this function returns the three
    covariates that ARE derivable from what fetch_weather_from_mesonet
    provides (rainfall total, mean radiation, and a late-season maximum-
    temperature heat-stress proxy). Add a 'vpd' field to the Mesonet pull
    and a 'vpd_mean' aggregate here if/when that data becomes available.

    :param weather_dta: list-of-dicts as produced by fetch_weather_from_mesonet
    :param window: optional (start_day, end_day) day-index slice, where day 0
        is the first day of weather_dta (i.e. the planting date used for that
        run). If None, the whole record is aggregated.
    :return: dict with rain_total, radn_mean, tmax_mean, tmin_mean, tmean_mean
    """
    if window is not None:
        start, end = window
        start = max(0, start)
        end = min(len(weather_dta), end)
        record = weather_dta[start:end]
    else:
        record = weather_dta

    if not record:
        return {"rain_total": np.nan, "radn_mean": np.nan,
                "tmax_mean": np.nan, "tmin_mean": np.nan, "tmean_mean": np.nan}

    rain  = np.array([d.get("rain", 0.0) or 0.0 for d in record], dtype=float)
    radn  = np.array([d.get("solar_rad", np.nan) for d in record], dtype=float)
    tmax  = np.array([d.get("TMAX", d.get("T_max", np.nan)) for d in record], dtype=float)
    tmin  = np.array([d.get("TMIN", d.get("T_min", np.nan)) for d in record], dtype=float)
    tmean = np.array([d.get("T_mean", np.nan) for d in record], dtype=float)

    return {
        "rain_total": float(np.nansum(rain)),
        "radn_mean": float(np.nanmean(radn)),
        "tmax_mean": float(np.nanmean(tmax)),
        "tmin_mean": float(np.nanmean(tmin)),
        "tmean_mean": float(np.nanmean(tmean)),
    }


def generate_scym_ensemble_specs(n_reps: int, seed: int = SEED,
                                  config_file: dict = config) -> list[dict]:
    """
    Draw a stratified random ensemble of management/soil scenarios used to
    train the SCYM regression (Lobell et al. 2015, "step 1: crop model
    simulations"). Ranges loosely follow their Table 2 (APSIM maize/soybean
    simulation factors): sowing date, cultivar duration, planting density,
    and soil water at sowing. Fertilizer rate is not represented since the
    current crop model (src/crop_model.py) has no nitrogen response.

    Optional config.yaml block (all keys optional -- sensible defaults are
    used if the block, or individual keys, are absent):

        scym:
          planting_offset_days: [0, 20]    # later-sowing shift, in days
          cultivar_scalar:      [0.95, 1.05]  # scales TT thresholds (short/long cultivar)
          density_scalar:       [0.85, 1.15]  # scales SLA (denser stands -> more LAI per g leaf)
          init_sm_frac:         [0.60, 1.00]  # soil water at sowing, fraction of field capacity

    :param n_reps: number of ensemble replicates to generate
    :param seed: RNG seed (defaults to the pipeline-wide seed in config.yaml)
    :param config_file: custom config dict
    :return: list of n_reps dicts: planting_offset_days, cultivar_scalar,
             density_scalar, init_sm_frac
    """
    scym_cfg = config_file.get("scym", {}) if config_file else {}
    planting_range = scym_cfg.get("planting_offset_days", [0, 20])
    cultivar_range = scym_cfg.get("cultivar_scalar", [0.95, 1.05])
    density_range  = scym_cfg.get("density_scalar", [0.85, 1.15])
    sm_range       = scym_cfg.get("init_sm_frac", [0.60, 1.00])

    rng = np.random.default_rng(seed)
    specs = []
    for _ in range(n_reps):
        specs.append({
            "planting_offset_days": int(rng.integers(planting_range[0], planting_range[1] + 1)),
            "cultivar_scalar": float(rng.uniform(*cultivar_range)),
            "density_scalar": float(rng.uniform(*density_range)),
            "init_sm_frac": float(rng.uniform(*sm_range)),
        })
    return specs



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