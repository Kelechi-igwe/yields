


from pathlib import Path
from datetime import timedelta
import pandas as pd
import polars as pl
import numpy as np
import yaml
from dotenv import load_dotenv
import os
import time

from src.cm_sim_engine import run_grid_simulation
from src.data_pull import (fetch_weather_from_mesonet, build_landsat_aligned_grid,
                            fetch_et_stack_from_openet,
                            fetch_ssurgo_soil_for_bbox,
                            assign_mukeys_to_grid, fetch_landsat_bands_for_grid,
                            fetch_cdl_crop_mask_for_grid,
                            build_ssurgo_soil_layers_grid, create_pipeline_sesh
                            )
from src.prosail_model import add_solar_geometry, map_to_prosail_params
from src.pros_sim_engine import run_prosail_grid, extract_landsat_bands_and_indices
from src.scym_model import run_scym_pipeline

from src.plotting import generate_pipeline_plots, save_scym_yield_geotiff

load_dotenv()

# Load credentials and create one session for external data services.
openet_api_key = os.getenv('OPENET_API_KEY')
pipeline_session = create_pipeline_sesh(openet_api_key)

ROOT = Path(__file__).resolve().parent

config_path = ROOT / 'config.yaml'

with open(config_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)


# Analysis period and weather station.
PLANTING_DATE = "2025-05-15"
SEASON_LENGTH = 180  # days to simulate
YEAR = 2025  # year used to filter Mesonet data

MESONET_STATION = "Manhattan"  # nearest Mesonet station name

START_DATE = pd.to_datetime(PLANTING_DATE)
END_DATE = START_DATE + timedelta(days=SEASON_LENGTH - 1)

# Spatial extent of the field (WGS84 longitude/latitude).
# BBOX = {
#     "min_lon": -97.2196413,
#     "min_lat": 39.4082070,
#     "max_lon": -97.2084830,
#     "max_lat": 39.4127515,
# }

BBOX = {
    "min_lon": -97.2189625,
    "min_lat": 39.4115753,
    "max_lon": -97.2175467,
    "max_lat": 39.4125774,
}



SITE_LAT = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
SITE_LON = (BBOX["min_lon"] + BBOX["max_lon"]) / 2  # for solar geometry in PROSAIL

# GRID_ROWS / GRID_COLS are no longer chosen up front -- they're derived
# below from the real Landsat pixel grid covering BBOX (see
# build_landsat_aligned_grid). Every simulated "cell" is therefore an
# actual 30 m USGS Landsat pixel, not an approximation of one.

t0_refactored = time.perf_counter()

# Retrieve and trim daily weather to the simulated season.
weather_df, weather_dta = fetch_weather_from_mesonet(station=MESONET_STATION,
                                            start=START_DATE,
                                            end=END_DATE,
                                            year_filter=YEAR,
                                            planting_date=PLANTING_DATE,
                                            config_file=config)

print('Got weather data')

# Use the reference Landsat scene to define the operational pixel grid.
grid_lats, grid_lons, grid_metadata = build_landsat_aligned_grid(
    bbox=BBOX, return_metadata=True
)
GRID_ROWS, GRID_COLS = grid_lats.shape

print(f'Got grid lats and lons ({GRID_ROWS} x {GRID_COLS} = {GRID_ROWS * GRID_COLS} real Landsat pixels)')

# Retrieve daily ET for every grid cell.
ET_stack, et_any_success = fetch_et_stack_from_openet(grid_lats=grid_lats,
                                        grid_lons=grid_lons,
                                        start_date=START_DATE,
                                        end_date=END_DATE,
                                        api_key=openet_api_key,
                                        season_length=SEASON_LENGTH)

if not et_any_success or ET_stack is None or np.isnan(ET_stack).all():
    raise RuntimeError(
        "OpenET returned no usable ET data for any grid cell in this "
        "date range. This pipeline does not substitute synthetic ET -- "
        "check OPENET_API_KEY, the requested date range/bbox, and OpenET "
        "service status before re-running."
    )

print('Got ET')

# Retrieve SSURGO soil data and assign a soil profile to each pixel.
soil_dta = fetch_ssurgo_soil_for_bbox(bbox=BBOX,
                                        session=pipeline_session,
                                        debug=False,
                                        config_file=config)

print('Got soil data')

if soil_dta is None or len(soil_dta) == 0:
    raise RuntimeError(
        "SSURGO/Soil Data Access returned no soil records for this bbox. "
        "This pipeline does not substitute placeholder/uniform soil -- "
        "check the bbox is within SSURGO survey coverage and that the SDA "
        "service is reachable before re-running."
    )

mukey_grid = assign_mukeys_to_grid(grid_lats, grid_lons, soil_dta,
                                    session=pipeline_session, config_file=config)
soil_layers_grid, soil_summary_grid = build_ssurgo_soil_layers_grid(soil_dta, mukey_grid)

print('Got soil layer grid')

# Run the spatial crop model with the observed weather, ET, and soil inputs.
cm_result = run_grid_simulation(weather_data=weather_dta,
                                ET_stack=ET_stack,
                                soil_layers_grid=soil_layers_grid,
                                config_file=config)

# Generate synthetic Landsat reflectance from crop-model states.
print("Running PROSAIL forward model ...")
df_ps  = add_solar_geometry(cm_result, lat=SITE_LAT, lon=SITE_LON)
df_ps  = map_to_prosail_params(df_ps)
refl   = run_prosail_grid(df_ps)
bands  = extract_landsat_bands_and_indices(refl)
df_out = pl.concat([df_ps, bands], how="horizontal_extend")

print(f"PROSAIL complete.")

stats = df_out.select(
    min_ndvi = pl.col('NDVI').min(),
    max_ndvi = pl.col('NDVI').max(),
    min_savi = pl.col('SAVI').min(),
    max_savi = pl.col('SAVI').max()
)

print(f"NDVI range : {stats['min_ndvi'].item():.3f} – {stats['max_ndvi'].item():.3f}")
print(f"SAVI range : {stats['min_savi'].item():.3f} – {stats['max_savi'].item():.3f}")

print(df_out.head())



# SCYM uses observed Landsat reflectance; PROSAIL output remains separate.
print("\nFetching real Landsat surface reflectance for SCYM ...")

scym_cfg = config.get("scym", {})

landsat_bands_df = fetch_landsat_bands_for_grid(
    grid_lats=grid_lats,
    grid_lons=grid_lons,
    start_date=START_DATE,
    end_date=END_DATE,
    max_cloud_cover=scym_cfg.get("max_cloud_cover", 70.0),
)

print("Fetching CDL crop-type classification ...")
# Use the latest available CDL year because the current season may not yet
# have been released; override this value in config['scym']['cdl_year'].
CDL_YEAR = scym_cfg.get("cdl_year", YEAR - 1)
TARGET_CROP = scym_cfg.get("target_crop", "maize")
crop_mask_df = fetch_cdl_crop_mask_for_grid(
    grid_lats=grid_lats,
    grid_lons=grid_lons,
    year=CDL_YEAR,
    target_crop=TARGET_CROP,
    # Use a separate client because this legacy endpoint can return HTTP 500s.
    session=None,
)

# Train SCYM on simulated management scenarios and apply it to Landsat GCVI.
print("Running SCYM (Lobell et al. 2015) yield estimation ...")
scym_df, scym_model_fit = run_scym_pipeline(
    weather_dta=weather_dta,
    soil_layers_grid=soil_layers_grid,
    cm_result=cm_result,
    landsat_bands_df=landsat_bands_df,
    planting_date=PLANTING_DATE,
    crop_mask_df=crop_mask_df,
    config_file=config,
    early_window=tuple(scym_cfg.get("early_window", (40, 75))),
    late_window=tuple(scym_cfg.get("late_window", (80, 115))),
    n_reps=scym_cfg.get("n_reps", 40),
    crop="maize",
)

print(f"SCYM training R^2 : {scym_model_fit['r2']:.3f}  (n = {scym_model_fit['n_train']} replicates)")
print(scym_df.select(["pixel_id", "cdl_class", "GCVI_early", "GCVI_late", "SCYM_yield_t_ha", "true_yield_t_ha"]))



generate_pipeline_plots(
    weather_df=weather_df,
    et_stack=ET_stack,
    soil_summary_grid=soil_summary_grid,
    cm_result=cm_result,
    df_out=df_out,
    scym_df=scym_df,
    scym_model=scym_model_fit,
    grid_rows=GRID_ROWS,
    grid_cols=GRID_COLS,
    output_dir=ROOT / "plots"
)

scym_geotiff_path = ROOT / "plots" / "09_scym_yield_map.tif"
save_scym_yield_geotiff(
    scym_df=scym_df,
    grid_rows=GRID_ROWS,
    grid_cols=GRID_COLS,
    grid_crs=grid_metadata["crs"],
    grid_transform=grid_metadata["transform"],
    output_path=scym_geotiff_path,
)

print(f"Plots saved to {ROOT / 'plots'}")
print(f"SCYM GeoTIFF saved to {scym_geotiff_path}")



total_refactored = time.perf_counter() - t0_refactored
print(total_refactored)