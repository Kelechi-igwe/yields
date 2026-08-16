

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
from src.data_pull import (fetch_weather_from_mesonet, build_grid_centroids,
                            fetch_et_stack_from_openet,
                            generate_et_stack_synthetic, fetch_ssurgo_soil_for_bbox,
                            assign_mukeys_to_grid, build_ssurgo_soil_layers_grid, create_pipeline_sesh
                            )
from src.prosail_model import add_solar_geometry, map_to_prosail_params
from src.pros_sim_engine import run_prosail_grid, extract_landsat_bands_and_indices

from src.plotting import generate_pipeline_plots

load_dotenv()

openet_api_key = os.getenv('OPENET_API_KEY')
pipeline_session = create_pipeline_sesh(openet_api_key)

ROOT = Path(__file__).resolve().parent

config_path = ROOT / 'config.yaml'

with open(config_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)


PLANTING_DATE = "2025-05-15"  # first day of simulation
SEASON_LENGTH = 180  # days to simulate
YEAR = 2025  # year used to filter Mesonet data

MESONET_STATION = "Manhattan"  # nearest Mesonet station name

START_DATE = pd.to_datetime(PLANTING_DATE)
END_DATE = START_DATE + timedelta(days=SEASON_LENGTH - 1)

BBOX = {"min_lon": -98.55, "min_lat": 38.30, "max_lon": -98.50, "max_lat": 38.35}
SITE_LAT, SITE_LON = 38.32, -98.52  # for solar geometry in PROSAIL

GRID_ROWS = 3  # spatial rows  (N→S)
GRID_COLS = 3

t0_refactored = time.perf_counter()

weather_df, weather_dta = fetch_weather_from_mesonet(station=MESONET_STATION,
                                            start=START_DATE,
                                            end=END_DATE,
                                            year_filter=YEAR,
                                            planting_date=PLANTING_DATE,
                                            config_file=config)

print('Got weather data')

grid_lats, grid_lons = build_grid_centroids(bbox=BBOX,
                                            n_rows=GRID_ROWS,
                                            n_cols=GRID_COLS)

print('Got grid lats and lons')

try:
    ET_stack, _ = fetch_et_stack_from_openet(grid_lats=grid_lats,
                                            grid_lons=grid_lons,
                                            start_date=START_DATE,
                                            end_date=END_DATE,
                                            api_key=openet_api_key,
                                            season_length=SEASON_LENGTH)

    if ET_stack is None or np.isnan(ET_stack).all():
        raise ValueError("OpenET API returned invalid data arrays due to upstream token errors.")

except Exception as e:
    print(f"\n[Warning] OpenET Ingestion Failed ({e}). Falling back to deterministic synthetic generation...")
    # fallback keeps if offline
    ET_stack = generate_et_stack_synthetic(n_rows=GRID_ROWS,
                                            n_cols=GRID_COLS,
                                            season_length=SEASON_LENGTH)

print('Got ET')

soil_dta = fetch_ssurgo_soil_for_bbox(bbox=BBOX,
                                        session=pipeline_session,
                                        debug=False,
                                        config_file=config)

print('Got soil data')

soil_layers_grid = None
soil_summary_grid = {
    "theta_fc": np.full((GRID_ROWS, GRID_COLS), 0.30),
    "theta_wp": np.full((GRID_ROWS, GRID_COLS), 0.12),
}

if soil_dta is not None and len(soil_dta) > 0:
    mukey_grid = assign_mukeys_to_grid(grid_lats, grid_lons, soil_dta)
    soil_layers_grid, soil_summary_grid = build_ssurgo_soil_layers_grid(soil_dta, mukey_grid)

print('Got soil layer grid')
# print(soil_layers_grid)

cm_result = run_grid_simulation(weather_data=weather_dta,
                                ET_stack=ET_stack,
                                soil_layers_grid=soil_layers_grid,
                                config_file=config)

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

# print(f"NDVI range : {df_out.select('NDVI').min():.3f} – {df_out.select('NDVI').max():.3f}")
# print(f"SAVI range : {df_out.select('SAVI').min():.3f} – {df_out.select('SAVI').max():.3f}")
print(df_out.head())


generate_pipeline_plots(
    weather_df=weather_df,
    et_stack=ET_stack,
    soil_summary_grid=soil_summary_grid,
    cm_result=cm_result,
    df_out=df_out,
    output_dir=ROOT / "plots"
)

print(f"Plots saved to {ROOT / 'plots'}")



total_refactored = time.perf_counter() - t0_refactored

print(total_refactored)

# print(cm_result.head())