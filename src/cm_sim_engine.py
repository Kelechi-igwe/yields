import polars as pl
import numpy as np
import requests
from src.crop_model import init_cell_state, step
from src.data_pull import config


def run_grid_simulation(weather_data: list[dict], ET_stack: np.ndarray,
                        soil_layers_grid: list, config_file: dict = config) -> pl.DataFrame:

    params = config_file["crop_parameters"]
    alloc = config_file["biomass_allo_stages"]
    init_template = config_file["initial_state_template"]

    n_rows = len(soil_layers_grid)
    n_cols = len(soil_layers_grid[0])
    days_to_run = min(len(weather_data), ET_stack.shape[0])

    # Initialise one state dict per cell
    cell_states = [
        [init_cell_state(init_template, soil_layers_grid[i][j]) for j in range(n_cols)]
        for i in range(n_rows)
    ]

    results = []

    # Run spatial simulation
    for day in range(days_to_run):
        wx = weather_data[day]
        date_val = str(wx["DATE"])

        for i in range(n_rows):
            for j in range(n_cols):
                et_val = ET_stack[day, i, j]
                et_override = None if np.isnan(et_val) else float(et_val)

                # stateless operational step
                cell_states[i][j] = step(
                    weather=wx,
                    state=cell_states[i][j],
                    params=params,
                    alloc=alloc,
                    ET_override=et_override
                )
                s = cell_states[i][j]

                results.append((
                    int(day),
                    str(date_val),
                    int(i),
                    int(j),
                    f"{i}_{j}",
                    float(s["soil_layers"][0]["theta_fc"]),
                    float(s["soil_layers"][0]["theta_wp"]),
                    str(s["stage"]),
                    float(s["TT"]),
                    float(s["LAI"]),
                    float(s["B"]),
                    float(s["B_leaf"]),
                    float(s["B_grain"]),
                    float(s["ET"]),
                    float(s["ET_pot"]),
                    float(s["f_water"]),
                    float(s["f_water_soil"]),
                    float(s["f_water_et"]),
                    float(s["Zr"]),
                    float(s["SM_total"]),
                    float(s["VWC"]) if not np.isnan(s["VWC"]) else None,
                    float(s["rainfall"])
                ))

    schema = [
        ("day", pl.Int32), ("DATE", pl.String), ("i", pl.Int32), ("j", pl.Int32), ("pixel_id", pl.String),
        ("theta_fc", pl.Float64), ("theta_wp", pl.Float64), ("stage", pl.String), ("TT", pl.Float64), ("LAI", pl.Float64),
        ("Biomass", pl.Float64), ("Leaf_Biomass", pl.Float64), ("Grain", pl.Float64), ("ET", pl.Float64), ("ET_pot", pl.Float64),
        ("f_water", pl.Float64), ("f_water_soil", pl.Float64), ("f_water_et", pl.Float64), ("Zr", pl.Float64),  ("SM_total", pl.Float64),
        ("VWC", pl.Float64), ("Rainfall", pl.Float64)
    ]

    # put results into polars frame
    return pl.DataFrame(results, schema=schema, orient="row")
