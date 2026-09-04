"""
scym_model.py
=============
Implementation of the SCYM method ("Scalable satellite-based Crop Yield
Mapper") of Lobell, Thau, Seifert, Engle & Little (2015), Remote Sensing of
Environment 164:324-333.

SCYM has four steps (paper Table 1), each with a function/section below:

  1. Crop model simulations   -> build_training_table() drives the existing
     point-scale crop model (src/crop_model.py) across a synthetic ensemble
     of sowing dates, cultivars, planting densities, and starting soil
     moisture (src/data_pull.generate_scym_ensemble_specs), producing daily
     LAI and a final simulated grain yield for each replicate. Each
     replicate runs on the site's REAL weather and REAL soil profile, 
     so the ensemble is a realistic representation of what a satellite would see 
     under the actual conditions of the season, but with a range of management scenarios.

  2. Pseudo-observations      -> lai_to_gcvi() converts each replicate's
     daily LAI into GCVI using the published LAI-GCVI relationship
     (Nguy-Robertson et al., 2012), exactly the way the paper turns crop
     model output into a proxy for what a satellite would see.

  3. Regression calibration   -> fit_scym_regression() fits a multiple
     linear regression of simulated yield on windowed GCVI and seasonal
     weather (paper Eq. 1).

  4. Yield estimation         -> run_scym_pipeline() applies that regression
     to REAL, per-pixel Landsat-observed GCVI (fetched via
     src.data_pull.fetch_landsat_bands_for_grid, Microsoft Planetary
     Computer STAC catalog) and the
     season's real weather, to produce a per-pixel SCYM yield estimate. The
     crop model's own "true" simulated yield is attached for comparison.

This module calls the existing crop-model, data-access, and vegetation-index
functions; it does not duplicate their internals.
"""

import copy

import numpy as np
import pandas as pd
import polars as pl

from src.crop_model import init_cell_state, step
from src.data_pull import (
    config as _default_config,
    compute_seasonal_weather_covariates,
    generate_scym_ensemble_specs,
)
from src.pros_sim_engine import compute_vegetation_indices_from_bands

# ------------------------------------------------------------------------
# SCYM step 2 — pseudo-observations: LAI -> GCVI
# ------------------------------------------------------------------------

# Nguy-Robertson et al. (2012), as used in Lobell et al. (2015) Eq. 3 (maize)
# and Eq. 5 (soybean).
_LAI_TO_GCVI_COEFFS = {
    "maize":   {"a": 1.4, "b": 1.03, "c": 0.93},
    "soybean": {"a": 1.4, "b": 1.30, "c": 1.10},
}


def lai_to_gcvi(lai, crop: str = "maize"):
    """
    Empirical LAI -> GCVI "pseudo-observation" relationship (SCYM step 2).

    GCVI = a * LAI^b + c

    :param lai: scalar or array of simulated LAI values (m^2/m^2)
    :param crop: 'maize' or 'soybean' (coefficient set to use)
    :return: GCVI, same shape as `lai`
    """
    if crop not in _LAI_TO_GCVI_COEFFS:
        raise ValueError(
            f"No LAI->GCVI relationship defined for crop '{crop}'. "
            f"Available: {list(_LAI_TO_GCVI_COEFFS)}"
        )
    coeffs = _LAI_TO_GCVI_COEFFS[crop]
    lai_arr = np.asarray(lai, dtype=float)
    return coeffs["a"] * lai_arr ** coeffs["b"] + coeffs["c"]


# ------------------------------------------------------------------------
# SCYM step 1 — crop model simulations (training ensemble)
# ------------------------------------------------------------------------

def _apply_ensemble_perturbation(params: dict, spec: dict) -> dict:
    """
    Return a perturbed copy of the crop-parameter dict for one ensemble
    replicate: `cultivar_scalar` stretches/compresses the thermal-time
    stage thresholds (short vs. long cultivar), and `density_scalar` scales
    SLA (a denser stand builds more leaf area per gram of leaf biomass).
    """
    p = copy.deepcopy(params)
    cultivar_scalar = spec["cultivar_scalar"]
    for key in ("TT_emergence", "TT_veg_end", "TT_grainfill", "TT_maturity"):
        if key in p:
            p[key] = p[key] * cultivar_scalar
    if "SLA" in p:
        p["SLA"] = p["SLA"] * spec["density_scalar"]
    return p


def _run_single_cell_series(weather_dta: list[dict], soil_layers: list,
                             params: dict, alloc: dict, init_template: dict,
                             init_sm_frac: float = 0.70) -> pl.DataFrame:
    """
    Run the point-scale crop model (src/crop_model.step) for one
    parameter/soil combination over the full supplied weather record and
    return the daily state trace as a Polars DataFrame with columns
    day, LAI, Grain, stage. This is the single-cell analogue of
    src/cm_sim_engine.run_grid_simulation, used here because the SCYM
    training ensemble needs many independent point-scale runs rather than
    one spatial grid.
    """
    state = init_cell_state(init_template, soil_layers, sm_frac=init_sm_frac)
    trace = []
    for day, wx in enumerate(weather_dta):
        state = step(weather=wx, state=state, params=params, alloc=alloc)
        trace.append({"day": day, "LAI": float(state["LAI"]),
                       "Grain": float(state["B_grain"]), "stage": str(state["stage"])})
    return pl.DataFrame(trace)


def build_training_table(weather_dta: list[dict], soil_layers: list,
                          config_file: dict,
                          early_window: tuple[int, int],
                          late_window: tuple[int, int],
                          n_reps: int = 40,
                          seed: int | None = None,
                          crop: str = "maize") -> pl.DataFrame:
    """
    SCYM steps 1-2 combined: run a stratified ensemble of point-scale
    crop-model replicates, convert each replicate's daily LAI into
    pseudo-GCVI at the two Landsat-like compositing windows (maximum value
    within each window, exactly as the paper composites real Landsat
    scenes), and pair those with season-level weather covariates and the
    crop model's own simulated grain yield.

    Sowing-date variation is implemented by trimming the front of the
    weather record (later "planting"), since the pipeline has only one
    real weather series to draw from -- there is no way to simulate an
    *earlier* start than the first available Mesonet day. Replicates whose
    shifted weather record is too short to reach the late-season window are
    skipped.

    :param weather_dta: full-season weather list-of-dicts (e.g. `weather_dta`
        as returned by fetch_weather_from_mesonet), starting at the nominal
        planting date
    :param soil_layers: one representative soil profile (list of layer dicts)
        for the site — the ensemble varies management/soil-moisture, not
        spatial soil texture
    :param config_file: pipeline config dict (needs crop_parameters,
        biomass_allo_stages, initial_state_template, data_specs.seed)
    :param early_window: (start_day, end_day) day-index window for the
        "early-season" GCVI composite
    :param late_window: (start_day, end_day) day-index window for the
        "late-season" GCVI composite
    :param n_reps: number of ensemble replicates to attempt
    :param seed: RNG seed override (defaults to config_file's pipeline seed)
    :param crop: 'maize' or 'soybean' — selects the LAI->GCVI relationship
    :return: Polars DataFrame, one row per usable replicate, with columns
        GCVI_early, GCVI_late, rain_total, radn_mean, tmax_mean, yield_t_ha,
        plus the ensemble spec columns for traceability
    """
    seed = seed if seed is not None else config_file["data_specs"]["seed"]
    specs = generate_scym_ensemble_specs(n_reps=n_reps, seed=seed, config_file=config_file)

    base_params   = config_file["crop_parameters"]
    alloc         = config_file["biomass_allo_stages"]
    init_template = config_file["initial_state_template"]

    rows = []
    for spec in specs:
        offset = spec["planting_offset_days"]
        wx_slice = weather_dta[offset:]
        if len(wx_slice) <= late_window[1]:
            continue  # not enough weather left after this sowing-date shift

        params = _apply_ensemble_perturbation(base_params, spec)
        trace_df = _run_single_cell_series(
            wx_slice, soil_layers, params, alloc, init_template,
            init_sm_frac=spec["init_sm_frac"],
        )

        lai_early = trace_df.filter(pl.col("day").is_between(*early_window))["LAI"].to_numpy()
        lai_late  = trace_df.filter(pl.col("day").is_between(*late_window))["LAI"].to_numpy()
        if lai_early.size == 0 or lai_late.size == 0:
            continue

        gcvi_early = float(np.nanmax(lai_to_gcvi(lai_early, crop=crop)))
        gcvi_late  = float(np.nanmax(lai_to_gcvi(lai_late, crop=crop)))

        w = compute_seasonal_weather_covariates(wx_slice)
        final_grain = float(trace_df["Grain"].to_numpy()[-1])
        yield_t_ha = final_grain * 0.01 / (1 - 0.155)  # g/m^2 -> t/ha at 15.5% moisture

        rows.append({
            "GCVI_early": gcvi_early,
            "GCVI_late": gcvi_late,
            "rain_total": w["rain_total"],
            "radn_mean": w["radn_mean"],
            "tmax_mean": w["tmax_mean"],
            "yield_t_ha": yield_t_ha,
            **spec,
        })

    if not rows:
        raise RuntimeError(
            "SCYM training ensemble produced zero usable replicates -- check "
            "that weather_dta is long enough to cover late_window for the "
            "requested planting_offset_days range in config['scym']."
        )
    return pl.DataFrame(rows)


# ------------------------------------------------------------------------
# SCYM step 3 — regression calibration
# ------------------------------------------------------------------------

# Main-effect predictors from Eq. 1: two GCVI compositing dates (RM) and
# three weather covariates (W) -- see compute_seasonal_weather_covariates()
# for why this is three rather than Lobell et al.'s four (no VPD field in
# the current weather feed).
_FEATURE_COLS = ["GCVI_early", "GCVI_late", "rain_total", "radn_mean", "tmax_mean"]


def fit_scym_regression(training_df: pl.DataFrame,
                         feature_cols: list[str] = _FEATURE_COLS,
                         target_col: str = "yield_t_ha",
                         ridge_alpha: float = 1.0) -> dict:
    """
    SCYM step 3: multiple linear regression of simulated yield on
    pseudo-GCVI and seasonal weather (Lobell et al. 2015, Eq. 1), plus one
    GCVI(late) x Tmax(late) interaction term standing in for the paper's
    W x RM interaction -- it captures how a hot late season changes what a
    high late-season GCVI implies about eventual yield.

    Fit with ridge-regularized least squares (L2 penalty on all
    coefficients except the intercept), not plain OLS. With only a few
    dozen training replicates and 7 correlated coefficients (GCVI_early,
    GCVI_late, and their interaction all move together), unregularized OLS
    was overfitting so hard that training R^2 hit 0.997-1.00 and, worse,
    the fitted coefficients blew up when applied to real operational GCVI
    values -- producing yield "estimates" over 40 t/ha for a crop whose
    world-record irrigated yield is around 27 t/ha. Ridge trades a little
    training fit for coefficients that behave sanely out of sample.

    `ridge_alpha` is a heuristic default, not cross-validated -- if you
    have time, tune it (e.g. leave-one-out CV over the training ensemble)
    rather than trusting 1.0 blindly. Increase it further if operational
    predictions are still implausible; decrease it if the fit degrades
    too much (check r2 and the training-fit scatter plot).

    :param training_df: output of build_training_table()
    :param feature_cols: main-effect predictor columns to include
    :param target_col: response column name
    :param ridge_alpha: L2 penalty strength (0 = plain OLS)
    :return: dict with feature_names, coefficients, r2, n_train, y_true,
        y_pred, ridge_alpha, and the training GCVI_early/GCVI_late ranges
        (used by run_scym_pipeline to flag operational pixels that fall
        outside the range the regression was ever trained on)
    """
    df = training_df.to_pandas()

    df = df.assign(GCVI_late_x_tmax=df["GCVI_late"] * df["tmax_mean"])
    x_cols = list(feature_cols) + ["GCVI_late_x_tmax"]

    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in x_cols])
    y = df[target_col].to_numpy()

    n_features = X.shape[1]
    penalty = ridge_alpha * np.eye(n_features)
    penalty[0, 0] = 0.0  # never penalize the intercept
    coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    y_hat = X @ coefficients

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "feature_names": ["intercept"] + x_cols,
        "coefficients": coefficients,
        "r2": r2,
        "n_train": len(df),
        "y_true": y,
        "y_pred": y_hat,
        "ridge_alpha": ridge_alpha,
        "gcvi_early_range": (float(df["GCVI_early"].min()), float(df["GCVI_early"].max())),
        "gcvi_late_range": (float(df["GCVI_late"].min()), float(df["GCVI_late"].max())),
    }


# ------------------------------------------------------------------------
# SCYM step 4 — yield estimation
# ------------------------------------------------------------------------

def predict_scym_yield(model: dict, gcvi_early, gcvi_late,
                        rain_total, radn_mean, tmax_mean):
    """
    SCYM step 4: apply a fitted regression to (satellite-derived) GCVI and
    gridded weather to estimate yield, vectorised over an arbitrary number
    of pixels. Argument/column order must match fit_scym_regression()'s
    `_FEATURE_COLS` + interaction term.

    :param model: dict returned by fit_scym_regression()
    :param gcvi_early, gcvi_late: per-pixel windowed max GCVI
    :param rain_total, radn_mean, tmax_mean: seasonal weather covariates
        (broadcast the same scalar to every pixel if only one weather
        station/series drives the whole grid, as in this pipeline)
    :return: np.ndarray of predicted yield (t/ha), one value per pixel
    """
    gcvi_early = np.asarray(gcvi_early, dtype=float)
    gcvi_late  = np.asarray(gcvi_late, dtype=float)
    rain_total = np.broadcast_to(np.asarray(rain_total, dtype=float), gcvi_early.shape)
    radn_mean  = np.broadcast_to(np.asarray(radn_mean, dtype=float), gcvi_early.shape)
    tmax_mean  = np.broadcast_to(np.asarray(tmax_mean, dtype=float), gcvi_early.shape)
    interaction = gcvi_late * tmax_mean

    X = np.column_stack([
        np.ones_like(gcvi_early), gcvi_early, gcvi_late,
        rain_total, radn_mean, tmax_mean, interaction,
    ])
    return X @ model["coefficients"]


# ------------------------------------------------------------------------
# End-to-end orchestrator
# ------------------------------------------------------------------------

def run_scym_pipeline(weather_dta: list[dict],
                       soil_layers_grid: list,
                       cm_result: pl.DataFrame,
                       landsat_bands_df: pl.DataFrame,
                       planting_date,
                       crop_mask_df: pl.DataFrame,
                       config_file: dict = _default_config,
                       early_window: tuple[int, int] = (40, 75),
                       late_window: tuple[int, int] = (80, 115),
                       n_reps: int = 40,
                       crop: str = "maize") -> tuple[pl.DataFrame, dict]:
    """
    End-to-end SCYM run for the current field/season.

      1. Train the SCYM regression on a synthetic management ensemble driven
         by a representative soil profile for the site.
      2. Restrict to pixels the USDA NASS Cropland Data Layer actually
         classifies as `crop` (paper Sec. 2.2's CDL masking step) --
         src.data_pull.fetch_cdl_crop_mask_for_grid.
      3. Composite REAL, per-pixel Landsat GCVI
         (src.data_pull.fetch_landsat_bands_for_grid ->
         src.pros_sim_engine.compute_vegetation_indices_from_bands) into the
         same two windows used for training (max-value compositing across
         however many cloud-free scenes fall in each window, exactly as the
         paper composites real Landsat scenes -- no simulated reflectance
         is involved here).
      4. Apply the trained regression to those per-pixel GCVI values plus
         the season's actual weather to obtain a per-pixel SCYM yield
         estimate.
      5. Attach the crop model's own "true" simulated yield for each pixel
         so the SCYM estimate can be checked against it (paper Fig. 4/9
         style validation, here used as an internal consistency check
         rather than against independent farmer-reported yields).

    :param weather_dta: the season's actual weather list-of-dicts (same one
        passed to run_grid_simulation)
    :param soil_layers_grid: the operational grid's per-cell soil layers
        (used to pick one representative profile for ensemble training)
    :param cm_result: output of run_grid_simulation (for the "true" yield)
    :param landsat_bands_df: output of
        src.data_pull.fetch_landsat_bands_for_grid() -- one row per
        (scene date, pixel) with DATE, pixel_id, i, j, blue, green, red,
        nir(, swir16) columns.
    :param planting_date: the season's planting date (anything
        pd.to_datetime accepts) -- used to convert each Landsat scene's
        calendar date into a day-of-season index comparable to
        early_window/late_window
    :param crop_mask_df: output of
        src.data_pull.fetch_cdl_crop_mask_for_grid() -- one row per pixel
        with pixel_id, i, j, cdl_class, is_target_crop. Required: SCYM is
        only valid where the pixel is actually the crop the regression was
        trained for, and this pipeline does not skip that check.
    :param config_file: pipeline config dict
    :param early_window, late_window: day-index compositing windows,
        shared between training and application
    :param n_reps: size of the SCYM training ensemble
    :param crop: 'maize' or 'soybean'
    :return: (scym_df, model) -- per-pixel results table (including the
        cdl_class each surviving pixel was classified as, for audit) and
        the fitted regression dict from fit_scym_regression() (kept for
        diagnostics/plots)
    """
    if landsat_bands_df.height == 0:
        raise RuntimeError(
            "landsat_bands_df is empty -- no usable (cloud-free) Landsat "
            "observations were found for this grid/season. SCYM cannot "
            "produce a real per-pixel estimate without them; widen the "
            "date range or max_cloud_cover in fetch_landsat_bands_for_grid() "
            "rather than substituting simulated reflectance."
        )

    if not soil_layers_grid:
        raise ValueError(
            "soil_layers_grid is empty/None -- SCYM requires a real, "
            "SSURGO-derived soil profile for the training ensemble (see "
            "src.data_pull.fetch_ssurgo_soil_for_bbox / "
            "build_ssurgo_soil_layers_grid). This pipeline does not "
            "substitute a fabricated uniform soil profile."
        )
    site_soil = soil_layers_grid[0][0]

    # Keep only pixels classified as the target crop in the CDL.
    target_pixels = crop_mask_df.filter(pl.col("is_target_crop"))
    target_pixel_ids = target_pixels["pixel_id"].to_list()
    if not target_pixel_ids:
        raise RuntimeError(
            f"No grid pixel is CDL-classified as '{crop}' -- SCYM has "
            f"nothing to run on. Check crop_mask_df, the CDL year passed "
            f"to fetch_cdl_crop_mask_for_grid(), and target_crop."
        )
    n_pixels_with_landsat = landsat_bands_df["pixel_id"].n_unique()
    landsat_bands_df = landsat_bands_df.filter(pl.col("pixel_id").is_in(target_pixel_ids))
    print(f"[SCYM] CDL mask: {len(target_pixel_ids)} pixel(s) classified as '{crop}' "
          f"out of {crop_mask_df.height} in the grid "
          f"({n_pixels_with_landsat} had Landsat data before masking)")
    if landsat_bands_df.height == 0:
        raise RuntimeError(
            f"No CDL-classified '{crop}' pixel has any usable Landsat "
            f"observation this season. Cannot produce a SCYM estimate."
        )

    print(f"[SCYM] Training regression on a {n_reps}-member management-scenario ensemble "
          f"(real weather + real soil; see build_training_table docstring) ...")
    training_df = build_training_table(
        weather_dta=weather_dta, soil_layers=site_soil, config_file=config_file,
        early_window=early_window, late_window=late_window, n_reps=n_reps, crop=crop,
    )
    model = fit_scym_regression(training_df)
    print(f"[SCYM] Training fit: R^2 = {model['r2']:.3f}  (n = {model['n_train']} replicates)")

    # Convert scene dates to day of season and calculate GCVI for each scene.
    landsat_pdf = landsat_bands_df.to_pandas()
    landsat_pdf["day"] = (pd.to_datetime(landsat_pdf["DATE"]) - pd.to_datetime(planting_date)).dt.days
    landsat_with_indices = compute_vegetation_indices_from_bands(pl.from_pandas(landsat_pdf))

    per_pixel = (
        landsat_with_indices.group_by("pixel_id")
        .agg([
            pl.col("i").first(),
            pl.col("j").first(),
            pl.col("GCVI").filter(pl.col("day").is_between(*early_window)).max().alias("GCVI_early"),
            pl.col("GCVI").filter(pl.col("day").is_between(*late_window)).max().alias("GCVI_late"),
        ])
        .drop_nulls(["GCVI_early", "GCVI_late"])
    )

    if per_pixel.height == 0:
        raise RuntimeError(
            "No pixel had a cloud-free Landsat observation in BOTH the "
            "early and late compositing windows -- cannot estimate SCYM "
            "yield for any pixel this season. Consider widening "
            "early_window/late_window or max_cloud_cover."
        )

    w_actual = compute_seasonal_weather_covariates(weather_dta)
    scym_yield = predict_scym_yield(
        model,
        gcvi_early=per_pixel["GCVI_early"].to_numpy(),
        gcvi_late=per_pixel["GCVI_late"].to_numpy(),
        rain_total=w_actual["rain_total"],
        radn_mean=w_actual["radn_mean"],
        tmax_mean=w_actual["tmax_mean"],
    )
    per_pixel = per_pixel.with_columns(pl.Series("SCYM_yield_t_ha", scym_yield))

    # Flag predictions that require extrapolation beyond the training range.
    lo_e, hi_e = model["gcvi_early_range"]
    lo_l, hi_l = model["gcvi_late_range"]
    in_range = (
        (pl.col("GCVI_early") >= lo_e) & (pl.col("GCVI_early") <= hi_e) &
        (pl.col("GCVI_late") >= lo_l) & (pl.col("GCVI_late") <= hi_l)
    )
    per_pixel = per_pixel.with_columns(in_range.alias("in_training_range"))
    n_extrap = int((~per_pixel["in_training_range"]).sum())
    if n_extrap:
        print(f"[SCYM] WARNING: {n_extrap}/{per_pixel.height} pixel(s) have GCVI outside "
              f"the training ensemble's range (early {lo_e:.2f}-{hi_e:.2f}, "
              f"late {lo_l:.2f}-{hi_l:.2f}) -- their SCYM_yield_t_ha is an "
              f"extrapolation and may be unreliable. See 'in_training_range' "
              f"column; consider a larger/more varied n_reps ensemble.")

    final_day = cm_result["day"].max()
    true_yield = (
        cm_result.filter(pl.col("day") == final_day)
        .with_columns((pl.col("Grain") * 0.01 / (1 - 0.155)).alias("true_yield_t_ha"))
        .select(["pixel_id", "true_yield_t_ha"])
    )

    scym_df = (
        per_pixel.join(true_yield, on="pixel_id", how="left")
        .join(target_pixels.select(["pixel_id", "cdl_class"]), on="pixel_id", how="left")
    )
    return scym_df, model
