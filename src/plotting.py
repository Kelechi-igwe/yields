"""
Plotting functions for workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "figure.dpi": 150,
    }
)

STAGE_COLORS = {
    "emergence": "#B4C9A8",
    "vegetative": "#2D5A27",
    "reproductive": "#FFD700",
    "grain_fill": "#E2A13D",
    "maturity": "#8B5A2B",
}


# -----------------------------------------------------------------------------
# Shared plotting helpers
# -----------------------------------------------------------------------------

def _as_pandas(df):
    """Convert Polars or pandas objects into a pandas DataFrame."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df.copy()


def add_stage_shading(axs, df):
    """Shade time-series plots by crop stage."""
    df = _as_pandas(df)
    for stage in ["emergence", "vegetative", "reproductive", "grain_fill", "maturity"]:
        sub = df[df["stage"] == stage]
        if sub.empty:
            continue
        color = STAGE_COLORS.get(stage, "#999999")
        for ax in axs:
            ax.axvspan(sub["day"].min(), sub["day"].max(), color=color, alpha=0.08)


# -----------------------------------------------------------------------------
# Input and crop-model diagnostics
# -----------------------------------------------------------------------------

def plot_et_inputs(et_stack: np.ndarray, source_label: str = "OpenET", title: Optional[str] = None):
    """Plot ET time series and spatial snapshots."""
    if et_stack is None:
        raise ValueError("et_stack cannot be None")

    et_stack = np.asarray(et_stack, dtype=float)
    if et_stack.ndim != 3:
        raise ValueError("et_stack must have shape (days, rows, cols)")

    n_days = et_stack.shape[0]
    date_axis = pd.date_range("2025-05-15", periods=n_days, freq="D")
    rows, cols = et_stack.shape[1:]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    et_mean = np.nanmean(et_stack, axis=(1, 2))
    et_std = np.nanstd(et_stack, axis=(1, 2))

    axes[0].fill_between(date_axis, et_mean - et_std, et_mean + et_std, color="#4C72B0", alpha=0.2)
    axes[0].plot(date_axis, et_mean, color="#4C72B0", lw=1.8, label="Field mean")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("ET (mm/day)")
    axes[0].set_title("Daily ET — Field Average", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

    for ax, snap_day, label in [
        (axes[1], 30, "Day 30 (early)"),
        (axes[2], min(90, n_days - 1), "Day 90 (peak)"),
    ]:
        snap = et_stack[snap_day]
        vmax = max(float(np.nanmax(et_stack)) * 0.9, 0.1)
        im = ax.imshow(snap, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="ET (mm/day)", fraction=0.046)
        for r in range(snap.shape[0]):
            for c in range(snap.shape[1]):
                v = snap[r, c]
                ax.text(c, r, "NaN" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if (not np.isnan(v) and v > vmax * 0.6) else "black")
        ax.set_title(f"ET Spatial Map — {label}", fontsize=11)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

    fig.suptitle(f"Input: Evapotranspiration  |  {source_label}", fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_soil_inputs(soil_summary_grid: dict, source_label: str = "gSSURGO"):
    """Plot soil geometry and moisture metrics."""
    if soil_summary_grid is None:
        raise ValueError("soil_summary_grid cannot be None")

    if "theta_fc" not in soil_summary_grid or "theta_wp" not in soil_summary_grid:
        raise KeyError("soil_summary_grid must contain theta_fc and theta_wp")

    theta_fc = np.asarray(soil_summary_grid["theta_fc"], dtype=float)
    theta_wp = np.asarray(soil_summary_grid["theta_wp"], dtype=float)
    paw = theta_fc - theta_wp

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, cmap, label in [
        (axes[0], theta_fc, "BrBG", "Field Capacity θ_fc (m³/m³)"),
        (axes[1], theta_wp, "YlOrBr", "Wilting Point θ_wp (m³/m³)"),
        (axes[2], paw, "RdYlGn", "Plant-Available Water (m³/m³)"),
    ]:
        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
        im = ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label, fraction=0.046)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if val > (vmin + vmax) / 2 else "black")
        ax.set_title(label.split("(")[0].strip(), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Input: Soil Properties  |  {source_label}", fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_weather_inputs(weather_df, station_name: str = "Kansas Mesonet"):
    """Plot weather drivers: temperature, radiation, rainfall, GDD."""
    wdf = _as_pandas(weather_df).copy()
    if "DATE" in wdf.columns:
        wdf["DATE"] = pd.to_datetime(wdf["DATE"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    axes[0, 0].plot(wdf["DATE"], wdf["T_mean"], color="tomato", label="T_mean")
    axes[0, 0].plot(wdf["DATE"], wdf["TMAX"], color="firebrick", lw=0.8, alpha=0.6, label="Tmax")
    axes[0, 0].plot(wdf["DATE"], wdf["TMIN"], color="steelblue", lw=0.8, alpha=0.6, label="Tmin")
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].set_title("Air Temperature")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(wdf["DATE"], wdf["solar_rad"], color="goldenrod")
    axes[0, 1].set_ylabel("Solar radiation (MJ m⁻² day⁻¹)")
    axes[0, 1].set_title("Solar Radiation")

    axes[1, 0].bar(wdf["DATE"], wdf["rain"], color="steelblue", alpha=0.7, width=1)
    axes[1, 0].set_ylabel("Rainfall (mm day⁻¹)")
    axes[1, 0].set_title("Precipitation")

    gdd_daily = (wdf["T_mean"] - 8).clip(lower=0)
    gdd_cum = gdd_daily.cumsum()
    axes[1, 1].plot(wdf["DATE"], gdd_cum, color="darkgreen")
    axes[1, 1].axhline(800, ls="--", color="orange", alpha=0.7, label="Veg end (800 °Cd)")
    axes[1, 1].axhline(1600, ls="--", color="brown", alpha=0.7, label="Maturity (1600 °Cd)")
    axes[1, 1].set_ylabel("Cumulative GDD (°C·d, base 8°C)")
    axes[1, 1].set_title("Growing Degree Days")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Input: Weather  |  Kansas Mesonet — {station_name}", fontsize=12)
    plt.tight_layout()
    return fig, axes


def plot_crop_growth(cm_result, title: str = "Field-Average Crop Growth Dynamics"):
    """Plot LAI and biomass for the crop model result table."""
    df = _as_pandas(cm_result)
    if "day" not in df.columns:
        raise KeyError("cm_result must contain a 'day' column")

    lai = df.groupby("day")["LAI"].agg(["mean", "min", "max"])
    biomass = df.groupby("day")[["Biomass", "Leaf_Biomass", "Grain"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(lai.index, lai["mean"], color="green", label="LAI mean")
    axes[0].fill_between(lai.index, lai["min"], lai["max"], color="green", alpha=0.15, label="Range")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("LAI (m²/m²)")
    axes[0].set_title("Leaf Area Index")
    axes[0].legend(fontsize=8)

    axes[1].plot(biomass.index, biomass["Biomass"], label="Total biomass")
    axes[1].plot(biomass.index, biomass["Leaf_Biomass"], label="Leaf biomass")
    axes[1].plot(biomass.index, biomass["Grain"], label="Grain biomass")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Biomass (g/m²)")
    axes[1].set_title("Biomass Accumulation")
    axes[1].legend(fontsize=8)

    add_stage_shading(axes, df)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_water_balance(cm_result, title: str = "Water Balance and Stress"):
    """Plot moisture and water-stress signals."""
    df = _as_pandas(cm_result)
    sm = df.groupby("day")[["VWC", "f_water", "f_water_soil", "f_water_et", "Rainfall"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(sm.index, sm["VWC"], color="saddlebrown", label="VWC")
    axes[0].set_ylabel("VWC (m³/m³)")
    axes[0].set_title("Soil Moisture")
    ax0r = axes[0].twinx()
    ax0r.bar(sm.index, sm["Rainfall"], color="steelblue", alpha=0.3)
    ax0r.set_ylabel("Rainfall (mm/day)", color="steelblue")

    axes[1].plot(sm.index, sm["f_water"], color="red", label="f_water (combined)")
    axes[1].plot(sm.index, sm["f_water_soil"], color="orange", label="f_water_soil")
    axes[1].plot(sm.index, sm["f_water_et"], color="pink", label="f_water_ET")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Stress factor")
    axes[1].set_title("Water Stress (0=full stress, 1=no stress)")
    axes[1].legend(fontsize=8)
    ax1r = axes[1].twinx()
    ax1r.bar(sm.index, sm["Rainfall"], color="steelblue", alpha=0.3)
    ax1r.set_ylabel("Rainfall (mm/day)", color="steelblue")

    add_stage_shading(axes, df)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig, axes


def plot_prosail_summary(df_out, title: str = "Simulated Landsat VIs vs LAI"):
    """Plot the PROSAIL output summary (LAI + NDVI + SAVI)."""
    df = _as_pandas(df_out)
    spec = df.groupby("day")[["LAI", "NDVI", "SAVI"]].mean()

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(spec.index, spec["LAI"], color="black", lw=2, label="LAI")
    ax1.set_xlabel("Day of season")
    ax1.set_ylabel("LAI (m² m⁻²)")

    ax2 = ax1.twinx()
    ax2.plot(spec.index, spec["NDVI"], color="#1b9e77", label="NDVI (Landsat)")
    ax2.plot(spec.index, spec["SAVI"], color="#7570b3", label="SAVI (Landsat)")
    ax2.set_ylabel("Vegetation Index")
    ax2.set_ylim(0, 1)

    add_stage_shading([ax1], df)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper left", frameon=False, fontsize=9)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2, right=False)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_final_yield(df_final, grid_rows: int, grid_cols: int):
    """Plot end-of-season yield map."""
    df_final = _as_pandas(df_final).copy()
    if "yield_t_ha" not in df_final.columns:
        df_final["yield_t_ha"] = df_final["Grain"] * 0.01 / (1 - 0.155)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (var, cmap, label) in zip(axes, [
        ("yield_t_ha", "YlGn", "Simulated yield (t/ha)"),
        ("f_water", "RdYlGn", "Mean water stress factor"),
        ("Grain", "YlOrBr", "Grain biomass (g/m²)"),
    ]):
        grid = df_final.pivot(index="i", columns="j", values=var)
        vmin, vmax = float(grid.min().min()), float(grid.max().max())
        im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label, fraction=0.046)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = float(grid.iloc[r, c])
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if v > (vmin + vmax) / 2 else "black")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    yv = df_final["yield_t_ha"]
    fig.suptitle(
        f"Final Yield Map  |  {grid_rows}×{grid_cols} grid  |  "
        f"Mean {yv.mean():.2f} ± {yv.std():.2f} t/ha  "
        f"(CV {yv.std() / yv.mean() * 100:.1f}%)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig, axes



# -----------------------------------------------------------------------------
# SCYM diagnostics and output maps
# -----------------------------------------------------------------------------

def plot_scym_yield(scym_df, model: dict, title: str = "SCYM Yield Estimation (Lobell et al., 2015)"):
    """
    Two-panel diagnostic for the SCYM method (src/scym_model.py):
 
      left  -- training-ensemble regression fit: crop-model simulated yield
               vs. the regression's own predicted yield, with R² (analogous
               to the paper's Fig. 3 calibration diagnostic).
      right -- operational per-pixel SCYM estimate vs. the crop model's own
               "true" simulated yield for that pixel, with the correlation
               coefficient r (analogous to the paper's Fig. 4/9 validation
               scatterplots, here checked internally rather than against
               independent farmer-reported yields).
    """
    df = _as_pandas(scym_df)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    y_true, y_pred = model["y_true"], model["y_pred"]
    axes[0].scatter(y_pred, y_true, s=25, color="#2D5A27", alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_xlabel("Regression-predicted yield (t/ha)")
    axes[0].set_ylabel("Crop-model simulated yield (t/ha)")
    axes[0].set_title(f"Training fit  (R\u00b2 = {model['r2']:.2f}, n={model['n_train']})")
    axes[0].grid(alpha=0.3)
 
    x = df["SCYM_yield_t_ha"].to_numpy()
    y = df["true_yield_t_ha"].to_numpy()
    y_has_variance = len(y) > 1 and np.nanstd(y) > 1e-9
    r = float(np.corrcoef(x, y)[0, 1]) if y_has_variance else float("nan")
    axes[1].scatter(x, y, s=60, color="#E2A13D", edgecolor="black")
    lims2 = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[1].plot(lims2, lims2, "k--", lw=1)
    axes[1].set_xlabel("SCYM yield estimate (t/ha)")
    axes[1].set_ylabel("Crop model \"true\" yield (t/ha)")
    if y_has_variance:
        axes[1].set_title(f"Operational pixels  (r = {r:.2f}, n={len(df)})")
    else:
        axes[1].set_title(f"Operational pixels  (n={len(df)})\n"
                           f"crop-model yield has ~zero variance across pixels -- "
                           f"correlation undefined", fontsize=10, color="#B22222")
    axes[1].grid(alpha=0.3)
 
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig, axes



def plot_scym_yield_map(scym_df, grid_rows: int, grid_cols: int,
                         title: str = "SCYM Yield Map (Lobell et al., 2015)"):
    """
    Spatial map of the SCYM per-pixel yield estimate, the crop model's own
    "true" simulated yield, and their difference -- the actual end
    deliverable of a yield-*mapping* method (paper Fig. 7 style), as
    opposed to plot_scym_yield()'s scatter diagnostics.

    Pixels the CDL mask excluded (or that had no usable Landsat
    observation) are left blank rather than filled with a guessed value.
    Pixels whose GCVI fell outside the training ensemble's range (see
    run_scym_pipeline's 'in_training_range' column) are marked with a
    trailing '*' on the SCYM panel -- their number is an extrapolation,
    not a validated estimate.
    """
    df = _as_pandas(scym_df)

    full_grid = pd.DataFrame(
        [(i, j) for i in range(grid_rows) for j in range(grid_cols)],
        columns=["i", "j"],
    )
    df = full_grid.merge(df, on=["i", "j"], how="left")
    has_flag = "in_training_range" in df.columns

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    df["diff_t_ha"] = df["SCYM_yield_t_ha"] - df["true_yield_t_ha"]

    panels = [
        ("SCYM_yield_t_ha", "YlGn", "SCYM yield estimate (t/ha)"),
        ("true_yield_t_ha", "YlGn", "Crop model \"true\" yield (t/ha)"),
        ("diff_t_ha", "RdBu_r", "SCYM \u2212 true (t/ha)"),
    ]
    for ax, (var, cmap, label) in zip(axes, panels):
        grid = df.pivot(index="i", columns="j", values=var)
        finite = grid.to_numpy()
        finite = finite[~np.isnan(finite)]
        if finite.size == 0:
            ax.set_title(f"{label}\n(no data)", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        if var == "diff_t_ha":
            lim = float(np.nanmax(np.abs(finite))) or 1.0
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
        im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label, fraction=0.046)
        flag_grid = df.pivot(index="i", columns="j", values="in_training_range") if has_flag else None
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid.iloc[r, c]
                if pd.notna(v):
                    star = ""
                    if var == "SCYM_yield_t_ha" and flag_grid is not None:
                        flag_val = flag_grid.iloc[r, c]
                        if pd.notna(flag_val) and not bool(flag_val):
                            star = "*"
                    ax.text(c, r, f"{v:.2f}{star}", ha="center", va="center", fontsize=10,
                            color="white" if var != "diff_t_ha" and v > (vmin + vmax) / 2 else "black")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    n_mapped = df["SCYM_yield_t_ha"].notna().sum()
    subtitle = f"{n_mapped}/{grid_rows * grid_cols} pixels mapped (rest excluded by CDL mask or missing Landsat obs.)"
    if has_flag:
        n_extrap = int((df["in_training_range"] == False).sum())
        if n_extrap:
            subtitle += f"  |  * = {n_extrap} pixel(s) extrapolated beyond training GCVI range"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12)
    plt.tight_layout()
    return fig, axes


def save_figure(fig, output_dir: str | Path, file_name: str):
    """Write a figure to disk."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / file_name, bbox_inches="tight")


def save_scym_yield_geotiff(scym_df, grid_rows: int, grid_cols: int,
                            grid_crs, grid_transform, output_path: str | Path):
    """Write the SCYM estimates on the source Landsat pixel grid."""
    import rasterio

    df = _as_pandas(scym_df)
    values = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    for row in df.itertuples(index=False):
        value = getattr(row, "SCYM_yield_t_ha")
        if pd.notna(value):
            values[int(row.i), int(row.j)] = float(value)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=grid_rows,
        width=grid_cols,
        count=1,
        dtype="float32",
        crs=grid_crs,
        transform=grid_transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(values, 1)


def generate_pipeline_plots(weather_df, et_stack, soil_summary_grid, cm_result, df_out,
                             scym_df=None, scym_model=None,
                             grid_rows: int = None, grid_cols: int = None,
                             output_dir: str | Path = "plots"):
    """
    Create the standard diagnostic plots from the pipeline outputs.

    :param scym_df: optional per-pixel SCYM results table (output of
        src.scym_model.run_scym_pipeline). If provided along with
        `scym_model`, two additional SCYM figures are generated: the
        training-fit/validation scatter diagnostics (plot_scym_yield) and
        the spatial SCYM yield map (plot_scym_yield_map).
    :param scym_model: optional fitted-regression dict, the second return
        value of src.scym_model.run_scym_pipeline.
    :param grid_rows, grid_cols: grid dimensions, needed for the final-yield
        and SCYM yield maps. If omitted, inferred from cm_result's i/j
        columns (max index + 1) -- pass explicitly if the simulated grid
        has empty edge rows/columns that would make that inference wrong.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer dimensions only when they were not supplied by the caller.
    cm_pdf = _as_pandas(cm_result)
    if grid_rows is None:
        grid_rows = int(cm_pdf["i"].max()) + 1
    if grid_cols is None:
        grid_cols = int(cm_pdf["j"].max()) + 1

    # Create the standard input, crop-model, and PROSAIL figures.
    et_fig, _ = plot_et_inputs(et_stack, source_label="OpenET")
    soil_fig, _ = plot_soil_inputs(soil_summary_grid, source_label="gSSURGO")
    weather_fig, _ = plot_weather_inputs(weather_df, station_name="Kansas Mesonet")
    crop_fig, _ = plot_crop_growth(cm_result)
    water_fig, _ = plot_water_balance(cm_result)
    prosail_fig, _ = plot_prosail_summary(df_out)

    final_day = cm_pdf["day"].max()
    final_yield_fig, _ = plot_final_yield(cm_pdf[cm_pdf["day"] == final_day], grid_rows, grid_cols)

    # Save figures with stable names for reports and downstream review.
    save_figure(et_fig, output_dir, "01_et_inputs.png")
    save_figure(soil_fig, output_dir, "02_soil_inputs.png")
    save_figure(weather_fig, output_dir, "03_weather_inputs.png")
    save_figure(crop_fig, output_dir, "04_crop_growth.png")
    save_figure(water_fig, output_dir, "05_water_balance.png")
    save_figure(prosail_fig, output_dir, "06_prosail_summary.png")
    save_figure(final_yield_fig, output_dir, "07_final_yield_map.png")

    figs = {
        "et": et_fig,
        "soil": soil_fig,
        "weather": weather_fig,
        "crop": crop_fig,
        "water": water_fig,
        "prosail": prosail_fig,
        "final_yield": final_yield_fig,
    }

    if scym_df is not None and scym_model is not None:
        # Add SCYM diagnostics when both results and the fitted model exist.
        scym_fig, _ = plot_scym_yield(scym_df, scym_model)
        scym_map_fig, _ = plot_scym_yield_map(scym_df, grid_rows, grid_cols)
        save_figure(scym_fig, output_dir, "08_scym_yield.png")
        save_figure(scym_map_fig, output_dir, "09_scym_yield_map.png")
        figs["scym"] = scym_fig
        figs["scym_map"] = scym_map_fig

    return figs


if __name__ == "__main__":
    import os
    from datetime import timedelta

    import yaml
    from dotenv import load_dotenv
    import polars as pl

    from src.cm_sim_engine import run_grid_simulation
    from src.data_pull import (
        build_grid_centroids,
        create_pipeline_sesh,
        fetch_et_stack_from_openet,
        fetch_ssurgo_soil_for_bbox,
        fetch_weather_from_mesonet,
        generate_et_stack_synthetic,
        assign_mukeys_to_grid,
        build_ssurgo_soil_layers_grid,
    )
    from src.pros_sim_engine import extract_landsat_bands_and_indices, run_prosail_grid
    from src.prosail_model import add_solar_geometry, map_to_prosail_params

    load_dotenv()

    ROOT = Path(__file__).resolve().parent.parent
    config_path = ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    planting_date = "2025-05-15"
    season_length = 180
    start_date = pd.to_datetime(planting_date)
    end_date = start_date + timedelta(days=season_length - 1)
    bbox = {"min_lon": -98.55, "min_lat": 38.30, "max_lon": -98.50, "max_lat": 38.35}
    site_lat, site_lon = 38.32, -98.52
    grid_rows, grid_cols = 3, 3

    weather_df, weather_data = fetch_weather_from_mesonet(
        station="Manhattan",
        start=start_date,
        end=end_date,
        year_filter=2025,
        planting_date=planting_date,
        config_file=config,
    )

    grid_lats, grid_lons = build_grid_centroids(bbox=bbox, n_rows=grid_rows, n_cols=grid_cols)
    session = create_pipeline_sesh(os.getenv("OPENET_API_KEY"))
    try:
        et_stack, _ = fetch_et_stack_from_openet(
            grid_lats=grid_lats,
            grid_lons=grid_lons,
            start_date=start_date,
            end_date=end_date,
            api_key=os.getenv("OPENET_API_KEY"),
            season_length=season_length,
            session=session,
        )
        if et_stack is None or np.isnan(et_stack).all():
            raise ValueError("OpenET returned invalid data")
    except Exception:
        et_stack = generate_et_stack_synthetic(grid_rows, grid_cols, season_length)

    soil_data = fetch_ssurgo_soil_for_bbox(bbox=bbox, session=session, debug=False, config_file=config)
    # soil_layers_grid = None
    # if soil_data is not None and len(soil_data) > 0:
    #     mukey_grid = assign_mukeys_to_grid(grid_lats, grid_lons, soil_data)
    #     soil_layers_grid, soil_summary_grid = build_ssurgo_soil_layers_grid(soil_data, mukey_grid)
    # else:
    #     soil_summary_grid = {
    #         "theta_fc": np.full((grid_rows, grid_cols), 0.30),
    #         "theta_wp": np.full((grid_rows, grid_cols), 0.12),
    #     }

    if soil_data is None or len(soil_data) == 0:
        raise RuntimeError(
            "SSURGO/Soil Data Access returned no soil records for this demo "
            "bbox. This pipeline does not substitute placeholder/uniform "
            "soil -- check the bbox is within SSURGO survey coverage."
        )
    mukey_grid = assign_mukeys_to_grid(grid_lats, grid_lons, soil_data)
    soil_layers_grid, soil_summary_grid = build_ssurgo_soil_layers_grid(soil_data, mukey_grid)
    
    cm_result = run_grid_simulation(weather_data=weather_data, ET_stack=et_stack, soil_layers_grid=soil_layers_grid, config_file=config)

    df_ps = add_solar_geometry(cm_result, lat=site_lat, lon=site_lon)
    df_ps = map_to_prosail_params(df_ps)
    refl = run_prosail_grid(df_ps)
    bands = extract_landsat_bands_and_indices(refl)
    df_out = pl.concat([df_ps, bands], how="horizontal_extend")

    final_df = _as_pandas(cm_result)
    final_day = final_df["day"].max()
    final_df = final_df[final_df["day"] == final_day].copy()
    final_df["yield_t_ha"] = final_df["Grain"] * 0.01 / (1 - 0.155)

    generate_pipeline_plots(weather_df, et_stack, soil_summary_grid, final_df, _as_pandas(df_out), output_dir="plots")
    print("Plots saved to ./plots")
