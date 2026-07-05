from pvlib.solarposition import get_solarposition
import polars as pl
import numpy as np

def add_solar_geometry(df: pl.DataFrame, lat, lon):
    """Add solar zenith angle (SZA) per record using pvlib (13:00 local)."""

    # df = df.copy()
    # ts = pd.to_datetime(df["DATE"]) + pd.Timedelta(hours=13)

    ts = (
        df.select(pl.col('DATE').str.to_datetime().dt.offset_by('13h'))
        .to_series()
        .to_pandas()
    )
    solpos = get_solarposition(ts, lat, lon)
    # df["SZA"] = np.clip(solpos["zenith"].values, 0, 85)
    # df["VZA"] = 0.0
    # df["RAA"] = 0.0
    return df.with_columns(
        SZA = pl.Series(solpos['zenith'].values).clip(0, 85),
        VZA = pl.lit(0.0, dtype=pl.Float64),
        RAA = pl.lit(0.0, dtype=pl.Float64)
    )


def map_to_prosail_params(df: pl.DataFrame):
    """
    Map crop model outputs → PROSAIL leaf/canopy parameters.
    """
    # df = df.copy()
    df = df.with_columns(
        lai_ps = pl.col('LAI').clip(0.01, 8),
        max_leaf = pl.col('Leaf_Biomass').max().over('pixel_id')
    )

    # df["lai_ps"] = df["LAI"].clip(0.01, 8)
    # if "pixel_id" not in df.columns:
    #     df["pixel_id"] = df["i"].astype(str) + "_" + df["j"].astype(str)

    df = df.with_columns(
        frac=pl.when(pl.col('max_leaf') > 0)
        .then(pl.col('Leaf_Biomass') / pl.col('max_leaf'))
        .otherwise(0.0)
    )

    df = df.with_columns(
        Cab = (10 + 50 * pl.col('frac')).clip(5, 75),
        # Car = (0.15 * pl.col("Cab")).clip(0.5, 20),
        Cw = (0.003 + 0.015 * pl.col("f_water")).clip(0.001, 0.02),
        Cm = (0.003 + 0.012 * pl.col('frac')).clip(0.001, 0.02),
        hspot = (0.01 + 0.1 * (pl.col('lai_ps') / 8)).clip(0.01, 0.3)

    )

    df = df.with_columns(
        Car=(0.15 * pl.col("Cab")).clip(0.5, 20),
    )

    # df["Cab"]    = (10 + 50 * frac).clip(5, 75)
    # df["Car"]    = (0.15 * df["Cab"]).clip(0.5, 20)
    # df["Cw"]     = (0.003 + 0.015 * df["f_water"]).clip(0.001, 0.02)
    # df["Cm"]     = (0.003 + 0.012 * frac).clip(0.001, 0.02)

    df = df.with_columns(
        pl.when(pl.col('stage') == 'maturity').then(0.8)
        .when(pl.col('stage') == 'grain_fill').then(0.4)
        .otherwise(0.8)
        .alias('Cbrown')
    )

    df = df.with_columns(
        pl.when(pl.col('stage') == 'emergence').then(40)
        .when(pl.col('stage') == 'vegetative').then(55)
        .when(pl.col('stage') == 'reproductive').then(60)
        .otherwise(65)
        .alias('lidfa')
    )
    # df["Cbrown"] = np.select(
    #     [df["stage"]=="maturity", df["stage"]=="grain_fill"], [0.8, 0.4], default=0.0)
    # df["lidfa"]  = np.select(
    #     [df["stage"]=="emergence", df["stage"]=="vegetative", df["stage"]=="reproductive"],
    #     [40, 55, 60], default=65)

    # df["hspot"]  = (0.01 + 0.1 * (df["lai_ps"] / 8)).clip(0.01, 0.3)
    return df


def build_soil_reflectance(df):
    """
    Dynamic soil background spectrum per record.
    Brightness decreases with soil wetness (Liu et al. 2002).
    Spectral shape: linear ramp 400→2500 nm (Stoner & Baumgardner 1981).
    """
    sw = df.select('f_water_soil').to_numpy().flatten()
    bright = 0.1 + 0.3 * (1 - sw) ** 1.5
    wl     = np.linspace(400, 2500, 2101)
    base   = 0.1 + 0.4 * (wl - 400) / 2100
    return np.outer(bright, base)







