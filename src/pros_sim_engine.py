import numpy as np
from prosail import run_prosail
from src.prosail_model import build_soil_reflectance


def run_prosail_grid(df):
    """Run PROSAIL for every row in df. Returns reflectance (n, 2101)."""

    n = df.height
    refl = np.zeros((n, 2101))
    rs   = build_soil_reflectance(df)

    for k, row in enumerate(df.iter_rows(named=True)):
        try:
            r = run_prosail(
                n=1.5,
                cab=float(row["Cab"]),
                car=float(row["Car"]),
                cbrown=float(row["Cbrown"]),
                cw=float(row["Cw"]),
                cm=float(row["Cm"]),
                lai=float(row["lai_ps"]),
                lidfa=float(row["lidfa"]),
                hspot=float(row["hspot"]),
                tts=float(row["SZA"]),
                tto=float(row["VZA"]),
                psi=float(row["RAA"]),
                rsoil0=rs[k]
            )
            refl[k] = np.clip(r, 0.0, 1.0)
        except Exception:
            refl[k] = np.nan
    return refl


import numpy as np
import polars as pl


# Landsat 8/9 OLI band windows (nm)
LANDSAT_BANDS = {
    "blue": (452, 512),
    "green": (533, 590),
    "red": (636, 673),
    "nir": (851, 879),
    "swir16": (1566, 1651),
}


def extract_landsat_bands_and_indices(refl: np.ndarray) -> pl.DataFrame:
    """
    Resample PROSAIL full-spectrum output to Landsat 8/9 OLI band averages
    and compute NDVI, NDRE-proxy, and SAVI.

    This is the step that connects PROSAIL output to what real Landsat
    pixels would look like — enabling inversion of real Landsat SR against
    the synthetic LUT for Method 3 of the YIELDS pipeline.
    """

    wl = np.arange(400, 2501)
    eps = 1e-6

    out = {}
    for name, (lo, hi) in LANDSAT_BANDS.items():
        sel = (wl >= lo) & (wl <= hi)
        out[name] = pl.Series(name, refl[:, sel].mean(axis=1), dtype=pl.Float64)

    bdf = pl.DataFrame(out)

    bdf = bdf.with_columns(
        NDVI=(pl.col("nir") - pl.col("red")) / (pl.col("nir") + pl.col("red") + eps),
        SAVI=1.5 * (pl.col("nir") - pl.col("red")) / (pl.col("nir") + pl.col("red") + 0.5 + eps)
    )

    return bdf
