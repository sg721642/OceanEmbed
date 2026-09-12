"""
03_compute_climatology.py
OceanEmbed (SIH PS 26066) — Shivansh's track, Step 4.

Computes a smooth seasonal climatology from the GLORYS temperature field using
a Gaussian temporal smoothing kernel (sigma = 15 days) fit strictly on the
training window (days 1..130), followed by a flat continuation of the
end-of-train smoothed baseline across the holdout window (days 131..180),
and derives:
    temperature_anomaly = temperature_3d - climatology

CRITICAL LEAKAGE & STABILITY RULES:
    1. LEAKAGE-SAFE: The climatology baseline is constructed ONLY on the training
       portion of the time series (days 0..129, Jan 1 to May 10, 2023). Zero holdout
       observations are used.
    2. STABLE BASELINE (NO ILL-CONDITIONED MATRIX INVERSION): Rather than fitting
       unconstrained 1st+2nd annual harmonics on a truncated 130-day window (which
       suffers from severe collinearity and polynomial-style extrapolation divergence),
       we use a Gaussian temporal kernel on the training series and a flat continuation
       for the holdout period.
    3. HONEST ANOMALIES: temperature_anomaly = temperature_3d - climatology.
       Anomalies are NOT clipped or truncated. A diagnostic envelope check is performed
       solely for reporting/warning.

Same HOLDOUT_DAYS value MUST be used later in Satyam's training script.
It is written to holdout_config.json here so it can't silently drift
between the two stages.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INPUT_CUBE_PATH = PROC_DIR / "oceanembed_cube.nc"
OUTPUT_CUBE_PATH = PROC_DIR / "oceanembed_cube_with_climatology.nc"

HOLDOUT_DAYS = 50  # chronological holdout length -- shared with Satyam's training script


SIGMA_DAYS = 15.0  # Gaussian smoothing kernel bandwidth in days


def compute_gaussian_weights(n_train: int, sigma: float = 15.0) -> np.ndarray:
    """
    Construct unnormalized 2D Gaussian weight matrix of shape (n_train, n_train):
        W[t, s] = exp(-0.5 * ((t - s) / sigma)**2)
    No ill-conditioned matrix inversion.
    """
    t_idx = np.arange(n_train, dtype=np.float32)
    diff = t_idx[:, None] - t_idx[None, :]  # shape (n_train, n_train)
    weights = np.exp(-0.5 * (diff / sigma) ** 2)
    return weights.astype(np.float32)


def main():
    print("=" * 70)
    print("Computing seasonal climatology + anomaly (leakage-safe)")
    print(f"Method: NaN-aware Gaussian temporal smoothing kernel (sigma={SIGMA_DAYS}d, train-only)")
    print("        Flat holdout continuation (no unconstrained harmonic extrapolation)")
    print(f"Input cube (read-only): {INPUT_CUBE_PATH}")
    print(f"Output cube:           {OUTPUT_CUBE_PATH}")
    print("=" * 70)

    cube = xr.open_dataset(INPUT_CUBE_PATH)
    n_time = cube.sizes["time"]
    assert n_time > HOLDOUT_DAYS + 30, (
        f"Only {n_time} days available; need enough beyond the "
        f"{HOLDOUT_DAYS}-day holdout to compute a stable seasonal baseline."
    )
    split_idx = n_time - HOLDOUT_DAYS

    times = pd.to_datetime(cube.time.values)

    print(f"Total days: {n_time} | Train: {split_idx} days "
          f"({times[0].date()} to {times[split_idx-1].date()}) | "
          f"Holdout: {HOLDOUT_DAYS} days "
          f"({times[split_idx].date()} to {times[-1].date()})")
    assert times[:split_idx].max() < times[split_idx:].min(), \
        "Chronological ordering violated -- check time coordinate sort order."

    temp = cube["temperature_3d"]  # dims: (time, depth, lat, lon)
    n_depth, n_lat, n_lon = temp.sizes["depth"], temp.sizes["lat"], temp.sizes["lon"]

    print("\nFiltering training window with NaN-aware Gaussian kernel...")
    print(f"  sigma = {SIGMA_DAYS} days across {split_idx} training days (no ill-conditioned matrix inversion)")
    
    # 1. Compute raw Gaussian kernel matrix for training days (shape: split_idx x split_idx)
    W_train = compute_gaussian_weights(split_idx, sigma=SIGMA_DAYS)

    # 2. Reshape training data to (split_idx, n_space) for vectorized evaluation
    temp_vals = temp.values  # shape (n_time, n_depth, n_lat, n_lon)
    temp_train_2d = temp_vals[:split_idx].reshape(split_idx, -1)  # shape: (T, S)

    # 3. NaN-aware normalized smoothing:
    #    numerator   = sum_s [ W(t, s) * y(s) ] only where y(s) is finite
    #    denominator = sum_s [ W(t, s) ]        only where y(s) is finite
    #    climatology = numerator / denominator
    # If a single observation is NaN, it does NOT poison other time steps.
    # If all observations are NaN (land/bathymetry), denominator == 0 -> remains NaN.
    valid_mask = np.isfinite(temp_train_2d).astype(np.float32)
    temp_zero_filled = np.nan_to_num(temp_train_2d, nan=0.0).astype(np.float32)

    numerator = W_train @ temp_zero_filled  # (split_idx, n_space)
    denominator = W_train @ valid_mask      # (split_idx, n_space)

    with np.errstate(divide="ignore", invalid="ignore"):
        clim_train_2d = np.where(denominator > 0.0, numerator / denominator, np.nan)

    clim_train = clim_train_2d.reshape(split_idx, n_depth, n_lat, n_lon).astype(np.float32)

    # 4. Holdout period: Flat continuation of end-of-train smoothed baseline
    # Prevents any unconstrained polynomial or harmonic extrapolation divergence
    print(f"Extrapolating across {HOLDOUT_DAYS} holdout days using flat end-of-train continuation...")
    clim_holdout_step = clim_train[-1]  # shape (n_depth, n_lat, n_lon)
    clim_holdout = np.repeat(clim_holdout_step[np.newaxis, ...], HOLDOUT_DAYS, axis=0)

    # 5. Assemble full 4D climatology
    climatology = np.concatenate([clim_train, clim_holdout], axis=0).astype(np.float32)

    climatology_da = xr.DataArray(
        climatology,
        dims=["time", "depth", "lat", "lon"],
        coords={"time": cube.time, "depth": cube.depth, "lat": cube.lat, "lon": cube.lon},
        name="climatology",
    )
    
    # 5. Temperature anomaly: strictly temperature_3d - climatology (unclipped)
    anomaly_da = (temp - climatology_da).rename("temperature_anomaly")

    cube["climatology"] = climatology_da
    cube["temperature_anomaly"] = anomaly_da

    # Save to OUTPUT_CUBE_PATH without modifying INPUT_CUBE_PATH
    for vname in cube.variables:
        cube[vname].encoding = {}
    encoding = {v: {"zlib": True, "complevel": 4} for v in cube.data_vars}
    cube.to_netcdf(OUTPUT_CUBE_PATH, mode="w", encoding=encoding)
    print(f"\nSaved climatology cube -> {OUTPUT_CUBE_PATH}")

    # 6. Detailed Diagnostics
    anom_train = anomaly_da.isel(time=slice(None, split_idx)).values
    anom_holdout = anomaly_da.isel(time=slice(split_idx, None)).values

    print("\n" + "=" * 70)
    print("CLIMATOLOGY + ANOMALY DIAGNOSTICS")
    print("=" * 70)
    print(f"TRAIN period (days 0..{split_idx-1}):")
    print(f"  mean = {np.nanmean(anom_train):6.2f} deg C  | std = {np.nanstd(anom_train):6.2f} deg C")
    print(f"  min  = {np.nanmin(anom_train):6.2f} deg C  | max = {np.nanmax(anom_train):6.2f} deg C")

    print(f"\nHOLDOUT period (days {split_idx}..{n_time-1}):")
    print(f"  mean = {np.nanmean(anom_holdout):6.2f} deg C  | std = {np.nanstd(anom_holdout):6.2f} deg C")
    print(f"  min  = {np.nanmin(anom_holdout):6.2f} deg C  | max = {np.nanmax(anom_holdout):6.2f} deg C")

    # Percentage of holdout ocean cells with large anomalies
    valid_holdout = np.isfinite(anom_holdout)
    n_valid = int(valid_holdout.sum())
    if n_valid > 0:
        pct_5 = float((np.abs(anom_holdout[valid_holdout]) > 5.0).sum()) / n_valid * 100.0
        pct_10 = float((np.abs(anom_holdout[valid_holdout]) > 10.0).sum()) / n_valid * 100.0
        pct_20 = float((np.abs(anom_holdout[valid_holdout]) > 20.0).sum()) / n_valid * 100.0
        print(f"\nHOLDOUT outlier analysis (total valid cells: {n_valid:,}):")
        print(f"  |anomaly| >  5 deg C: {pct_5:6.2f}%")
        print(f"  |anomaly| > 10 deg C: {pct_10:6.2f}%")
        print(f"  |anomaly| > 20 deg C: {pct_20:6.2f}%")

    # Min/Max anomaly coordinates in holdout
    min_val = float(np.nanmin(anom_holdout))
    min_loc = np.unravel_index(np.nanargmin(anom_holdout), anom_holdout.shape)
    min_t_idx = split_idx + min_loc[0]
    print(f"\nHOLDOUT Minimum anomaly:")
    print(f"  value: {min_val:6.2f} deg C at Date={str(cube.time.values[min_t_idx])[:10]}, "
          f"Depth={float(cube.depth.values[min_loc[1]])}m, "
          f"Lat={float(cube.lat.values[min_loc[2]])}N, Lon={float(cube.lon.values[min_loc[3]])}E")

    max_val = float(np.nanmax(anom_holdout))
    max_loc = np.unravel_index(np.nanargmax(anom_holdout), anom_holdout.shape)
    max_t_idx = split_idx + max_loc[0]
    print(f"HOLDOUT Maximum anomaly:")
    print(f"  value: {max_val:6.2f} deg C at Date={str(cube.time.values[max_t_idx])[:10]}, "
          f"Depth={float(cube.depth.values[max_loc[1]])}m, "
          f"Lat={float(cube.lat.values[max_loc[2]])}N, Lon={float(cube.lon.values[max_loc[3]])}E")

    # Diagnostic Sanity Check Envelope (Warning only -- does NOT clip the anomaly)
    if n_valid > 0 and pct_10 > 5.0:
        print("\nDIAGNOSTIC WARNING: More than 5% of holdout cells exceed +/-10 deg C anomaly.")
    else:
        print("\nDIAGNOSTIC CHECK PASSED: Climatology and anomalies are within physically plausible bounds.")

    config = {
        "method": "gaussian_temporal_smoothing_with_flat_holdout",
        "sigma_days": SIGMA_DAYS,
        "input_cube": str(INPUT_CUBE_PATH.name),
        "output_cube": str(OUTPUT_CUBE_PATH.name),
        "holdout_days": HOLDOUT_DAYS,
        "n_total_days": int(n_time),
        "train_start": str(times[0].date()),
        "train_end": str(times[split_idx - 1].date()),
        "holdout_start": str(times[split_idx].date()),
        "holdout_end": str(times[-1].date()),
        "glorys_temperature_var": "temperature_3d",
        "argo_temperature_var": "argo_temperature_3d",
        "anomaly_var": "temperature_anomaly",
        "climatology_var": "climatology",
    }
    config_path = PROC_DIR / "holdout_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved shared config -> {config_path}")
    print("Satyam's training script reads HOLDOUT_DAYS and variable names "
          "from this file -- do not let the two scripts define these "
          "independently.")


if __name__ == "__main__":
    main()
