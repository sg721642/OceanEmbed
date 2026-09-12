"""
05_verify_before_handoff.py
OceanEmbed (SIH PS 26066) — Shivansh's track, Step 6 (final gate).

Run this LAST. It is the literal checklist from the team's task file,
executable rather than eyeballed. It exits with a non-zero code and a
clear failure message if anything is wrong -- do not send oceanembed_cube.nc
to Satyam until this prints ALL CHECKS PASSED.
"""

import json
import sys
from pathlib import Path
import numpy as np
import xarray as xr

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CUBE_PATH = PROC_DIR / "oceanembed_cube_with_climatology.nc"
CONFIG_PATH = PROC_DIR / "holdout_config.json"

failures = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def main():
    print("=" * 70)
    print(f"FINAL VERIFICATION CHECKPOINT -- validating {CUBE_PATH.name}")
    print("=" * 70)

    if not CUBE_PATH.exists():
        print(f"FATAL: {CUBE_PATH} does not exist. Run scripts 01-04 first.")
        sys.exit(1)

    cube = xr.open_dataset(CUBE_PATH)
    print("Variables:", list(cube.data_vars))
    print("Sizes:", dict(cube.sizes))

    n_lat = cube.sizes.get("lat", 0)
    n_lon = cube.sizes.get("lon", 0)
    check("Grid is 100 x 240 (24,000 cells)", n_lat == 100 and n_lon == 240,
          f"got {n_lat} x {n_lon} = {n_lat*n_lon:,}")

    # ---- 1. GLORYS temperature_3d checks ----
    check("temperature_3d (GLORYS) exists", "temperature_3d" in cube.data_vars)
    if "temperature_3d" in cube.data_vars:
        check("temperature_3d has 15 depth levels",
              cube.sizes.get("depth", 0) == 15, f"got {cube.sizes.get('depth')}")
        expected_depths = {0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000}
        got_depths = set(int(round(d)) for d in cube.depth.values)
        check("Depth levels match PS standard set exactly",
              got_depths == expected_depths,
              f"missing={expected_depths - got_depths}, extra={got_depths - expected_depths}")

    # ---- 2. Gridded ARGO checks (sparse by design) ----
    check("argo_temperature_3d (Gridded ARGO) exists",
          "argo_temperature_3d" in cube.data_vars)
    if "argo_temperature_3d" in cube.data_vars:
        argo_vals = cube["argo_temperature_3d"].values
        n_finite_argo = int(np.isfinite(argo_vals).sum())
        check("Gridded ARGO is not empty/all-NaN (finite count > 0)",
              n_finite_argo > 0, f"finite observations = {n_finite_argo:,}")

        # Check observation dates in the 180-day time axis
        finite_dates = []
        for it in range(cube.sizes.get("time", 0)):
            if np.any(np.isfinite(cube["argo_temperature_3d"].isel(time=it).values)):
                finite_dates.append(str(cube.time.values[it])[:10])

        check("Gridded ARGO has exactly 17 native observation dates",
              len(finite_dates) == 17, f"found {len(finite_dates)} dates: {finite_dates}")
        check("Gridded ARGO retains 2023-06-20 observation date",
              "2023-06-20" in finite_dates, f"2023-06-20 present={ '2023-06-20' in finite_dates }")

    # ---- 3. Climatology and Anomaly checks ----
    check("climatology exists", "climatology" in cube.data_vars)
    check("temperature_anomaly exists", "temperature_anomaly" in cube.data_vars)
    if "temperature_anomaly" in cube.data_vars:
        anom = cube["temperature_anomaly"].values
        anom_finite = anom[np.isfinite(anom)]
        if len(anom_finite) > 0:
            anom_std = float(np.nanstd(anom_finite))
            anom_absmax = float(np.nanmax(np.abs(anom_finite)))
            check("temperature_anomaly has a sensible physical range",
                  anom_std < 8 and anom_absmax < 25,
                  f"std={anom_std:.2f}°C, max_abs={anom_absmax:.2f}°C")

    # ---- 4. Mask checks ----
    check("ocean_mask exists", "ocean_mask" in cube.data_vars)
    if "ocean_mask" in cube.data_vars:
        om_vals = cube["ocean_mask"].values
        unique_om = set(np.unique(om_vals))
        check("ocean_mask is binary (0/1 values only)",
              unique_om.issubset({0, 1}), f"unique values = {unique_om}")

    check("argo_valid_mask exists", "argo_valid_mask" in cube.data_vars)
    if "argo_valid_mask" in cube.data_vars:
        avm_vals = cube["argo_valid_mask"].values
        unique_avm = set(np.unique(avm_vals))
        check("argo_valid_mask is binary (0/1 values only)",
              unique_avm.issubset({0, 1}), f"unique values = {unique_avm}")

    # ---- 5. Input channel checks ----
    for v in ["sst", "sss", "ssh", "sla", "u_geo", "v_geo", "u_ageo", "v_ageo",
              "wind_u", "wind_v"]:
        check(f"Input channel '{v}' present", v in cube.data_vars)

    # ---- 6. Config checks ----
    check("holdout_config.json exists", CONFIG_PATH.exists())
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        check("Config records exact GLORYS variable name",
              cfg.get("glorys_temperature_var") == "temperature_3d")
        check("Config records exact ARGO variable name",
              cfg.get("argo_temperature_var") == "argo_temperature_3d")
        check("Config holdout_days is set and positive",
              isinstance(cfg.get("holdout_days"), int) and cfg["holdout_days"] > 0)

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED -- DO NOT SEND TO SATYAM:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL CHECKS PASSED -- safe to send {CUBE_PATH.name}, "
              "sanity_check.png, and holdout_config.json to Satyam.")
        print("=" * 70)


if __name__ == "__main__":
    main()
