"""
04_sanity_check.py
OceanEmbed (SIH PS 26066) — Shivansh's track, Step 5.

Loads the finished cube and plots a 4-panel figure for one sample date:
SST, SSH, GLORYS temperature at 100m, and temperature_anomaly at 100m.
This is a visual confirmation that nothing is flipped (lat/lon axis
order), misaligned (land where ocean should be), or suspiciously blank
(all-zero / all-NaN regions that mean a merge silently failed).

Run this BEFORE sending anything to Satyam. If any panel looks wrong,
go back to 02_harmonize.py -- do not proceed with a cube you haven't
visually checked.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CUBE_PATH = PROC_DIR / "oceanembed_cube_with_climatology.nc"
OUT_PATH = PROC_DIR / "sanity_check.png"

SAMPLE_DEPTH_M = 100


def main():
    print(f"Reading cube for sanity check: {CUBE_PATH}")
    if not CUBE_PATH.exists():
        raise FileNotFoundError(f"Cube file not found: {CUBE_PATH}")

    cube = xr.open_dataset(CUBE_PATH)
    n_time = cube.sizes["time"]
    sample_idx = n_time // 2  # a date safely inside the training portion
    sample_date = str(cube.time.values[sample_idx])[:10]

    print("\n" + "=" * 70)
    print(f"OceanEmbed sanity check — validating {CUBE_PATH.name}")
    print("=" * 70)
    print(f"Dimensions: {dict(cube.sizes)}")
    print(f"Variables present: {list(cube.data_vars)}")

    # ---- Validation checks for required pipeline variables ----
    print("\nVariable presence and statistics:")
    for var in ["climatology", "temperature_anomaly", "ocean_mask", "argo_valid_mask"]:
        if var in cube:
            vals = cube[var].values
            finite_pct = float(np.isfinite(vals).mean()) * 100.0
            vmin = float(np.nanmin(vals)) if np.any(np.isfinite(vals)) else np.nan
            vmax = float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan
            print(f"  [PRESENT] {var:22s} shape={str(cube[var].shape):18s} "
                  f"finite={finite_pct:5.1f}%  range=[{vmin:.2f}, {vmax:.2f}]")
        else:
            print(f"  [ABSENT]  {var:22s} (not found in current cube)")

    # Assert required variables for the 4-panel visual check exist
    for req in ["sst", "ssh", "temperature_3d", "temperature_anomaly"]:
        if req not in cube:
            raise KeyError(f"Required variable '{req}' is missing from {CUBE_PATH.name}")

    sst = cube["sst"].isel(time=sample_idx)
    ssh = cube["ssh"].isel(time=sample_idx)
    depth_idx = int(np.argmin(np.abs(cube.depth.values - SAMPLE_DEPTH_M)))
    actual_depth = float(cube.depth.values[depth_idx])
    temp_100m = cube["temperature_3d"].isel(time=sample_idx, depth=depth_idx)
    anom_100m = cube["temperature_anomaly"].isel(time=sample_idx, depth=depth_idx)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"OceanEmbed sanity check — {sample_date} "
                 f"(North Indian Ocean, 5-30N 45-105E)", fontsize=13)

    for ax, data, title, cmap in [
        (axes[0, 0], sst, "SST (°C)", "inferno"),
        (axes[0, 1], ssh, "SSH / ADT (m)", "viridis"),
        (axes[1, 0], temp_100m, f"GLORYS temperature @ {actual_depth:g}m (°C)", "inferno"),
        (axes[1, 1], anom_100m, f"temperature_anomaly @ {actual_depth:g}m (°C)", "RdBu_r"),
    ]:
        im = data.plot(ax=ax, x="lon", y="lat", cmap=cmap, add_colorbar=True)
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved sanity check figure -> {OUT_PATH}")

    print("\nManual checklist before sending onward:")
    print("  [ ] Land is masked as NaN (not zero) in ocean fields")
    print("  [ ] SST is roughly 20-32°C in this domain, not Kelvin (~290-305)")
    print("  [ ] The 100m temperature field is noticeably cooler / smoother "
          "than SST, not identical to it")
    print("  [ ] temperature_anomaly panel is centered near 0 and mostly "
          "small (+/- a few degrees), not a copy of the raw temperature panel")
    print("  [ ] No obvious east-west or north-south flip versus a real map "
          "of the Bay of Bengal / Arabian Sea")


if __name__ == "__main__":
    main()
