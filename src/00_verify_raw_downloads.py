"""
00_verify_raw_downloads.py
OceanEmbed (SIH PS 26066) — run this BEFORE 02_harmonize.py.

Checks every file 01_fetch_data.py was supposed to produce in data/raw/:
  1. Exists, and there is exactly ONE canonical copy (flags duplicate
     download artifacts like "_(1).nc", "_(2).nc", " copy.nc" etc. --
     these are extremely common when a download is retried without
     overwriting, and if harmonize.py silently opens the wrong one you
     won't find out until much later).
  2. Opens cleanly with xarray (catches truncated/corrupted downloads --
     an incomplete NetCDF file usually still exists on disk with a
     plausible size, but fails to open or is missing its data section).
  3. Has the expected variable inside it.
  4. Covers the official domain (5-30N, 45-105E) -- catches a file that
     downloaded successfully but for the wrong bounding box.
  5. Covers the expected date range.
  6. Has values in a physically sane range (catches unit bugs -- e.g.
     Kelvin vs Celsius, or a variable that's actually all-NaN/all-zero).

Run this, fix anything it flags, and only then move to 02_harmonize.py.
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0

# ----------------------------------------------------------------------
# What 01_fetch_data.py is SUPPOSED to have produced, and how to check
# each one. Add/adjust "candidates" here if your filenames differ
# slightly from the script's defaults.
# ----------------------------------------------------------------------
EXPECTED = [
    dict(canonical="ostia_sst.nc", var="analysed_sst",
         value_range=(270, 320), value_note="Kelvin expected here (OSTIA native units) -- "
         "this file should be ~290-305K in this domain; conversion to Celsius happens in harmonize.py, not before"),
    dict(canonical="smap_smos_sss.nc", var="sos",
         value_range=(20, 40), value_note="practical salinity units (PSU), ~30-37 typical for this basin"),
    dict(canonical="duacs_ssh.nc", var=["adt", "sla"],
         value_range=(-2, 2), value_note="meters, small values expected"),
    dict(canonical="oscar_currents.nc", var=["u", "v"],
         value_range=(-3, 3), value_note="m/s, current speed"),
    dict(canonical="ccmp_winds.nc", var=["uwnd", "vwnd"],
         value_range=(-40, 40), value_note="m/s, wind speed"),
    dict(canonical="glorys_temperature_stddepths.nc", var="thetao",
         value_range=(-5, 35), value_note="Celsius expected, GLORYS is stored in Celsius natively",
         check_depth=True),
    dict(canonical="gridded_argo_temperature.nc", var=None,  # name may vary -- script will search
         value_range=(-5, 35), value_note="Celsius expected"),
]

# filename patterns that indicate a duplicate/retry artifact rather than
# a second genuinely different file
DUPLICATE_MARKERS = ["_(1)", "_(2)", "_(3)", " (1)", " (2)", " copy", "-copy", "_copy"]


def find_all_nc_files():
    return sorted(RAW_DIR.glob("*.nc"))


def group_by_likely_canonical(files):
    """Group files that look like duplicates of the same download
    (e.g. duacs_ssh.nc, duacs_ssh_(1).nc, duacs_ssh_(2).nc all map to
    the 'duacs_ssh' family)."""
    groups = defaultdict(list)
    for f in files:
        stem = f.stem
        base = stem
        for marker in DUPLICATE_MARKERS:
            if marker in base:
                base = base.split(marker)[0]
        groups[base].append(f)
    return groups


def check_duplicates(files):
    print("=" * 78)
    print("STEP 1 — Duplicate / retry-artifact check")
    print("=" * 78)
    groups = group_by_likely_canonical(files)
    problems = []
    for base, members in sorted(groups.items()):
        if len(members) > 1:
            print(f"\n  [DUPLICATE GROUP] '{base}' has {len(members)} files:")
            sizes = []
            for m in members:
                size_mb = m.stat().st_size / 1e6
                sizes.append((m, size_mb))
                print(f"      {m.name:45s} {size_mb:8.2f} MB")
            # flag if sizes differ meaningfully -- likely one is a failed/partial download
            mb_values = [s for _, s in sizes]
            if max(mb_values) - min(mb_values) > 0.5:
                print(f"      -> sizes differ noticeably. The smallest is likely a "
                      f"truncated/failed download. Keep the LARGEST, delete the rest, "
                      f"then rename it to exactly '{base}.nc'.")
            else:
                print(f"      -> sizes are near-identical, likely just repeat downloads "
                      f"of the same successful file. Keep ONE, delete the rest, rename "
                      f"to exactly '{base}.nc'.")
            problems.append(base)
    if not problems:
        print("  [OK] No duplicate-looking filenames found.")
    return problems


def resolve_canonical_path(canonical_name):
    """Return the path xarray should actually open for this expected file --
    prefers an exact-name match; otherwise falls back to the largest file
    in that duplicate group (best guess at the complete download) and
    warns loudly that a rename is still needed."""
    exact = RAW_DIR / canonical_name
    if exact.exists():
        return exact, False
    base = canonical_name.replace(".nc", "")
    candidates = [f for f in RAW_DIR.glob(f"{base}*.nc")]
    if not candidates:
        return None, False
    best = max(candidates, key=lambda f: f.stat().st_size)
    return best, True  # True = "found via fallback, needs rename"


def check_file_contents(entry):
    canonical = entry["canonical"]
    path, used_fallback = resolve_canonical_path(canonical)

    print(f"\n--- {canonical} ---")
    if path is None:
        print(f"  [MISSING] No file matching '{canonical}' (or a duplicate variant) "
              f"found in {RAW_DIR}. Re-run the relevant fetch function in "
              f"01_fetch_data.py for this dataset.")
        return False

    if used_fallback:
        print(f"  [WARN] Exact filename '{canonical}' not found -- using largest "
              f"match instead: {path.name}. Rename this file to '{canonical}' "
              f"before running harmonize.py, which expects the exact name.")

    size_mb = path.stat().st_size / 1e6
    if size_mb < 0.05:
        print(f"  [FAIL] File is only {size_mb*1000:.1f} KB -- almost certainly an "
              f"empty or failed download, not real data. Re-download this one.")
        return False

    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        print(f"  [FAIL] Could not open with xarray: {e}")
        print(f"         This usually means a truncated/corrupted download. Re-download.")
        return False

    ok = True

    # --- variable presence ---
    var_spec = entry["var"]
    var_candidates = [var_spec] if isinstance(var_spec, str) else (var_spec or [])
    found_var = None
    if var_candidates:
        for v in var_candidates:
            if v in ds.data_vars:
                found_var = v
                break
        if found_var is None:
            # search for a plausible name containing "temp" for the ARGO case
            guesses = [v for v in ds.data_vars if "temp" in v.lower()]
            if canonical.startswith("gridded_argo") and guesses:
                found_var = guesses[0]
                print(f"  [INFO] Expected variable name not fixed for Gridded ARGO -- "
                      f"detected candidate '{found_var}'. WRITE THIS EXACT NAME DOWN, "
                      f"harmonize.py and Satyam's metrics script both need it verbatim.")
            else:
                print(f"  [FAIL] None of expected variable(s) {var_candidates} found. "
                      f"Available: {list(ds.data_vars)}")
                ok = False
    else:
        guesses = [v for v in ds.data_vars if "temp" in v.lower()]
        found_var = guesses[0] if guesses else None
        if found_var:
            print(f"  [INFO] Detected likely temperature variable: '{found_var}' "
                  f"(available vars: {list(ds.data_vars)})")
        else:
            print(f"  [FAIL] No variable containing 'temp' found. Available: {list(ds.data_vars)}")
            ok = False

    if found_var:
        print(f"  [OK] Variable '{found_var}' present. Size: {size_mb:.1f} MB")

        # --- domain coverage ---
        lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
        lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
        if lat_name and lon_name:
            lat_vals = ds[lat_name].values
            lon_vals = ds[lon_name].values
            lat_span_ok = lat_vals.min() <= LAT_MIN + 1 and lat_vals.max() >= LAT_MAX - 1
            lon_span_ok = lon_vals.min() <= LON_MIN + 1 and lon_vals.max() >= LON_MAX - 1
            print(f"  Domain: lat [{lat_vals.min():.2f}, {lat_vals.max():.2f}], "
                  f"lon [{lon_vals.min():.2f}, {lon_vals.max():.2f}]  "
                  f"(expected roughly lat [{LAT_MIN},{LAT_MAX}], lon [{LON_MIN},{LON_MAX}])")
            if not (lat_span_ok and lon_span_ok):
                print(f"  [WARN] Domain coverage looks narrower than the official box. "
                      f"If this was fetched with a tight exact-box request this can be "
                      f"fine (edge cells), but double-check it isn't a wrong-region download.")
        else:
            print(f"  [WARN] Could not find lat/lon coordinate names to check domain "
                  f"coverage (coords found: {list(ds.coords)})")

        # --- time coverage ---
        if "time" in ds.coords:
            n_time = ds.sizes.get("time", 0)
            print(f"  Time steps: {n_time}  "
                  f"({str(ds.time.values.min())[:10]} to {str(ds.time.values.max())[:10]})")
            if n_time < 30:
                print(f"  [WARN] Fewer than 30 time steps -- confirm this matches your "
                      f"intended WINDOW_DAYS in 01_fetch_data.py, not a partial download.")
        else:
            print(f"  [WARN] No 'time' coordinate found.")

        # --- depth coverage (GLORYS / ARGO only) ---
        if entry.get("check_depth") and "depth" in ds.coords:
            depths = sorted(int(round(d)) for d in ds.depth.values)
            expected = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
            print(f"  Depth levels found: {depths}")
            missing = set(expected) - set(depths)
            if missing:
                print(f"  [WARN] Missing standard depths: {sorted(missing)} -- "
                      f"harmonize.py selects nearest available, but check these are close.")
            else:
                print(f"  [OK] All 15 PS standard depths present.")

        # --- value sanity range ---
        vals = ds[found_var].values
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size == 0:
            print(f"  [FAIL] Variable '{found_var}' is entirely NaN/missing -- "
                  f"download likely failed silently or the wrong subset was requested.")
            ok = False
        else:
            vmin, vmax, vmean = finite_vals.min(), finite_vals.max(), finite_vals.mean()
            lo, hi = entry["value_range"]
            print(f"  Value range: [{vmin:.2f}, {vmax:.2f}], mean {vmean:.2f}  "
                  f"({entry['value_note']})")
            if not (lo - 5 <= vmin and vmax <= hi + 5):
                print(f"  [WARN] Values fall outside the expected plausible range "
                      f"[{lo}, {hi}] by more than a small margin -- check units "
                      f"(e.g. Kelvin vs Celsius) before trusting this file.")
            else:
                print(f"  [OK] Values are in a physically plausible range.")

    ds.close()
    return ok


def main():
    print("#" * 78)
    print("# OceanEmbed raw-download verification -- run before 02_harmonize.py")
    print("#" * 78)

    if not RAW_DIR.exists():
        print(f"FATAL: {RAW_DIR} does not exist.")
        sys.exit(1)

    all_files = find_all_nc_files()
    print(f"\nFound {len(all_files)} .nc files in {RAW_DIR}:")
    for f in all_files:
        print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

    dup_groups = check_duplicates(all_files)

    print("\n" + "=" * 78)
    print("STEP 2 — Per-dataset content verification")
    print("=" * 78)
    results = {}
    for entry in EXPECTED:
        results[entry["canonical"]] = check_file_contents(entry)

    print("\n" + "#" * 78)
    print("SUMMARY")
    print("#" * 78)
    n_ok = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAIL/CHECK'}] {name}")
    print(f"\n{n_ok}/{len(results)} datasets passed content checks.")

    if dup_groups:
        print(f"\n[ACTION NEEDED] Resolve {len(dup_groups)} duplicate-filename group(s) "
              f"listed above before proceeding -- delete extras, keep one canonical "
              f"file per dataset, named exactly as 01_fetch_data.py expects.")

    if n_ok < len(results) or dup_groups:
        print("\n>>> DO NOT run 02_harmonize.py yet. Fix the items flagged above first.")
        sys.exit(1)
    else:
        print("\n>>> All checks passed. Safe to proceed to 02_harmonize.py.")


if __name__ == "__main__":
    main()
