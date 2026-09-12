"""
01_fetch_data.py
OceanEmbed (SIH PS 26066) — Shivansh's track, Step 2.

Downloads all official PS-mandated datasets over the exact official domain:
  Domain : 5N-30N, 45E-105E, 0.25 deg, daily
  Inputs : SST (OSTIA), SSS (SMAP/SMOS), SSH/SLA (DUACS),
           surface currents (OSCAR L4), surface winds (ASCAT/CCMP)
  Target : GLORYS Global Ocean Reanalysis temperature, 15 standard depths
  Independent validation : Gridded ARGO via INCOIS Live Access Server (LAS)

All datasets are saved as raw NetCDF into data/raw/. Nothing is regridded or
merged here -- that happens in 02_harmonize.py. This script's only job is to
get honest, real data onto disk, at the right box and the right window, with
no silent substitutions.

CREDENTIALS YOU NEED BEFORE RUNNING THIS:
  - Copernicus Marine Service (CMEMS) account: https://data.marine.copernicus.eu
    Used for: OSTIA SST, SMAP/SMOS SSS, DUACS SSH, GLORYS temperature.
    Install: pip install copernicusmarine
    Login once: `copernicusmarine login` (stores credentials locally)
  - NASA Earthdata account: https://urs.earthdata.nasa.gov
    Used for: OSCAR L4 currents, ASCAT/CCMP winds (PO.DAAC).
    Install: pip install podaacpy   (or use `earthaccess`, simpler in 2025+)
    pip install earthaccess
  - INCOIS LAS (Gridded ARGO): no login required, but the OPeNDAP/THREDDS
    endpoint path can change. If the URL below 404s, go to
    https://las.incois.gov.in and locate the current Gridded ARGO NetCDF
    Subset/OPeNDAP link for the same box/window and paste it into
    ARGO_LAS_URL below -- do NOT substitute raw scattered float data for
    this. The PS explicitly names Gridded ARGO via LAS as the independent
    validation source.

WHY A 180-DAY DEMO WINDOW (not the full multi-year record):
  - The PS's "Expected Solution" only requires a *working PoC* over the
    Bay of Bengal / Arabian Sea, not multi-year reanalysis-grade training.
  - A 180-day window (~6 months) is long enough to (a) fit a stable
    first+second-harmonic seasonal climatology, (b) hold out 45-60 days
    chronologically and still have ~120-135 days to train on, and
    (c) keep raw+processed data small enough to iterate fast on a laptop.
  - You can lengthen WINDOW_DAYS later once the pipeline is verified end
    to end -- nothing downstream assumes exactly 180 days.
"""

import os
from pathlib import Path
import xarray as xr

# ----------------------------------------------------------------------
# OFFICIAL DOMAIN -- do not change without re-verifying against the PS PDF
# ----------------------------------------------------------------------
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RESOLUTION_DEG = 0.25
EXPECTED_LAT_CELLS = int(round((LAT_MAX - LAT_MIN) / RESOLUTION_DEG))   # 100
EXPECTED_LON_CELLS = int(round((LON_MAX - LON_MIN) / RESOLUTION_DEG))  # 240

# ----------------------------------------------------------------------
# DEMO WINDOW -- pick a period with good multi-sensor coverage.
# 2023 is a safe default: OSTIA/DUACS/GLORYS/OSCAR/ASCAT all have full
# operational coverage; SMAP SSS coverage is also good from 2020 onward.
# ----------------------------------------------------------------------
START_DATE = "2023-01-01"
WINDOW_DAYS = 180
# (computed properly below with pandas to avoid the xarray offset quirk)
import pandas as pd
END_DATE = (pd.Timestamp(START_DATE) + pd.Timedelta(days=WINDOW_DAYS - 1)).strftime("%Y-%m-%d")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Domain check: expecting {EXPECTED_LAT_CELLS} lat x {EXPECTED_LON_CELLS} lon "
      f"= {EXPECTED_LAT_CELLS*EXPECTED_LON_CELLS:,} cells")
print(f"Demo window: {START_DATE} to {END_DATE} ({WINDOW_DAYS} days)")

# ======================================================================
# 1) OSTIA SST  (via Copernicus Marine)
# ======================================================================
def fetch_ostia_sst():
    """
    OSTIA SST, 0.05 deg native, daily.
    Dataset DOI: https://doi.org/10.48670/moi-00168
    CMEMS dataset id (2024/2025): METOFFICE-GLO-SST-L4-REP-OBS-SST
    (verify current id at https://data.marine.copernicus.eu -- CMEMS
    occasionally renames dataset ids on catalogue refreshes)
    """
    import copernicusmarine
    out_path = RAW_DIR / "ostia_sst.nc"
    copernicusmarine.subset(
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        variables=["analysed_sst"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{START_DATE}T00:00:00", end_datetime=f"{END_DATE}T00:00:00",
        output_filename=str(out_path),
    )
    print(f"[OK] OSTIA SST -> {out_path}")
    return out_path


# ======================================================================
# 2) SMAP/SMOS SSS  (via Copernicus Marine)
# ======================================================================
def fetch_sss():
    """
    SSS, 0.125 deg native, daily.
    Dataset DOI: https://doi.org/10.48670/moi-00051
    CMEMS dataset id: MULTIOBS_GLO_PHY_SSS_L4_MY_015_014 (verify current id)
    """
    import copernicusmarine
    out_path = RAW_DIR / "smap_smos_sss.nc"
    copernicusmarine.subset(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        variables=["sos"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{START_DATE}T00:00:00", end_datetime=f"{END_DATE}T00:00:00",
        output_filename=str(out_path),
    )
    print(f"[OK] SSS -> {out_path}")
    return out_path


# ======================================================================
# 3) DUACS SSH/SLA  (via Copernicus Marine)
# ======================================================================
def fetch_ssh():
    """
    SSH/SLA, 0.25 deg native, daily.
    Dataset DOI: https://doi.org/10.48670/moi-00145
    CMEMS dataset id: cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D
    """
    import copernicusmarine
    out_path = RAW_DIR / "duacs_ssh.nc"
    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["sla", "adt"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{START_DATE}T00:00:00", end_datetime=f"{END_DATE}T00:00:00",
        output_filename=str(out_path),
    )
    print(f"[OK] SSH/SLA -> {out_path}")
    return out_path


# ======================================================================
# 4) OSCAR L4 surface currents (via NASA Earthdata / PO.DAAC)
# ======================================================================
def fetch_currents():
    """
    OSCAR L4 Final, 0.25 deg, daily.
    Download one granule at a time and validate each NetCDF before
    combining, to avoid corrupted/incomplete parallel downloads.
    """
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")

    results = earthaccess.search_data(
        short_name="OSCAR_L4_OC_FINAL_V2.0",
        temporal=(START_DATE, END_DATE),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )

    if not results:
        raise RuntimeError("No OSCAR granules found")

    raw_dir = RAW_DIR / "oscar_currents_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    valid_files = []

    for i, granule in enumerate(results, start=1):

        granule_name = granule.data_links()[0].split("/")[-1]
        expected_path = raw_dir / granule_name

        # Already downloaded and valid -> skip it
        if expected_path.exists():
            try:
                with xr.open_dataset(expected_path) as test_ds:
                    test_ds.load()

                valid_files.append(str(expected_path))
                print(f"[SKIP] Already valid: {expected_path.name}")
                continue

            except Exception:
                print(f"[RE-DOWNLOAD] Corrupt file: {expected_path.name}")
                expected_path.unlink()

        print(f"Downloading OSCAR granule {i}/{len(results)}...")

        downloaded = earthaccess.download(
            [granule],
            str(raw_dir),
        )

        if not downloaded:
            print(f"[WARN] No file returned for granule {i}")
            continue

        for file_path in downloaded:
            file_path = str(file_path)

            try:
                with xr.open_dataset(file_path) as test_ds:
                    test_ds.load()

                valid_files.append(file_path)
                print(f"[VALID] {file_path}")

            except Exception as e:
                print(f"[BAD] {file_path}: {e}")

    if not valid_files:
        raise RuntimeError("No valid OSCAR NetCDF files were downloaded")

    print(f"Valid OSCAR files: {len(valid_files)}")

    ds = xr.open_mfdataset(valid_files, combine="by_coords")

    # Clip to required project domain
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.sel(
            latitude=slice(LAT_MIN, LAT_MAX),
            longitude=slice(LON_MIN, LON_MAX),
        )

    out_path = RAW_DIR / "oscar_currents.nc"
    ds.to_netcdf(out_path)

    print(f"[OK] OSCAR currents -> {out_path}")
    return out_path


# ======================================================================
# 5) ASCAT/CCMP surface winds (via NASA Earthdata / PO.DAAC)
# ======================================================================
def fetch_winds():
    """
    CCMP L4 winds, 0.25 deg, 6-hourly -> resample to daily mean.
    Download one granule at a time, validate each file, and resume
    safely if some files already exist.
    """
    import earthaccess

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata authentication failed")

    results = earthaccess.search_data(
        short_name="CCMP_WINDS_10M6HR_L4_V3.1",
        temporal=(START_DATE, END_DATE),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )

    if not results:
        raise RuntimeError("No CCMP wind granules found")

    raw_dir = RAW_DIR / "ccmp_winds_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    valid_files = []

    for i, granule in enumerate(results, start=1):

        granule_name = granule.data_links()[0].split("/")[-1]
        expected_path = raw_dir / granule_name

        # Already downloaded and valid -> skip
        if expected_path.exists():
            try:
                with xr.open_dataset(expected_path) as test_ds:
                    test_ds.load()

                valid_files.append(str(expected_path))
                print(f"[SKIP] Already valid: {expected_path.name}")
                continue

            except Exception:
                print(f"[RE-DOWNLOAD] Corrupt file: {expected_path.name}")
                expected_path.unlink()

        print(f"Downloading CCMP wind granule {i}/{len(results)}...")

        downloaded = earthaccess.download(
            [granule],
            str(raw_dir),
        )

        if not downloaded:
            print(f"[WARN] No file returned for granule {i}")
            continue

        for file_path in downloaded:
            file_path = str(file_path)

            try:
                with xr.open_dataset(file_path) as test_ds:
                    test_ds.load()

                valid_files.append(file_path)
                print(f"[VALID] {file_path}")

            except Exception as e:
                print(f"[BAD] {file_path}: {e}")

    if not valid_files:
        raise RuntimeError("No valid CCMP NetCDF files were downloaded")

    print(f"Valid CCMP files: {len(valid_files)}")

    ds = xr.open_mfdataset(valid_files, combine="by_coords")

    ds = ds.sel(
        latitude=slice(LAT_MIN, LAT_MAX),
        longitude=slice(LON_MIN, LON_MAX),
    )

    # CCMP is 6-hourly -> daily mean
    ds_daily = ds.resample(time="1D").mean()

    out_path = RAW_DIR / "ccmp_winds.nc"
    ds_daily.to_netcdf(out_path)

    print(f"[OK] Winds -> {out_path}")
    return out_path

# ======================================================================
# 6) GLORYS temperature, 15 standard depths (TARGET) (via Copernicus Marine)
# ======================================================================
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def fetch_glorys_temperature():
    """
    GLORYS12 Global Ocean Reanalysis, temperature (thetao), all 15 PS
    standard depths.
    Dataset DOI: https://doi.org/10.48670/moi-00021
    CMEMS dataset id: cmems_mod_glo_phy_my_0.083deg_P1D-m (daily mean)
    """
    import copernicusmarine
    out_path = RAW_DIR / "glorys_temperature.nc"
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        minimum_depth=0, maximum_depth=1000,
        start_datetime=f"{START_DATE}T00:00:00", end_datetime=f"{END_DATE}T00:00:00",
        output_filename=str(out_path),
    )
    # GLORYS native depth levels don't exactly equal the 15 PS standard
    # depths -- select nearest available levels here so harmonize.py can
    # interpolate cleanly onto the exact standard set.
    ds = xr.open_dataset(out_path)
    ds_std = ds.sel(depth=STANDARD_DEPTHS, method="nearest")
    ds_std.to_netcdf(RAW_DIR / "glorys_temperature_stddepths.nc")
    print(f"[OK] GLORYS temperature (15 std depths) -> "
          f"{RAW_DIR / 'glorys_temperature_stddepths.nc'}")
    return RAW_DIR / "glorys_temperature_stddepths.nc"


# ======================================================================
# 7) Gridded ARGO via INCOIS LAS (INDEPENDENT VALIDATION -- PS-named source)
# ======================================================================
# NOTE: LAS endpoint paths change more often than CMEMS/PO.DAAC. If this
# 404s, open https://las.incois.gov.in in a browser, find the Gridded ARGO
# dataset, use its OPeNDAP/Data-subset link for the same box+window, and
# replace ARGO_LAS_URL below. Do not substitute raw float-level data --
# the PS explicitly names the *gridded* product.
ARGO_LAS_URL = (
    "https://erddap.incois.gov.in/erddap/griddap/"
    "incois_argo_10d_VAM"
)

def fetch_gridded_argo():
    out_path = RAW_DIR / "gridded_argo_temperature.nc"

    # INCOIS ERDDAP: download only the required ARGO subset.
    # - time: 2023-01-10 to 2023-06-20 (available 10-day observations)
    # - depth: all 24 available levels
    # - latitude: 5.5 to 29.5 N
    # - longitude: 45.5 to 104.5 E
    #
    # -k is required here because Python on this Mac is currently
    # unable to validate the INCOIS server certificate.
    url = (
        "https://erddap.incois.gov.in/erddap/griddap/"
        "incois_argo_10d_VAM.nc?"
        "TEMP%5B684%3A700%5D"
        "%5B0%3A23%5D"
        "%5B35%3A59%5D"
        "%5B15%3A74%5D"
    )

    import subprocess

    print("[INFO] Downloading constrained INCOIS ARGO subset...")

    subprocess.run(
        [
            "curl",
            "-k",
            "-L",
            "-f",
            "-o",
            str(out_path),
            url,
        ],
        check=True,
    )

    # Validate the downloaded NetCDF file.
    with xr.open_dataset(out_path) as ds:
        print("[INFO] ARGO dataset downloaded:")
        print(ds)

        if "TEMP" not in ds:
            raise ValueError("ARGO file does not contain TEMP variable.")

    print(f"[OK] Gridded ARGO (independent validation) -> {out_path}")
    return out_path


if __name__ == "__main__":
    print("=" * 70)
    print("OceanEmbed data fetch -- official PS 26066 domain and datasets")
    print("=" * 70)

    fetched = {}
    for name, fn in [
        ("OSTIA SST", fetch_ostia_sst),
        ("SSS", fetch_sss),
        ("DUACS SSH", fetch_ssh),
        ("OSCAR currents", fetch_currents),
        ("CCMP winds", fetch_winds),
        ("GLORYS temperature", fetch_glorys_temperature),
        ("Gridded ARGO", fetch_gridded_argo),
    ]:
        try:
            fetched[name] = fn()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            print(f"       -> fix credentials/dataset id/endpoint above, "
                  f"then re-run. Do NOT proceed to harmonize.py until every "
                  f"one of these 7 says [OK].")

    print("\nSummary:")
    for name in ["OSTIA SST", "SSS", "DUACS SSH", "OSCAR currents",
                 "CCMP winds", "GLORYS temperature", "Gridded ARGO"]:
        status = "OK" if name in fetched else "MISSING"
        print(f"  [{status}] {name}")
