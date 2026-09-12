"""
02_harmonize.py  --  OceanEmbed (PS 26066), clean rewrite

Common grid
-----------
  latitude  : 5.125 .. 29.875 N,  0.25 deg, 100 cells
  longitude : 45.125 .. 104.875 E, 0.25 deg, 240 cells
  time      : 2023-01-01 .. 2023-06-29, 180 daily steps
  depth     : 15 PS-standard levels

Outputs: data/processed/oceanembed_cube.nc

Variables: sst sss ssh sla u_geo v_geo u_ageo v_ageo wind_u wind_v
           temperature_3d (GLORYS 15 depths)
           argo_temperature_3d (Gridded ARGO 15 depths, sparse)

Fixes vs broken previous version
---------------------------------
1. DataArray has no .rename_vars() -- use .rename() instead.
2. SSS has singleton depth=[0.] -- squeeze before regrid.
3. OSCAR is global -- clip to domain BEFORE regrid (key NaN fix).
4. ARGO ZAX: regrid each level, then interp ZAX->STANDARD_DEPTHS.
5. GLORYS depth reassigned to exact STANDARD_DEPTHS after loading.
6. Ageostrophic reindexed to TARGET_TIME -> always (180,100,240).
7. Final merge: join=outer, compat=no_conflicts.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths and target axes
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR     = PROJECT_DIR / "data" / "raw"
PROC_DIR    = PROJECT_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RES = 0.25

TARGET_LAT  = np.round(np.arange(LAT_MIN + RES/2, LAT_MAX, RES), 10)
TARGET_LON  = np.round(np.arange(LON_MIN + RES/2, LON_MAX, RES), 10)
TARGET_TIME = pd.date_range("2023-01-01", "2023-06-29", freq="D")

assert len(TARGET_LAT)  == 100, f"Expected 100 lat cells, got {len(TARGET_LAT)}"
assert len(TARGET_LON)  == 240, f"Expected 240 lon cells, got {len(TARGET_LON)}"
assert len(TARGET_TIME) == 180, f"Expected 180 time steps, got {len(TARGET_TIME)}"

STANDARD_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=np.float64,
)

OMEGA          = 7.2921159e-5
G              = 9.80665
EARTH_RADIUS_M = 6_371_000.0

# ---------------------------------------------------------------------------
# Coordinate utility helpers
# ---------------------------------------------------------------------------

def _find_coord(da, candidates):
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    return None


def _drop_dup_1d(da, dim):
    vals = np.asarray(da[dim].values)
    _, idx = np.unique(vals, return_index=True)
    if len(idx) < len(vals):
        da = da.isel({dim: np.sort(idx)})
    return da


def standardize_spatial(da, label=""):
    """
    Rename lat/lon to canonical names, sort, deduplicate.

    Handles the OSCAR case where the coordinate variables are named 'lat'/'lon'
    but the underlying *dimensions* are named 'latitude'/'longitude'.
    xr.DataArray.rename({'old': 'new'}) renames only a coordinate, not its
    underlying dimension.  We must rename the dimension too, or interp() will
    fail because it operates on dimension names, not coordinate names.
    """
    lat_src = _find_coord(da, ("lat", "latitude"))
    lon_src = _find_coord(da, ("lon", "longitude"))
    if lat_src is None or lon_src is None:
        raise ValueError(
            f"{label}: cannot find lat/lon. "
            f"dims={tuple(da.dims)}, coords={list(da.coords)}"
        )

    # Find the underlying dimension that backs each coordinate.
    def _backing_dim(da, coord_name):
        if coord_name in da.dims:
            return coord_name
        if coord_name in da.coords:
            backing = da[coord_name].dims
            if len(backing) == 1:
                return backing[0]
        return None

    lat_dim = _backing_dim(da, lat_src)
    lon_dim = _backing_dim(da, lon_src)

    # Step 1: rename dimensions via swap_dims (avoids xarray UserWarning about indexing)
    dim_renames = {}
    if lat_dim is not None and lat_dim != "lat":
        dim_renames[lat_dim] = "lat"
    if lon_dim is not None and lon_dim != "lon":
        dim_renames[lon_dim] = "lon"
    if dim_renames:
        da = da.swap_dims(dim_renames)
        # After swap_dims, update what we think the coord is named
        if lat_src in dim_renames:
            lat_src = dim_renames[lat_src]
        if lon_src in dim_renames:
            lon_src = dim_renames[lon_src]

    # Step 2: rename coord variables if they still differ from canonical
    coord_renames = {}
    if lat_src != "lat" and lat_src in da.coords:
        coord_renames[lat_src] = "lat"
    if lon_src != "lon" and lon_src in da.coords:
        coord_renames[lon_src] = "lon"
    if coord_renames:
        da = da.rename(coord_renames)

    da = da.sortby("lat").sortby("lon")
    da = _drop_dup_1d(da, "lat")
    da = _drop_dup_1d(da, "lon")
    return da


def _cftime_to_datetime64(values):
    out = []
    for v in values:
        if isinstance(v, np.datetime64):
            out.append(np.datetime64(str(v)[:10], "ns"))
        elif hasattr(v, "year"):
            out.append(np.datetime64(
                f"{int(v.year):04d}-{int(v.month):02d}-{int(v.day):02d}", "ns"
            ))
        else:
            out.append(pd.Timestamp(v).to_datetime64().astype("datetime64[ns]"))
    return np.asarray(out, dtype="datetime64[ns]")


def normalize_time(da, label=""):
    """Re-index onto TARGET_TIME; missing days -> NaN; duplicates -> mean."""
    if "time" not in da.dims and "time" not in da.coords:
        return da
    times = _cftime_to_datetime64(da["time"].values)
    da    = da.assign_coords(time=("time", times))
    if len(np.unique(times)) < len(times):
        da = da.groupby("time").mean(keep_attrs=True)
    da = da.reindex(time=TARGET_TIME, fill_value=np.nan)
    da = da.assign_coords(time=("time", TARGET_TIME.values))
    return da


# ---------------------------------------------------------------------------
# Mask-aware spatial regridding
# ---------------------------------------------------------------------------

def regrid(da, label=""):
    """Bilinear interp with NaN-mask weighting (no land bleed)."""
    da = standardize_spatial(da, label)
    src_lat = np.asarray(da.lat.values, dtype=float)
    src_lon = np.asarray(da.lon.values, dtype=float)
    if src_lat.max() < TARGET_LAT.min() or src_lat.min() > TARGET_LAT.max():
        raise ValueError(f"{label}: source lat does not overlap target.")
    if src_lon.max() < TARGET_LON.min() or src_lon.min() > TARGET_LON.max():
        raise ValueError(f"{label}: source lon does not overlap target.")
    target    = {"lat": TARGET_LAT, "lon": TARGET_LON}
    valid     = xr.where(np.isfinite(da), 1.0, 0.0)
    filled    = da.fillna(0.0)
    numerator = filled.interp(target, method="linear")
    weight    = valid.interp(target, method="linear")
    out = (numerator / weight.where(weight > 0.0)).where(weight >= 0.5)
    out = out.assign_coords(lat=("lat", TARGET_LAT), lon=("lon", TARGET_LON))
    return out


# ---------------------------------------------------------------------------
# Surface field loader
# ---------------------------------------------------------------------------

def load_surface(filename, variable, output_name, kelvin_offset=0.0):
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    print(f"  {output_name}  <--  {filename}:{variable}")
    with xr.open_dataset(path) as ds:
        if variable not in ds.data_vars:
            raise KeyError(
                f"{filename}: variable {variable!r} not found. "
                f"Available: {list(ds.data_vars)}"
            )
        da = ds[variable].load()
    # Squeeze singleton depth/ZAX dims (e.g. SSS has depth=[0.])
    for dim in ("depth", "ZAX", "lev", "pressure"):
        if dim in da.dims and da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)
            print(f"    squeezed singleton dim {dim!r}")
    # Verify no depth dim remains in a surface field
    for dim in ("depth", "ZAX", "lev", "pressure"):
        if dim in da.dims:
            raise ValueError(
                f"{output_name}: unexpected depth dim {dim!r}. dims={da.dims}"
            )
    da = regrid(da, output_name)
    da = normalize_time(da, output_name)
    if kelvin_offset != 0.0:
        da = da + kelvin_offset
    da = da.transpose("time", "lat", "lon").astype(np.float32)
    da = da.assign_coords(
        time=("time", TARGET_TIME.values),
        lat=("lat",  TARGET_LAT),
        lon=("lon",  TARGET_LON),
    )
    da.name = output_name
    finite = float(np.isfinite(da.values).mean())
    print(f"    finite: {finite:.3%}  shape: {da.shape}")
    return da


# ---------------------------------------------------------------------------
# OSCAR loader (global -> regional clip BEFORE regrid)
# ---------------------------------------------------------------------------

def load_oscar():
    """
    OSCAR is global (lat -89.75..89.75, lon 0..359.75, cftime Julian).
    Clip to the project domain FIRST so xarray alignment cannot produce
    all-NaN ageostrophic output.
    """
    path = RAW_DIR / "oscar_currents.nc"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    print("  u_oscar / v_oscar  <--  oscar_currents.nc  (clip global -> regional)")
    halo = 0.5
    with xr.open_dataset(path) as ds:
        # OSCAR: coord names are 'lat'/'lon' (non-dim coords that index over
        # dims 'latitude'/'longitude'). xr.Dataset.sel() can slice on non-dim
        # coordinates by matching their float values, so use lat/lon here.
        lat_coord = _find_coord(ds, ("lat", "latitude"))
        lon_coord = _find_coord(ds, ("lon", "longitude"))
        if lat_coord is None or lon_coord is None:
            raise ValueError(f"Cannot find lat/lon in OSCAR. coords={list(ds.coords)}")
        ds_clip = ds.sel({
            lat_coord: slice(LAT_MIN - halo, LAT_MAX + halo),
            lon_coord: slice(LON_MIN - halo, LON_MAX + halo),
        })
        u_raw = ds_clip["u"].load()
        v_raw = ds_clip["v"].load()

    def _process(da, name):
        da = regrid(da, name)
        da = normalize_time(da, name)
        da = da.transpose("time", "lat", "lon").astype(np.float32)
        da = da.assign_coords(
            time=("time", TARGET_TIME.values),
            lat=("lat",  TARGET_LAT),
            lon=("lon",  TARGET_LON),
        )
        da.name = name
        return da

    u = _process(u_raw, "u_oscar")
    v = _process(v_raw, "v_oscar")
    print(f"    u_oscar finite: {float(np.isfinite(u.values).mean()):.3%}  "
          f"v_oscar finite: {float(np.isfinite(v.values).mean()):.3%}")
    return u, v


# ---------------------------------------------------------------------------
# Geostrophic currents from SSH
# ---------------------------------------------------------------------------

def compute_geostrophic(ssh):
    ssh = ssh.transpose("time", "lat", "lon")
    lat_rad  = np.deg2rad(TARGET_LAT)
    f_values = 2.0 * OMEGA * np.sin(lat_rad)
    f_values = np.where(np.abs(f_values) < 1e-8, np.nan, f_values)
    f_da   = xr.DataArray(f_values,        coords={"lat": TARGET_LAT}, dims=("lat",))
    cos_da = xr.DataArray(np.cos(lat_rad), coords={"lat": TARGET_LAT}, dims=("lat",))
    metres_per_deg = np.pi * EARTH_RADIUS_M / 180.0
    dssh_dlat = ssh.differentiate("lat")
    dssh_dlon = ssh.differentiate("lon")
    dssh_dy   = dssh_dlat / metres_per_deg
    dssh_dx   = dssh_dlon / (metres_per_deg * cos_da)
    u_geo = (-(G / f_da) * dssh_dy).astype(np.float32)
    v_geo = ( (G / f_da) * dssh_dx).astype(np.float32)
    # Guarantee canonical dim order and coordinate values
    u_geo = u_geo.transpose("time", "lat", "lon").assign_coords(
        time=("time", TARGET_TIME.values),
        lat=("lat",  TARGET_LAT),
        lon=("lon",  TARGET_LON),
    )
    v_geo = v_geo.transpose("time", "lat", "lon").assign_coords(
        time=("time", TARGET_TIME.values),
        lat=("lat",  TARGET_LAT),
        lon=("lon",  TARGET_LON),
    )
    u_geo.name = "u_geo"
    v_geo.name = "v_geo"
    return u_geo, v_geo


# ---------------------------------------------------------------------------
# Ageostrophic residual
# ---------------------------------------------------------------------------

def compute_ageostrophic(u_osc, v_osc, u_geo, v_geo):
    """Match by calendar-date keys; reindex result to TARGET_TIME."""
    print("  Computing ageostrophic residual (OSCAR - geostrophic)...")

    def date_keys(da):
        keys = []
        for t in da.time.values:
            if hasattr(t, "year"):
                keys.append((int(t.year), int(t.month), int(t.day)))
            else:
                s = str(np.datetime64(t, "D"))
                y, m, d = s.split("-")
                keys.append((int(y), int(m), int(d)))
        return keys

    osc_keys = date_keys(u_osc)
    geo_keys = date_keys(u_geo)
    osc_map  = {k: i for i, k in enumerate(osc_keys)}
    geo_map  = {k: i for i, k in enumerate(geo_keys)}
    common   = sorted(set(osc_map) & set(geo_map))
    print(f"    OSCAR: {len(osc_keys)} days  geo: {len(geo_keys)} days  common: {len(common)} days")
    if not common:
        msg = ("OSCAR and geostrophic current share no common calendar dates. "
               f"OSCAR first 3: {sorted(osc_map)[:3]}  "
               f"Geo first 3: {sorted(geo_map)[:3]}")
        raise RuntimeError(msg)
    canonical = np.array(
        [f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in common],
        dtype="datetime64[ns]",
    )
    u_osc_m = u_osc.isel(time=[osc_map[k] for k in common]).assign_coords(time=("time", canonical))
    v_osc_m = v_osc.isel(time=[osc_map[k] for k in common]).assign_coords(time=("time", canonical))
    u_geo_m = u_geo.isel(time=[geo_map[k] for k in common]).assign_coords(time=("time", canonical))
    v_geo_m = v_geo.isel(time=[geo_map[k] for k in common]).assign_coords(time=("time", canonical))
    u_ageo = (u_osc_m - u_geo_m).astype(np.float32)
    v_ageo = (v_osc_m - v_geo_m).astype(np.float32)
    u_ageo = u_ageo.reindex(time=TARGET_TIME, fill_value=np.nan).assign_coords(
        time=("time", TARGET_TIME.values))
    v_ageo = v_ageo.reindex(time=TARGET_TIME, fill_value=np.nan).assign_coords(
        time=("time", TARGET_TIME.values))
    u_ageo.name = "u_ageo"
    v_ageo.name = "v_ageo"
    return u_ageo, v_ageo


# ---------------------------------------------------------------------------
# GLORYS 3-D temperature
# ---------------------------------------------------------------------------

def build_glorys_temperature():
    path = RAW_DIR / "glorys_temperature_stddepths.nc"
    print(f"  temperature_3d  <--  {path.name}")
    with xr.open_dataset(path) as ds:
        if "thetao" not in ds.data_vars:
            raise KeyError(f"GLORYS: no thetao. Available: {list(ds.data_vars)}")
        n_depth = ds.sizes["depth"]
    if n_depth != len(STANDARD_DEPTHS):
        raise ValueError(f"GLORYS has {n_depth} depth levels; expected {len(STANDARD_DEPTHS)}.")
    layers = []
    for i, target_depth in enumerate(STANDARD_DEPTHS):
        with xr.open_dataset(path) as ds:
            layer = ds["thetao"].isel(depth=i).load()
        layer = regrid(layer, f"GLORYS {target_depth:g}m")
        layer = normalize_time(layer, f"GLORYS {target_depth:g}m")
        layer = layer.expand_dims(depth=[float(target_depth)])
        layers.append(layer.astype(np.float32))
    out = xr.concat(layers, dim="depth")
    out = out.transpose("time", "depth", "lat", "lon")
    out = out.assign_coords(
        time=("time",   TARGET_TIME.values),
        depth=("depth", STANDARD_DEPTHS),
        lat=("lat",     TARGET_LAT),
        lon=("lon",     TARGET_LON),
    )
    out.name = "temperature_3d"
    print(f"    shape: {out.shape}  finite: {float(np.isfinite(out.values).mean()):.3%}")
    return out.load()


# ---------------------------------------------------------------------------
# ARGO 3-D temperature (independent validation, genuinely sparse)
# ---------------------------------------------------------------------------

def build_argo_temperature():
    """
    1. Standardize spatial coord names.
    2. Regrid each ZAX level spatially to TARGET_LAT/LON.
    3. Normalize time onto TARGET_TIME.
    4. Interpolate ZAX -> STANDARD_DEPTHS (no extrapolation).
    """
    path = RAW_DIR / "gridded_argo_temperature.nc"
    print(f"  argo_temperature_3d  <--  {path.name}")
    with xr.open_dataset(path) as ds:
        if "TEMP" in ds.data_vars:
            da = ds["TEMP"].load()
        else:
            cands = [v for v in ds.data_vars if "temp" in v.lower()]
            if not cands:
                raise KeyError(f"No temperature variable in ARGO. vars={list(ds.data_vars)}")
            da = ds[cands[0]].load()
            print(f"    Using {cands[0]!r} as ARGO temperature variable")
        zax_values = np.asarray(ds["ZAX"].values, dtype=float)
    print(f"    Native ZAX ({len(zax_values)} levels): [{zax_values[0]:.0f}..{zax_values[-1]:.0f}] m")
    print(f"    Native finite: {float(np.isfinite(da.values).mean()):.3%}")

    # Step 1: standardize spatial coord names
    da = standardize_spatial(da, "ARGO")
    # da dims: (time, ZAX, lat, lon)

    # Step 2: regrid each ZAX level
    spatial_layers = []
    for iz, z in enumerate(zax_values):
        layer = da.isel(ZAX=iz)
        layer = regrid(layer, f"ARGO {z:.0f}m")
        layer = layer.expand_dims(ZAX=[z])
        spatial_layers.append(layer)
    da_regrid = xr.concat(spatial_layers, dim="ZAX")

    # Step 3: normalize time
    da_regrid = normalize_time(da_regrid, "ARGO")

    # Step 4: interpolate ZAX -> STANDARD_DEPTHS
    zax_min = float(zax_values.min())
    zax_max = float(zax_values.max())
    depth_layers = []
    for target_depth in STANDARD_DEPTHS:
        if target_depth < zax_min or target_depth > zax_max:
            ref   = da_regrid.isel(ZAX=0).drop_vars("ZAX", errors="ignore")
            layer = xr.full_like(ref, fill_value=np.nan).astype(np.float32)
        else:
            layer = da_regrid.interp(ZAX=float(target_depth), method="linear")
            layer = layer.drop_vars("ZAX", errors="ignore").astype(np.float32)
        layer = layer.expand_dims(depth=[float(target_depth)])
        depth_layers.append(layer)
    out = xr.concat(depth_layers, dim="depth")
    out = out.transpose("time", "depth", "lat", "lon")
    out = out.assign_coords(
        time=("time",   TARGET_TIME.values),
        depth=("depth", STANDARD_DEPTHS),
        lat=("lat",     TARGET_LAT),
        lon=("lon",     TARGET_LON),
    )
    out.name = "argo_temperature_3d"
    finite = float(np.isfinite(out.values).mean())
    print(f"    shape: {out.shape}  finite: {finite:.3%}  (35-60% NaN is expected)")
    return out.load()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_surface(da, name):
    assert da.dims == ("time", "lat", "lon"), f"{name}: dims={da.dims}"
    assert da.shape == (180, 100, 240),        f"{name}: shape={da.shape}"
    assert np.array_equal(da.time.values, TARGET_TIME.values), f"{name}: time mismatch"
    assert np.allclose(da.lat.values, TARGET_LAT, atol=1e-6),  f"{name}: lat mismatch"
    assert np.allclose(da.lon.values, TARGET_LON, atol=1e-6),  f"{name}: lon mismatch"


def validate_3d(da, name):
    assert da.dims == ("time", "depth", "lat", "lon"), f"{name}: dims={da.dims}"
    assert da.shape == (180, 15, 100, 240),             f"{name}: shape={da.shape}"
    assert np.array_equal(da.depth.values, STANDARD_DEPTHS), f"{name}: depth mismatch"
    assert np.allclose(da.lat.values, TARGET_LAT, atol=1e-6), f"{name}: lat mismatch"
    assert np.allclose(da.lon.values, TARGET_LON, atol=1e-6), f"{name}: lon mismatch"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 70
    print(sep)
    print("OceanEmbed -- harmonize all datasets onto 100x240 / 0.25 deg grid")
    print(sep)

    print("\n[1/4] Surface fields")
    sst    = load_surface("ostia_sst.nc",     "analysed_sst", "sst",    kelvin_offset=-273.15)
    sss    = load_surface("smap_smos_sss.nc", "sos",          "sss")
    ssh    = load_surface("duacs_ssh.nc",     "adt",          "ssh")
    sla    = load_surface("duacs_ssh.nc",     "sla",          "sla")
    wind_u = load_surface("ccmp_winds.nc",    "uwnd",         "wind_u")
    wind_v = load_surface("ccmp_winds.nc",    "vwnd",         "wind_v")

    print("\n[2/4] OSCAR currents + geostrophic / ageostrophic")
    u_oscar, v_oscar = load_oscar()
    print("  Computing geostrophic current from SSH gradient (g/f)...")
    u_geo, v_geo = compute_geostrophic(ssh)
    print(f"    u_geo finite: {float(np.isfinite(u_geo.values).mean()):.3%}  "
          f"v_geo finite: {float(np.isfinite(v_geo.values).mean()):.3%}")
    u_ageo, v_ageo = compute_ageostrophic(u_oscar, v_oscar, u_geo, v_geo)
    print(f"    u_ageo finite: {float(np.isfinite(u_ageo.values).mean()):.3%}  "
          f"v_ageo finite: {float(np.isfinite(v_ageo.values).mean()):.3%}")

    print("\n[3/4] 3-D temperature (GLORYS + Gridded ARGO)")
    temperature_3d      = build_glorys_temperature()
    argo_temperature_3d = build_argo_temperature()

    print("\n[4/4] Validating shapes and coordinates...")
    for name, da in [
        ("sst",    sst),    ("sss",    sss),    ("ssh",    ssh),    ("sla",    sla),
        ("u_geo",  u_geo),  ("v_geo",  v_geo),
        ("u_ageo", u_ageo), ("v_ageo", v_ageo),
        ("wind_u", wind_u), ("wind_v", wind_v),
    ]:
        validate_surface(da, name)
        print(f"    {name:12s}  OK  finite={float(np.isfinite(da.values).mean()):.3%}")
    for name, da in [
        ("temperature_3d",      temperature_3d),
        ("argo_temperature_3d", argo_temperature_3d),
    ]:
        validate_3d(da, name)
        print(f"    {name:22s}  OK  finite={float(np.isfinite(da.values).mean()):.3%}")

    print("\n[5/5] Building masks and short-gap filling...")
    # Derive canonical 2D ocean mask from GLORYS surface layer (depth=0)
    # 1 = Ocean, 0 = Land
    glorys_surface_valid = np.isfinite(temperature_3d.isel(depth=0)).any(dim="time")
    ocean_mask = xr.where(glorys_surface_valid, np.int8(1), np.int8(0))
    ocean_mask.name = "ocean_mask"
    ocean_mask = ocean_mask.assign_coords(lat=("lat", TARGET_LAT), lon=("lon", TARGET_LON))
    print(f"    ocean_mask created: {int((ocean_mask == 1).sum())} ocean cells ({float((ocean_mask == 1).mean()):.1%})")

    # Short gap-fill (<=3 days) on ocean cells for inputs only
    # Targets (temperature_3d, argo_temperature_3d) are NEVER gap filled
    input_vars = {
        "sst": sst, "sss": sss, "ssh": ssh, "sla": sla,
        "u_geo": u_geo, "v_geo": v_geo, "u_ageo": u_ageo, "v_ageo": v_ageo,
        "wind_u": wind_u, "wind_v": wind_v,
    }
    for vname, da in input_vars.items():
        # Fill short dropouts along time axis
        da_filled = da.interpolate_na(dim="time", method="linear", limit=3)
        # Ensure land cells remain NaN
        input_vars[vname] = da_filled.where(ocean_mask == 1)

    # Gridded ARGO validity mask: 1 = real measurement present, 0 = missing/land
    argo_valid_mask = xr.where(np.isfinite(argo_temperature_3d), np.int8(1), np.int8(0))
    argo_valid_mask = argo_valid_mask.where(ocean_mask == 1, np.int8(0))
    argo_valid_mask.name = "argo_valid_mask"
    argo_valid_mask = argo_valid_mask.assign_coords(
        time=("time", TARGET_TIME.values),
        depth=("depth", STANDARD_DEPTHS),
        lat=("lat", TARGET_LAT),
        lon=("lon", TARGET_LON),
    )

    print("\nMerging all variables into oceanembed_cube.nc...")

    # Strip any stray non-dimension alias coordinates
    _KEEP_COORDS = {"time", "lat", "lon", "depth"}

    def _clean(da):
        drop = [c for c in da.coords if c not in _KEEP_COORDS and c not in da.dims]
        if drop:
            da = da.drop_vars(drop)
        return da

    all_vars = [
        _clean(input_vars["sst"]), _clean(input_vars["sss"]),
        _clean(input_vars["ssh"]), _clean(input_vars["sla"]),
        _clean(input_vars["u_geo"]), _clean(input_vars["v_geo"]),
        _clean(input_vars["u_ageo"]), _clean(input_vars["v_ageo"]),
        _clean(input_vars["wind_u"]), _clean(input_vars["wind_v"]),
        _clean(temperature_3d),
        _clean(argo_temperature_3d),
        _clean(ocean_mask),
        _clean(argo_valid_mask),
    ]

    cube = xr.merge(
        all_vars,
        join="outer",
        compat="no_conflicts",
    )
    for vname in cube.variables:
        cube[vname].encoding = {}
    out_path = PROC_DIR / "oceanembed_cube.nc"
    encoding = {vname: {"zlib": True, "complevel": 4} for vname in cube.data_vars}
    cube.to_netcdf(out_path, encoding=encoding, format="NETCDF4")

    # Export variable_names.json for training and downstream pipelines
    names_path = PROC_DIR / "variable_names.json"
    with open(names_path, "w") as f:
        json.dump(
            {
                "input_vars": list(input_vars.keys()),
                "target_vars": ["temperature_3d", "argo_temperature_3d"],
                "glorys_temperature_var": "temperature_3d",
                "argo_temperature_var": "argo_temperature_3d",
                "ocean_mask_var": "ocean_mask",
                "argo_valid_mask_var": "argo_valid_mask",
                "grid": {"n_lat": len(TARGET_LAT), "n_lon": len(TARGET_LON)},
                "depths": STANDARD_DEPTHS.tolist(),
                "time_days": len(TARGET_TIME),
            },
            f,
            indent=2,
        )
    print(f"Saved metadata: {names_path}")

    print("\n" + sep)
    print("HARMONIZE COMPLETE")
    print(sep)
    print("Variables:", list(cube.data_vars))
    print("Sizes    :", dict(cube.sizes))
    print()
    for vname in cube.data_vars:
        nan_frac = float(cube[vname].isnull().mean().values)
        note = "  <- ARGO sparsity is expected" if "argo" in vname and "mask" not in vname else ""
        print(f"  {vname:25s}  NaN={nan_frac:.3%}  finite={1-nan_frac:.3%}{note}")
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("\nVariable name reference:")
    print("  GLORYS temperature     : temperature_3d")
    print("  ARGO  temperature      : argo_temperature_3d")
    print("  Geostrophic            : u_geo, v_geo")
    print("  Ageostrophic residual  : u_ageo, v_ageo")
    print("  Ocean Mask             : ocean_mask (1=ocean, 0=land)")
    print("  ARGO Validity Mask     : argo_valid_mask (1=valid observation)")


if __name__ == "__main__":
    main()
