# OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

**SIH Problem Statement:** PS 26066  
**Status:** `READY_WITH_TRANSPARENT_CAVEATS`  
**Model Freeze Date:** 2026-09-12  
**Python Environment:** `/opt/anaconda3/envs/oceanembed/bin/python` (Python 3.12, TF 2.16.2, Keras 3.15.1)

---

## Overview

OceanEmbed is a deep learning system that reconstructs the full vertical ocean temperature structure at 15 standard depth levels (0–1000 m) across the North Indian Ocean from daily satellite surface observations.

**OceanEmbed is trained to reconstruct GLORYS-derived subsurface temperature anomalies. It is evaluated on unseen held-out GLORYS data and independently validated against Gridded ARGO observations.**

The model learns to predict the deviation of daily ocean temperature from a Gaussian-smoothed seasonal climatological background, then adds that predicted anomaly back to produce absolute temperatures:

```
T_pred(x, y, z, t) = T_climatology(x, y, z, doy) + ΔT_model(x, y, z, t)
```

---

## Domain

| Parameter | Value |
|---|---|
| Region | North Indian Ocean |
| Latitude | 5°N – 30°N |
| Longitude | 45°E – 105°E |
| Spatial resolution | 0.25° |
| Temporal resolution | Daily |
| Grid dimensions | 100 (lat) × 240 (lon) = 24,000 cells |
| Active ocean cells | 22,333 (land/bathymetry masked) |

---

## Canonical Depth Levels

**15 standard depths (meters):**
```
[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
```

---

## Input Variables

10 daily satellite surface fields, standardized to 0.25° resolution:

| Variable | Description | Source Product |
|---|---|---|
| `sst` | Sea Surface Temperature | OSTIA (CMEMS) |
| `sss` | Sea Surface Salinity | SMAP/SMOS (CMEMS) |
| `ssh` | Sea Surface Height | DUACS altimetry (CMEMS) |
| `sla` | Sea Level Anomaly | DUACS altimetry (CMEMS) |
| `u_geo` | Geostrophic current U | Derived from SSH |
| `v_geo` | Geostrophic current V | Derived from SSH |
| `u_ageo` | Ageostrophic current U | OSCAR L4 (PO.DAAC) |
| `v_ageo` | Ageostrophic current V | OSCAR L4 (PO.DAAC) |
| `wind_u` | 10m wind U | CCMP L4 (PO.DAAC) |
| `wind_v` | 10m wind V | CCMP L4 (PO.DAAC) |

**Temporal window:** 7 consecutive days (t, t−1, t−2, t−3, t−4, t−5, t−6)  
**Total model input channels:** 10 variables × 7 days = **70 channels** (plus regime gate)

---

## Model Architecture

**Architecture:** OceanEmbed-v3-Strengthened (CNN with regime gate)

| Component | Shape |
|---|---|
| Input 0 — `surface_features` | `(None, 100, 240, 70)` |
| Input 1 — `regime_gate` | `(None, 100, 240, 1)` (0=Arabian Sea, 1=Bay of Bengal) |
| Output 0 — temperature anomaly | `(None, 100, 240, 15)` |
| Output 1 — predicted uncertainty σ | `(None, 100, 240, 15)` |
| Total parameters | **187,842** |
| Dtype | float32 |

---

## Training Target and Data Split

| Period | Role |
|---|---|
| 2023-01-01 → 2023-05-10 (130 days) | Training + validation |
| **2023-05-11 → 2023-06-29 (50 days)** | **Frozen holdout** |

**Training target:** GLORYS12V1 temperature anomaly at 15 depth levels.  
**GLORYS12V1** is a numerical ocean reanalysis model that assimilates satellite observations and in-situ profiles — it is the training target and held-out reference benchmark, **not physical ground truth**.

**Validation split within training:** Chronological (no shuffling). The model checkpoint was selected at epoch 5 (val_loss = 0.2528) before the holdout window. Zero data leakage from holdout to training.

---

## Final Headline Metrics (50-Day Holdout)

### GLORYS Reanalysis Benchmark (16,064,650 valid 4D pairs)

| Metric | Value |
|---|---|
| **Pooled RMSE** | **1.4913°C** |
| Pooled MAE | 1.0519°C |
| Mean Bias | +0.7673°C (warm bias) |
| Pooled 4D Pearson r | 0.9869 ⚠️ |

### Independent ARGO Observations (25,904 valid pairs)

| Metric | Value |
|---|---|
| **Pooled RMSE** | **1.8488°C** |
| Pooled MAE | 1.5654°C |
| Mean Bias | +1.5333°C |
| Pooled 4D Pearson r | 0.9876 ⚠️ |

> ⚠️ **IMPORTANT: Pooled 4D Pearson r is NOT a spatial-skill metric.** The high r (~0.987) is dominated by the vertical temperature gradient (~28°C at surface vs ~2°C at 1000 m depth). Within-depth spatial Pearson r for GLORYS has mean 0.459, median 0.614. Always label pooled 4D r as a secondary metric only.

### Per-Stratum GLORYS RMSE

| Depth Stratum | RMSE |
|---|---|
| 0–100 m | 1.6055°C |
| 125–300 m | 1.6831°C |
| 500–1000 m | 0.5762°C |

> Note: Pooled RMSE = `sqrt(mean((pred−obs)²))` over all jointly finite 4D pairs simultaneously — **NOT** the arithmetic mean of per-depth RMSEs (which equals ~1.40°C, a different number).

---

## Independent ARGO Validation — Important Limitations

- **Coverage:** 4 discrete 10-day cycle dates only: May 15, May 25, June 4, June 14, 2023
- **Spatial scope:** Arabian Sea only — **zero ARGO floats in Bay of Bengal during holdout**
- **0m depth:** Zero observations (ARGO CTD sensors shut off near surface to prevent biofouling)
- **N valid pairs:** 25,904 independent in-situ observations
- Correct nomenclature: "independent in-situ ARGO float observations" — never "ground truth"
- The Gridded ARGO product referenced in the PS is from INCOIS LAS

---

## Uncertainty Calibration — NOT DEMONSTRATED

The model outputs a predicted uncertainty σ(x,y,z,t) at each grid point and depth. This must be treated as **uncalibrated model-inferred variance**, not as Bayesian confidence bounds:

| Check | Empirical | Gaussian Nominal |
|---|---|---|
| r(σ, \|error\|) | ~0.20 | — |
| 1σ coverage | ~12.9% | 68.3% |
| 2σ coverage | ~24.9% | 95.5% |

---

## Thermal Inversion Analysis

A thermal inversion in reconstructed profiles is defined as: an adjacent-depth temperature increase ΔT ≥ +0.05°C within the upper 100 m (depth levels 0–100 m). Bin notation uses strict `[low, high)` intervals.

| Region | Qualifying Profiles | Frequency |
|---|---|---|
| Arabian Sea | 12,465 | 2.24% of profiles |
| Bay of Bengal | 4,453 | 0.80% of profiles |
| **Total** | **16,918** | — |

Total inversion segments: **19,233** (verified sum).

---

## Reproducibility

### Frozen Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `exports/oceanembed_model.keras` | `39eb06cd6f73e8866f1914fd6ce560079eb3bc3a271829c330aa923cdcc91bb5` |
| `data/processed/oceanembed_cube_with_climatology.nc` | `c56ae615750acba2c7fd9e36955716d500d61d29482001bd9daea494aac92b79` |
| `data/processed/holdout_config.json` | `75f82449d2f2a7b45222e8b3d6dfbef03171e114cc49406e0cb8d970ef4e7ef8` |
| `notebooks/OceanEmbed_Training_Pipeline.ipynb` | `62236fce5e4e601b00801d4e9891cf0cafb8bc668a254d280b2ab289aed31eaf` |

See `exports/final_freeze_audit.json` for the complete freeze record.

### Python Environment

> ⚠️ Use the `oceanembed` conda environment (Python 3.12). The base Anaconda Python 3.13 causes a segmentation fault during Keras backend initialization.

```bash
conda activate oceanembed
# or
/opt/anaconda3/envs/oceanembed/bin/python
```

**Pinned package versions** (see `exports/requirements-final.txt`):
```
tensorflow==2.16.2
tensorflow-metal==1.2.0
keras==3.15.1
numpy==1.26.4
pandas==3.0.5
xarray==2026.7.0
scipy==1.17.1
h5py==3.16.0
netCDF4==1.7.4
pyarrow==25.0.1
```

### Reproducing Frozen Inference

```python
import os, numpy as np, xarray as xr, keras
os.environ["KERAS_BACKEND"] = "tensorflow"

PROJECT_ROOT     = "/path/to/oceanembed_project"
CANONICAL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
INPUT_CHANNELS   = ["sst","sss","ssh","sla","u_geo","v_geo","u_ageo","v_ageo","wind_u","wind_v"]
TEMPORAL_WINDOW  = 7
HOLDOUT_DAYS     = 50
REGIME_SPLIT_LON = 77.0

cube  = xr.open_dataset(f"{PROJECT_ROOT}/data/processed/oceanembed_cube_with_climatology.nc")
times, lats, lons = cube.time.values, cube.lat.values, cube.lon.values
split_idx        = len(times) - HOLDOUT_DAYS
time_offset      = TEMPORAL_WINDOW - 1
split_idx_win    = split_idx - time_offset

X_raw  = np.stack([cube[ch].values for ch in INPUT_CHANNELS], axis=-1).astype(np.float32)
X_flat = X_raw[:split_idx].reshape(-1, X_raw.shape[-1])
ch_mean, ch_std = np.nanmean(X_flat, axis=0), np.nanstd(X_flat, axis=0)
ch_std[ch_std < 1e-6] = 1.0
X_norm = np.nan_to_num((X_raw - ch_mean) / ch_std, nan=0.0)

def add_window(X, w):
    T, H, W, C = X.shape
    out = np.zeros((T - w + 1, H, W, C * w), dtype=np.float32)
    for t in range(w - 1, T):
        out[t - w + 1] = np.concatenate([X[t - k] for k in range(w)], axis=-1)
    return out

X_win    = add_window(X_norm, TEMPORAL_WINDOW)
X_hold   = X_win[split_idx_win:]                        # (50, 100, 240, 70)
rg       = np.where(lons[None,:] < REGIME_SPLIT_LON, 0., 1.).astype(np.float32)
rg       = np.repeat(rg, len(lats), axis=0)
rg_batch = np.repeat(rg[None,:,:,None], HOLDOUT_DAYS, axis=0)

model       = keras.models.load_model(f"{PROJECT_ROOT}/exports/oceanembed_model.keras", compile=False)
anom, unc   = model.predict([X_hold, rg_batch], batch_size=4, verbose=0)

clim     = cube["climatology"].transpose("time","lat","lon","depth").values.astype(np.float32)
clim_h   = clim[time_offset:][split_idx_win:]
temp_abs = anom + clim_h    # (50, 100, 240, 15) absolute temperature °C
```

**Determinism verified:** Two independent inference runs produce bitwise-identical results (`max|run1−run2| = 0.0`).

---

## Pipeline Steps

```bash
# Step 0: Verify raw downloads integrity
python src/00_verify_raw_downloads.py

# Step 1: Fetch satellite data (requires credentials - see below)
python src/01_fetch_data.py

# Step 2: Harmonize all fields to common 100×240 daily grid
python src/02_harmonize.py

# Step 3: Compute climatology and anomaly fields
python src/03_compute_climatology.py

# Step 4: Sanity check processed cube
python src/04_sanity_check.py

# Step 5: Pre-handoff verification gate
python src/05_verify_before_handoff.py

# Training + Inference: run the Jupyter notebook
jupyter notebook notebooks/OceanEmbed_Training_Pipeline.ipynb
```

> ⚠️ **DO NOT retrain the frozen model.** The committed `exports/oceanembed_model.keras` is the final frozen artifact. Retraining without identical data, environment, and random seeds will produce different weights.

---

## Fetching Raw Data

Raw datasets (~18 GB total) are **not committed to this repository**. They are downloaded via:

- **CMEMS (Copernicus Marine Service):** `copernicusmarine login` (stores token locally via `~/.copernicusmarine`)
- **NASA Earthdata / PO.DAAC:** `earthaccess.login(strategy="netrc")` (uses `~/.netrc`)

Credentials are **never hardcoded** in any source file. See `src/01_fetch_data.py` for exact dataset IDs, DOIs, and date ranges.

Raw data products:
| Product | Source | ~Size |
|---|---|---|
| OSTIA SST | CMEMS `SST_GLO_SST_L4_REP_OBSERVATIONS_010_011` | ~200 MB |
| SMAP/SMOS SSS | CMEMS `MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013` | ~65 MB |
| DUACS SSH/SLA | CMEMS `SEALEVEL_GLO_PHY_L4_MY_008_047` | ~130 MB |
| OSCAR currents | PO.DAAC `OSCAR_L4_OC_FINAL_V2.0` | ~5.6 GB |
| CCMP winds | PO.DAAC `CCMP_WINDS_10M6HR_L4_V3.1` | ~5.6 GB |
| GLORYS temperature | CMEMS `GLOBAL_MULTIYEAR_PHY_001_030` | ~1.1 GB |
| Gridded ARGO | INCOIS LAS / Argo GDAC | ~2.3 MB |

---

## Large Artifact Storage (Git LFS)

The following files are stored in **Git LFS**:

| File | Size | Role |
|---|---|---|
| `exports/oceanembed_model.keras` | 0.8 MB | Frozen model weights |
| `data/processed/oceanembed_cube_with_climatology.nc` | 409 MB | Canonical processed cube |
| `exports/reconstructions.csv` | 269 MB | Final reconstruction output (50-day holdout) |
| `exports/reconstructions.parquet` | 96 MB | Parquet copy of above |
| `exports/inversion_candidates.csv` | 2.7 MB | Inversion segment data (19,233 segments) |
| `exports/inversion_profiles.csv` | 1.4 MB | Inversion profile list (16,918 profiles) |
| `exports/all_scientific_nulls_ledger.csv` | 12.7 MB | Complete 97,263-row NULL ledger |
| `exports/missingness_cell_audit_argo.csv` | 1.4 MB | Cell-level ARGO missingness audit |

To clone with LFS content:
```bash
git lfs install
git clone https://github.com/sg721642/OceanEmbed.git
```

To clone without LFS blobs (code only, LFS pointers only):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/sg721642/OceanEmbed.git
```

See `exports/artifact_checksums.json` for SHA-256 of all large artifacts.

**Note:** `exports/regime_profile_level.csv` (89 MB — full depth-profile dataset used in regime audit) is NOT committed to Git. It is reproducible from the frozen model and cube using the pipeline notebook. Its SHA-256 is recorded in `exports/artifact_checksums.json`.

---

## What Is NOT Committed

| Excluded | Reason |
|---|---|
| `data/raw/` | 18+ GB raw satellite datasets — fetch via `src/01_fetch_data.py` |
| `data/processed/oceanembed_cube.nc` | Intermediate cube (pre-climatology) — reproducible |
| `exports/regime_profile_level.csv` | 89 MB intermediate audit artifact — reproducible |
| `.venv/`, `.keras/`, `.matplotlib_cache/` | Local environment artifacts |
| `OceanEmbed_MASTER.md` | Internal development scratch document |
| `test_out.txt`, `train_candidate.log` | Scratch/test files |
| `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store` | OS/IDE artifacts |
| `exports/residual_correction_head.keras` | Experimental artifact not used in final frozen inference |

---

## Repository Structure

```
OceanEmbed/
├── README.md                              ← this file
├── requirements.txt                       ← loose requirements for pipeline
├── .gitignore
├── .gitattributes                         ← Git LFS tracking rules
│
├── src/                                   ← data pipeline scripts
│   ├── 00_verify_raw_downloads.py
│   ├── 01_fetch_data.py
│   ├── 02_harmonize.py
│   ├── 03_compute_climatology.py
│   ├── 04_sanity_check.py
│   └── 05_verify_before_handoff.py
│
├── notebooks/
│   └── OceanEmbed_Training_Pipeline.ipynb ← final training + evaluation notebook
│
├── data/processed/
│   ├── holdout_config.json                ← holdout definition
│   ├── variable_names.json                ← canonical variable name map
│   └── oceanembed_cube_with_climatology.nc ← [Git LFS] canonical processed cube
│
└── exports/
    ├── oceanembed_model.keras              ← [Git LFS] frozen model (SHA verified)
    ├── input_scaler.npz                   ← channel normalization statistics
    ├── model_output_order.json
    ├── requirements-final.txt             ← pinned exact versions
    ├── environment_manifest.json
    ├── artifact_checksums.json            ← SHA-256 of all large artifacts
    │
    ├── reconstructions.csv                ← [Git LFS] final reconstruction output
    ├── reconstructions.parquet            ← [Git LFS] parquet copy
    ├── regime_metrics.csv                 ← regime-level metrics (app handoff)
    ├── inversion_case_studies.csv         ← case study data (app handoff)
    ├── inversion_candidates.csv           ← [Git LFS] inversion segments
    ├── inversion_profiles.csv             ← [Git LFS] inversion profiles
    ├── all_scientific_nulls_ledger.csv    ← [Git LFS] NULL ledger
    ├── missingness_cell_audit_argo.csv    ← [Git LFS] ARGO missingness audit
    │
    ├── (Scientific result tables — small CSVs)
    ├── final_model_performance_summary.csv
    ├── final_depth_performance_reconciled.csv
    ├── final_statistical_recompute.csv
    ├── final_regime_performance.csv
    ├── final_daily_performance.csv
    ├── final_uncertainty_summary.csv
    ├── final_inversion_summary.csv
    ├── final_case_study_index.csv
    ├── scientific_claim_audit.csv
    ├── headline_metric_lineage.csv
    ├── metric_crosscheck_recompute.csv
    ├── inference_result_reconciliation.csv
    │
    ├── (Audit/reproducibility JSONs)
    ├── final_statistical_validity_summary.json
    ├── reproducibility_environment_audit.json
    ├── final_handoff_audit.json
    ├── final_freeze_audit.json
    ├── final_forensic_audit.json
    ├── inversion_selection_audit.json
    ├── run_summary.json
    │
    ├── (Reports)
    ├── README_FINAL.md
    ├── final_results_summary.md
    ├── final_model_audit.md
    ├── publication_readiness_audit.md
    ├── final_package_manifest.csv
    │
    └── (Publication figures)
        ├── final_daily_performance.png
        ├── final_spatial_error.png
        ├── argo_coverage_map.png
        ├── reconstruction_coverage.png
        ├── reconstruction_example_maps.png
        ├── reconstruction_uncertainty.png
        ├── regime_coverage.png
        ├── regime_daily_rmse.png
        ├── regime_depth_rmse.png
        ├── regime_spatial_breakdown.png
        ├── inversion_region_spatial_map.png
        ├── inversion_case_overview.png
        ├── inversion_case_plot.png
        ├── inversion_delta_distribution.png
        ├── inversion_depth_distribution.png
        ├── inversion_daily_frequency.png
        ├── inversion_magnitude_distribution.png
        └── inversion_frequency_by_region.png
```

---

## Backend / Application Handoff

Three primary application integration files:

### 1. `exports/reconstructions.csv`
Daily reconstructed temperature + uncertainty profiles for the 50-day holdout.

| Column | Type | Description |
|---|---|---|
| `obs_date` | ISO date string | Date (YYYY-MM-DD) |
| `lat` | float64 | Latitude (°N, 5.125–29.875) |
| `lon` | float64 | Longitude (°E, 45.125–104.875) |
| `temperature_c` | PG array string | `{t0,t5,...,t1000}` — 15 depths, NULL if below seafloor |
| `uncertainty_c` | PG array string | `{σ0,σ5,...,σ1000}` — predicted σ (uncalibrated) |
| `regime` | string | `"arabian_sea"` or `"bay_of_bengal"` |
| `ocean_valid` | bool | True = active ocean cell |
| `depth_coverage_count` | int | Number of valid (non-NULL) depth levels |
| `min_valid_depth_m` | int | Shallowest valid depth |
| `max_valid_depth_m` | int | Deepest valid depth |

Parse temperature array: `{29.2,28.5,...,12.1,NULL,NULL}` → split by comma, convert NULL→NaN.

### 2. `exports/regime_metrics.csv`
Regime-stratified RMSE/MAE/bias by depth.

### 3. `exports/inversion_case_studies.csv`
24 deterministic inversion case studies (12 Arabian Sea, 12 Bay of Bengal). Columns: `case_id, obs_date, region, lat, lon, depth_m, glorys_temp_c, model_temp_c, argo_temp_c, climatology_temp_c, model_uncertainty_c`.

---

## Publication Caveats (Required)

> All results must be presented with these caveats:

1. **Domain/time scope:** North Indian Ocean only; January–June 2023 only. Generalizability not demonstrated.
2. **50-day holdout:** Short evaluation window. Seasonal representativeness is limited.
3. **GLORYS is a numerical reanalysis benchmark** — not physical ground truth. Assimilates satellite SLA/SST and in-situ profiles.
4. **ARGO is sparse independent in-situ:** 4 cycle dates, Arabian Sea only, 0m excluded, 0 in Bay of Bengal.
5. **Uncertainty calibration NOT DEMONSTRATED** — empirical 1σ ~12.9% vs nominal 68.3%.
6. **Pooled 4D r (~0.987) dominated by vertical temperature gradient** — NOT spatial skill.
7. **Results are configuration-specific** — may differ under different temporal splits.

---

## License and Attribution

**SIH Problem Statement:** PS 26066  
**Institution:** INCOIS  
**Team:** Satyam Gupta, Shivansh, Ayush, Sudipto  
**Framework:** OceanEmbed v3  

*Model weights, training code, and evaluation outputs are provided for reproducibility and demonstration. Not for operational deployment without further validation.*
