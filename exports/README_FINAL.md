# OceanEmbed — Final Handoff README

**Version:** Production Freeze 2026-09-12  
**Status:** `READY_WITH_TRANSPARENT_CAVEATS`  
**Project Root:** `/Users/satyamgupta/Documents/OceanEmbed/files/oceanembed_project`

---

## 1. What Is OceanEmbed?

OceanEmbed is a deep learning system for **subsurface ocean temperature reconstruction** across the North Indian Ocean. It reconstructs full vertical temperature profiles at 15 canonical depth levels (0–1000 m) from daily satellite surface observations, using a 7-day temporal window of surface forcing fields.

**Reconstruction formula:**
```
T_pred(x, y, z, t) = T_climatology(x, y, z, doy) + ΔT_model(x, y, z, t)
```

The model learns anomaly deviations from the seasonal climatological background, then adds them back to produce absolute temperatures.

---

## 2. Canonical Data

| Artifact | Path | SHA-256 |
|---|---|---|
| **Canonical NetCDF cube** | `data/processed/oceanembed_cube_with_climatology.nc` | `c56ae615...aac92b79` |
| **Frozen model weights** | `exports/oceanembed_model.keras` | `39eb06cd...cc91bb5` |
| **Holdout config** | `data/processed/holdout_config.json` | `75f82449...ef4e7ef8` |
| **Training notebook** | `notebooks/OceanEmbed_Training_Pipeline.ipynb` | `62236fce...ed31eaf` |

The canonical NetCDF cube covers **180 days** (Jan 1 – Jun 29, 2023), the **North Indian Ocean** (55°E–84.75°E, -0.25°N–24.5°N, 0.25° resolution), and **15 canonical depth levels**: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters.

---

## 3. Frozen Model

- **Path:** `exports/oceanembed_model.keras`
- **Architecture:** `OceanEmbed-v3-Strengthened`
- **Input 0:** `surface_features` — shape `(None, 100, 240, 70)` — 10 surface channels × 7 temporal lag days
- **Input 1:** `regime_gate` — shape `(None, 100, 240, 1)` — binary flag (0 = Arabian Sea lon<77°E, 1 = Bay of Bengal lon≥77°E)
- **Output 0:** Temperature anomaly — shape `(None, 100, 240, 15)` — per-depth anomaly prediction
- **Output 1:** Predicted uncertainty σ — shape `(None, 100, 240, 15)` — uncalibrated model variance
- **Total parameters:** 187,842
- **Load command:** `model = keras.models.load_model("exports/oceanembed_model.keras", compile=False)`

---

## 4. Holdout Definition

| Parameter | Value |
|---|---|
| Training window | Day 0 (2023-01-01) through Day 129 (2023-05-10) |
| Validation window | Days 90–129 (used only for early stopping) |
| **Holdout window** | **Day 130 (2023-05-11) through Day 179 (2023-06-29) — 50 consecutive days** |
| Holdout ocean cells | 22,333 active ocean cells (100×120 grid, land-masked) |
| Total model predictions | 16,749,750 values (22,333 × 50 × 15) |
| GLORYS valid paired count | 16,064,650 (685,100 cells below seafloor bathymetry = NaN) |

**The holdout was strictly withheld from training. The checkpoint was selected at Epoch 5 (val_loss=0.2528) before the holdout start date.**

---

## 5. Exact Python Environment

**CRITICAL:** Use the `oceanembed` conda environment (Python 3.12). The base Anaconda Python 3.13 causes a **segmentation fault** during Keras backend initialization and cannot be used.

```bash
# Activate the correct environment
conda activate oceanembed

# Verify Python version
python --version  # Should print Python 3.12.x

# Path
/opt/anaconda3/envs/oceanembed/bin/python
```

**Pinned package versions (`exports/requirements-final.txt`):**
```
tensorflow==2.16.2
tensorflow-metal==1.2.0
keras==3.15.1
numpy==1.26.4
pandas==3.0.5
xarray==2026.7.0
matplotlib==3.11.1
scipy==1.17.1
h5py==3.16.0
netCDF4==1.7.4
pyarrow==25.0.1
```

See `exports/environment_manifest.json` for full OS and architecture details.

---

## 6. Reproducing Frozen Inference

```python
import os, numpy as np, xarray as xr, keras
os.environ["KERAS_BACKEND"] = "tensorflow"

PROJECT_ROOT   = "/Users/satyamgupta/Documents/OceanEmbed/files/oceanembed_project"
CANONICAL_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
INPUT_CHANNELS   = ["sst","sss","ssh","sla","u_geo","v_geo","u_ageo","v_ageo","wind_u","wind_v"]
TEMPORAL_WINDOW  = 7
HOLDOUT_DAYS     = 50
REGIME_SPLIT_LON = 77.0

cube  = xr.open_dataset(f"{PROJECT_ROOT}/data/processed/oceanembed_cube_with_climatology.nc")
times, lats, lons = (cube.time.values, cube.lat.values, cube.lon.values)
split_idx, time_offset = len(times) - HOLDOUT_DAYS, TEMPORAL_WINDOW - 1
split_idx_windowed     = split_idx - time_offset

X_raw = np.stack([cube[ch].values for ch in INPUT_CHANNELS], axis=-1).astype(np.float32)
X_flat = X_raw[:split_idx].reshape(-1, X_raw.shape[-1])
ch_mean, ch_std = np.nanmean(X_flat, axis=0), np.nanstd(X_flat, axis=0)
ch_std[ch_std < 1e-6] = 1.0
X_norm = np.nan_to_num((X_raw - ch_mean) / ch_std, nan=0.0)

# Build 7-day temporal window
def add_temporal_window(X, window):
    T, H, W, C = X.shape
    out = np.zeros((T - window + 1, H, W, C * window), dtype=np.float32)
    for t in range(window - 1, T):
        out[t - window + 1] = np.concatenate([X[t - k] for k in range(window)], axis=-1)
    return out

X_windowed = add_temporal_window(X_norm, TEMPORAL_WINDOW)
X_holdout  = X_windowed[split_idx_windowed:]

N_LAT, N_LON = len(lats), len(lons)
regime_gate  = np.where(lons[None,:] < REGIME_SPLIT_LON, 0., 1.).astype(np.float32)
regime_gate  = np.repeat(regime_gate, N_LAT, axis=0)
regime_batch = np.repeat(regime_gate[None,:,:,None], HOLDOUT_DAYS, axis=0)

model = keras.models.load_model(f"{PROJECT_ROOT}/exports/oceanembed_model.keras", compile=False)
anom_pred, unc_pred = model.predict([X_holdout, regime_batch], batch_size=4, verbose=0)

clim = cube["climatology"].transpose("time","lat","lon","depth").values.astype(np.float32)
clim_holdout = clim[time_offset:][split_idx_windowed:]
temp_absolute = anom_pred + clim_holdout   # (50, 100, 240, 15)
```

---

## 7. Reproducing Canonical Metrics

```python
ocean_mask = cube["ocean_mask"].values.astype(bool)
glorys_3d  = cube["temperature_3d"].transpose("time","lat","lon","depth").values[time_offset:][split_idx_windowed:]

def apply_mask(arr, mask):
    a = arr.copy(); a[:, ~mask, :] = np.nan; return a

pred_m   = apply_mask(temp_absolute, ocean_mask)
glorys_m = apply_mask(glorys_3d, ocean_mask)

# Pooled RMSE (NOT arithmetic mean of per-depth RMSEs)
v = np.isfinite(pred_m) & np.isfinite(glorys_m)
rmse = float(np.sqrt(np.mean((pred_m[v] - glorys_m[v])**2)))  # canonical = 1.4913°C
```

All metrics saved to `exports/final_statistical_recompute.csv` (50 rows) and verified against `exports/metric_crosscheck_recompute.csv`.

---

## 8. What Each Export Contains

| File | Contents |
|---|---|
| `reconstructions.parquet` | Primary reconstruction output. 1,116,650 rows. PG-array temperature/uncertainty strings. |
| `reconstructions.csv` | CSV copy of above. 269 MB. |
| `final_model_performance_summary.csv` | 6 headline metric rows. |
| `final_depth_performance_reconciled.csv` | 15 individual depths + 3 pooled strata for GLORYS and ARGO. |
| `final_statistical_recompute.csv` | Full independent metric recompute (50 rows) including Pearson r decomposition. |
| `final_inversion_summary.csv` | 6-bin inversion segment summary. 19,233 total segments. |
| `inversion_profiles.csv` | 16,918 qualifying inversion profiles. |
| `inversion_candidates.csv` | Segment-level inversion data. |
| `all_scientific_nulls_ledger.csv` | 97,263-row NULL ledger with mechanism labels. |
| `scientific_claim_audit.csv` | 7 claims, each SUPPORTED / SUPPORTED_WITH_CAVEAT / UNSUPPORTED_AND_REMOVED. |
| `final_statistical_validity_summary.json` | Master audit summary. |
| `reproducibility_environment_audit.json` | Environment, inference reproducibility, uncertainty calibration. |
| `headline_metric_lineage.csv` | 18-row metric lineage. |
| `requirements-final.txt` | Pinned pip packages for reproduction. |
| `environment_manifest.json` | OS, architecture, all package versions. |
| `final_package_manifest.csv` | This file — complete inventory with SHA-256 and purpose. |
| `publication_readiness_audit.md` | Full publication readiness report. |
| `final_results_summary.md` | Headline scientific results reference. |
| `README_FINAL.md` | This file. |

---

## 9. How NULLs Should Be Interpreted

**Total scientific NULLs in all production CSVs:** 97,263 (fully reconciled, zero unexplained).

| Mechanism | Count | Explanation |
|---|---|---|
| `NO_VALID_PAIR` | 58,611 | No ARGO float observation at this cell/date/depth. |
| `ARGO_OFF_CYCLE` | 36,594 | ARGO float on its 10-day descent cycle, not transmitting (46 of 50 holdout days). |
| `NO_ARGO_SOURCE_COVERAGE` | 1,962 | Cycle day but no float profile in this spatial cell. |
| `BELOW_LOCAL_SEAFLOOR` | 72 | Depth exceeds local bathymetry in case study tables. |
| `ARGO_SURFACE_CUTOFF` | 24 | 0m depth excluded (ARGO sensors shut off near surface). |

NULLs in `temperature_c` / `uncertainty_c` array strings (e.g., `{...,NULL,NULL}`) indicate depths below local seafloor bathymetry or land cells. These are **physically correct** and must not be imputed.

---

## 10. ARGO Limitations

- **Coverage:** 4 discrete 10-day cycle dates only: **May 15, May 25, June 4, June 14, 2023**.
- **Spatial scope:** Arabian Sea **only** — zero ARGO floats in Bay of Bengal during the 50-day holdout.
- **0m depth:** Zero observations (ARGO CTD sensors shut off near surface to prevent biofouling and air ingestion).
- **N valid:** 25,904 independent in-situ measurements.
- **Nomenclature:** Always refer to as "independent in-situ ARGO float observations" — never "truth", "ground truth", or "validation truth".
- **Sparsity:** ARGO within-depth Pearson r should not be interpreted as a spatial skill metric due to extreme sparsity.

---

## 11. GLORYS Role

GLORYS12V1 (Mercator Ocean Global Reanalysis) is used as the **primary numerical reference benchmark**, not physical ground truth. It is a numerical ocean model that:
- Assimilates satellite altimetry (SLA), sea surface temperature, and in-situ profiles
- Provides continuous 4D spatial-temporal ocean state estimates
- Contains numerical model physics errors and observation assimilation artifacts

**Always label as:** "GLORYS12V1 numerical ocean reanalysis benchmark" — **never** "ground truth" or "truth".

---

## 12. Uncertainty Limitation

**Calibration Status: NOT DEMONSTRATED.**

| Metric | Value | Nominal (Gaussian) |
|---|---|---|
| r(σ, \|error\|) | ~0.20 | — |
| 1σ empirical coverage | ~12.9% | 68.27% |
| 2σ empirical coverage | ~24.8% | 95.45% |

The predicted uncertainty output σ(x,y,z,t) must be reported strictly as **uncalibrated model-inferred variance**, not as calibrated Bayesian confidence bounds.

---

## 13. Inversion Definition

A **thermal inversion** in this context is defined as:
> An adjacent-depth temperature increase ΔT ≥ +0.05°C in the upper 100 meters of the water column (specifically, within the depth levels 0–100m).

Detected in reconstructed OceanEmbed temperature profiles (not GLORYS directly). Discrete inversion segments are adjacent depth-pair intervals where the temperature increases with depth.

**Bin notation:** All bins use strict left-closed, right-open intervals: `[low, high)` — e.g., `[0.05, 0.10)°C`.

---

## 14. Database Loading Notes

**`reconstructions.parquet` / `reconstructions.csv`:**
```python
df = pd.read_parquet("exports/reconstructions.parquet")
# temperature_c is a PostgreSQL-style array string: '{29.1,28.5,...,NULL}'
# Parse with:
import ast, numpy as np
def parse_temp(s):
    if not isinstance(s, str): return [np.nan]*15
    toks = s.strip().strip('{}').split(',')
    return [float(t) if t.strip().upper() != 'NULL' else np.nan for t in toks]
df['temps'] = df['temperature_c'].apply(parse_temp)
```

**Coordinate conventions:**
- `lon`: degrees East (positive-East convention, range 55.0–84.75°E)
- `lat`: degrees North (range -0.25–24.5°N)
- `depth_m`: meters below surface (positive downward)
- `obs_date`: ISO 8601 `YYYY-MM-DD` string format
- `regime`: `"arabian_sea"` or `"bay_of_bengal"` (lon < 77°E = Arabian Sea)

**Missing value representation:** `NULL` (string) in array columns; `NaN` in numeric columns.

---

## Final Scientific Headline

```
GLORYS Reanalysis Benchmark (16,064,650 valid 4D pairs):
  Pooled RMSE:   1.4913°C   ← PRIMARY metric (pool all pairs, then RMSE)
  Pooled MAE:    1.0519°C
  Mean Bias:    +0.7673°C
  Pooled 4D r:   0.9869     ← SECONDARY; dominated by vertical gradient; NOT spatial skill

Independent In-Situ ARGO Observations (25,904 valid pairs):
  Pooled RMSE:   1.8488°C
  Pooled MAE:    1.5654°C
  Mean Bias:    +1.5333°C
  Pooled 4D r:   0.9876     ← SECONDARY; dominated by vertical gradient; NOT spatial skill

GLORYS Mean Within-Depth r:   0.4592  ← PRIMARY spatial reconstruction skill indicator
GLORYS Median Within-Depth r: 0.6143
```

**IMPORTANT:** Pooled RMSE = `sqrt(mean((pred-obs)²))` over ALL jointly finite 4D pairs simultaneously. This is NOT the arithmetic mean of per-depth RMSEs (which equals ~1.40°C for GLORYS — a different and less representative metric).

---

*OceanEmbed Research — Final Reproducibility Lock 2026-09-12*
