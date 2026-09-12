# OceanEmbed: Subsurface Ocean Thermal Structure Reconstruction
## Final Scientific Results & Comprehensive Reference Summary

**Release Date:** 2026-09-12  
**Evaluation Scope:** 50-Day Temporal Holdout (`2023-05-11` to `2023-06-29`)  
**Domain:** North Indian Ocean ($55.0^\circ\text{E} - 84.75^\circ\text{E}$, $-0.25^\circ\text{N} - 24.5^\circ\text{N}$)  
**Repository:** `/Users/satyamgupta/Documents/OceanEmbed/files/oceanembed_project`  
**Model Checkpoint:** `exports/oceanembed_model.keras` (SHA-256: `39eb06cd6f73e8866f1914fd6ce560079eb3bc3a271829c330aa923cdcc91bb5`)  
**Canonical Data Cube:** `data/processed/oceanembed_cube_with_climatology.nc` (SHA-256: `c56ae615750acba2c7fd9e36955716d500d61d29482001bd9daea494aac92b79`)  

---

## 1. Architecture & Reconstruction Methodology

OceanEmbed is a deep neural network architecture designed for subsurface ocean temperature reconstruction across 15 canonical oceanographic depth levels from surface satellite observations.

### Multi-Modal Surface Boundary Forcing:
- **Altimetry:** Sea Level Anomaly (SLA), Absolute Dynamic Topography (ADT), and surface geostrophic current velocities ($u_{gos}, v_{gos}$).
- **Surface Atmospheric:** Sea Surface Temperature (SST), Sea Surface Salinity (SSS), and surface wind stress components ($\tau_x, \tau_y$).
- **Oceanographic Climatology:** Historical multi-year seasonal baseline temperature profiles $T_{\text{clim}}(x, y, z, \text{doy})$.

### Subsurface Reconstruction:
$$\hat{T}(x, y, z, t) = T_{\text{clim}}(x, y, z, \text{doy}) + \Delta T_{\text{pred}}(x, y, z, t)$$

The model predicts anomaly deviations $\Delta T_{\text{pred}}$ added directly to the seasonal climatological background across all 15 canonical depth levels:  
`[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters.

---

## 2. Headline Performance Summary

All metrics were evaluated deterministically over the 50-day holdout (`exports/final_model_performance_summary.csv` and `exports/headline_metric_lineage.csv`):

```
+===================================================================================================================+
|                                    MASTER PERFORMANCE BENCHMARK                                                   |
+================================+==========+==========+==========+==========+==========+==========+================+
| Evaluation Scope               | N Pts    | RMSE (°C)| MAE (°C) | Bias (°C)| MedAE(°C)| P95AE(°C)| Pearson r      |
+================================+==========+==========+==========+==========+==========+==========+================+
| Overall GLORYS Reanalysis      |16,064,650| 1.4913   | 1.0519   | +0.7673  | 0.7260   | 3.2309   | 0.9869         |
| Arabian Sea GLORYS             | 7,664,650| 1.6688   | 1.1519   | +0.8662  | 0.7574   | 3.6767   | 0.9828         |
| Bay of Bengal GLORYS           | 8,400,000| 1.3084   | 0.9606   | +0.6771  | 0.7042   | 2.8416   | 0.9906         |
+--------------------------------+----------+----------+----------+----------+----------+----------+----------------+
| Independent In-Situ ARGO Floats| 25,904   | 1.8488   | 1.5654   | +1.5333  | 1.4684   | 3.3569   | 0.9876         |
|   Arabian Sea ARGO             | 25,904   | 1.8488   | 1.5654   | +1.5333  | 1.4684   | 3.3569   | 0.9876         |
|   Bay of Bengal ARGO           | 0        | —        | —        | —        | —        | —        | [No Floats]    |
+===================================================================================================================+
```

### Global Count Accounting:
- **Total Holdout Profiles Screened:** **1,116,650** vertical profiles ($22,333 \times 50$).
- **Total Holdout Model Depth Predictions:** **16,749,750** values ($1,116,650 \times 15$).
- **GLORYS Valid Paired Count:** **16,064,650** points (exactly $685,100$ cells are below local seafloor bathymetry where GLORYS is NaN).
- **ARGO Valid Paired Count:** **25,904** points (Arabian Sea, 4 discrete cycle dates).

---

## 3. Depth-Stratified Performance Benchmarks

Evaluated on the 15 canonical depths and pooled aggregates (`final_depth_performance_reconciled.csv`):

```
+==============================================================================================================+
|                             STRATIFIED DEPTH REGIME BENCHMARKS                                               |
+=========================+========================================+===========================================+
| Depth Scope             | GLORYS Reanalysis Benchmark            | Independent In-Situ ARGO Benchmark        |
|                         | N Valid Pairs | RMSE (°C) | MAE (°C)   | N Valid Pairs | RMSE (°C) | MAE (°C)      |
+=========================+===============+===========+============+===============+===========+===============+
| 0–100m aggregate        | 8,800,800     | 1.6055    | 1.1213     | 18,668        | 2.0340    | 1.8053        |
| 125–300m aggregate      | 4,250,250     | 1.6831    | 1.3204     |  5,108        | 0.9758    | 0.7962        |
| 500–1000m aggregate     | 3,013,600     | 0.5762    | 0.4705     |  2,128        | 1.7414    | 1.3072        |
+-------------------------+---------------+-----------+------------+---------------+-----------+---------------+
| 1000m depth             |   987,600     | 0.6089    | 0.5148     |    664        | 2.9288    | 2.9116        |
+==============================================================================================================+
```

### Empirical Observational Findings by Depth:
- In the GLORYS benchmark, the largest empirical error occurred at depth 125m (RMSE = 2.0790°C), while the smallest empirical error occurred at depth 500m (RMSE = 0.4316°C).
- In the independent ARGO float observations, the largest empirical error occurred at depth 100m (RMSE = 3.2662°C), while the smallest empirical error occurred at depth 500m (RMSE = 0.2590°C).
- These depth error distributions represent empirical findings across the 50-day holdout; dynamic causal interpretations (e.g., pycnocline displacement, internal wave shear) remain oceanographic hypotheses not directly measured or demonstrated by this audit.

---

## 4. Regional Regime Differences

- **Arabian Sea:** OceanEmbed achieves 1.67°C RMSE against GLORYS (7,664,650 points) and 1.85°C against independent ARGO floats (25,904 points).
- **Bay of Bengal:** OceanEmbed achieves 1.31°C RMSE against GLORYS (8,400,000 points; MAE 0.96°C; $r = 0.9906$). No ARGO floats operated in the Bay of Bengal during the 50-day holdout.

---

## 5. Subsurface Thermal Inversion Science

- **Definition:** Temperature increase $\Delta T \ge +0.05^\circ\text{C}$ between adjacent depth levels in the upper 100 meters.
- **Profiles with Inversions:** **16,918** profiles (1.52% overall frequency).
  - *Arabian Sea:* 12,465 profiles (2.24% regional frequency).
  - *Bay of Bengal:* 4,453 profiles (0.80% regional frequency).
- **Total Discrete Inversion Segments:** **19,233** segments.
- **Magnitude Distribution (Strict Reconciled Bins):**
  - $0.05 \le \Delta T < 0.10^\circ\text{C}$: **13,654** (70.99%)
  - $0.10 \le \Delta T < 0.20^\circ\text{C}$: **4,398** (22.87%)
  - $0.20 \le \Delta T < 0.30^\circ\text{C}$: **554** (2.88%)
  - $0.30 \le \Delta T < 0.50^\circ\text{C}$: **377** (1.96%)
  - $0.50 \le \Delta T < 1.00^\circ\text{C}$: **248** (1.29%)
  - $\Delta T \ge 1.00^\circ\text{C}$: **2** (0.01%, Max $\Delta T = +1.1318^\circ\text{C}$)
  - **Sum:** **19,233** segments ($100.00\%$).
- **Case Study Sampling:** Selected via **deterministic chronological-regional sampling** (12 Arabian Sea, 12 Bay of Bengal; indexed in `final_case_study_index.csv`).

---

## 6. Uncertainty Diagnostics & Limitations

- **Correlation($\sigma$, $|e|$):** 0.2017 (Modest positive error tracking).
- **Empirical 1-$\sigma$ Interval Coverage:** 12.92% (Nominal Gaussian expectation: 68.27%).
- **Empirical 2-$\sigma$ Interval Coverage:** 24.84% (Nominal Gaussian expectation: 95.45%).
- **Uncertainty Calibration Status:** **NOT DEMONSTRATED.**  
  Predicted uncertainty $\sigma(x, y, z, t)$ must not be interpreted as calibrated Gaussian confidence bounds; it functions strictly as uncalibrated model-inferred variance.

---

## 7. Lineage and Data Availability

- **Headline Lineage:** Documented in `exports/headline_metric_lineage.csv` (18 verified rows).
- **Reconciled Depth Metrics:** `exports/final_depth_performance_reconciled.csv`.
- **Master NULL Ledger:** `exports/all_scientific_nulls_ledger.csv` (97,263 verified scientific missing values).
- **Inversion Population Summary:** `exports/final_inversion_summary.csv`.
- **Claim Audit:** `exports/scientific_claim_audit.csv`.
- **Model Checkpoint:** `exports/oceanembed_model.keras`.

---

*Google DeepMind — Advanced Agentic Coding / OceanEmbed Research Suite*
