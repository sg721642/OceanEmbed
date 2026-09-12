# OceanEmbed-v3 — Final Scientific Forensic Audit Report

**Document Version**: 8.0-FINAL-RECONCILED-NULL-ACCOUNTING  
**Date**: 2026-09-12  
**Project Root**: `/Users/satyamgupta/Documents/OceanEmbed/files/oceanembed_project`  
**Evaluation Scope**: Strict 50-day Holdout (`2023-05-11` to `2023-06-29`)  

---

## 1. Frozen Environment & Cryptographic Hashes

The production model, canonical NetCDF dataset, holdout configuration, and pipeline notebooks are completely frozen. Cryptographic SHA256 hashes are recorded in `exports/final_freeze_audit.json`:

| Artifact | File Path | Size (Bytes) | SHA256 Digest | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Model Weights** | `exports/oceanembed_model.keras` | 853,228 | `39eb06cd6f73e8866f1914fd6ce560079eb3bc3a271829c330aa923cdcc91bb5` | **FROZEN** |
| **Canonical Cube** | `data/processed/oceanembed_cube_with_climatology.nc` | 429,264,108 | `c56ae615750acba2c7fd9e36955716d500d61d29482001bd9daea494aac92b79` | **FROZEN** |
| **Holdout Config** | `data/processed/holdout_config.json` | 519 | `75f82449d2f2a7b45222e8b3d6dfbef03171e114cc49406e0cb8d970ef4e7ef8` | **FROZEN** |
| **Project Notebook**| `notebooks/OceanEmbed_Training_Pipeline.ipynb` | 230,843 | `62236fce5e4e601b00801d4e9891cf0cafb8bc668a254d280b2ab289aed31eaf` | **IDENTICAL** |
| **Canonical Notebook**| `OceanEmbed_Training_Pipeline_CORRECTED.ipynb` | 230,843 | `62236fce5e4e601b00801d4e9891cf0cafb8bc668a254d280b2ab289aed31eaf` | **IDENTICAL** |

---

## 2. Authoritative NULL Accounting Reconciliation

Every single NULL value across all 23 production CSV deliverables (**97,263 total scientific NULL cells**) has been cataloged in `exports/all_scientific_nulls_ledger.csv` (97,263 rows) and verified against physical source data and masks:
- **Direct CSV Scientific NULLs**: **97,263**
- **Master Ledger NULLs**: **97,263**
- **Difference**: **0**
- **Duplicate Ledger Rows**: **0**
- **Omitted NULL Cells**: **0**
- **Unexplained NULL Cells**: **0**
- **Recoverable NULL Cells**: **0** (Zero valid real source observations were omitted from any deliverable)
- **Unrecoverable NULL Cells**: **97,263** (Genuinely absent from physical instruments or bathymetrically undefined)
- **Production CSV Count**: **23 files**
- **Total Production Columns**: **284 columns** (Audited in `exports/field_completeness_audit.csv`)
- **Field Audit Mismatches**: **0**

### Reconciled Missingness Breakdown by Physical Mechanism:

| Physical Mechanism | NULL Cell Count | Percentage | Artifacts & Fields Affected | Verified Cause |
| :--- | :---: | :---: | :--- | :--- |
| **`NO_VALID_PAIR`** | **58,611** | 60.26% | Metric tables (22,488) + Inversion daily (184) + Candidate argo_inversion (19,110) + Profile ARGO_supported (16,805) + Case summary argo_inversion (24) | Target has 0 observations (e.g. BoB ARGO, 0m depth, off-cycle dates, or 0 float profiles in candidate segments) |
| **`ARGO_OFF_CYCLE`** | **36,594** | 37.62% | Candidate argo_top_c (18,143) + argo_bottom_c (18,143) + Case studies argo_temp_c (308) | Date is not one of the 4 active float profiling dates (May 20, May 30, Jun 10, Jun 20) |
| **`NO_ARGO_SOURCE_COVERAGE`** | **1,962** | 2.02% | Candidate argo_top_c (967) + argo_bottom_c (967) + Case studies argo_temp_c (28) | Active cycle date, but coordinate lacks float profile (including entire Bay of Bengal) |
| **`BELOW_LOCAL_SEAFLOOR`** | **72** | 0.07% | Case studies glorys_temp_c (24) + model_temp_c (24) + climatology_temp_c (24) | Depths $\ge 75	ext{m}$ below local continental shelf bathymetry |
| **`ARGO_SURFACE_CUTOFF`** | **24** | 0.02% | Case studies argo_temp_c at 0m (24) | Float CTD sensor cutoff near 3–5m depth (depth 0m physically has 0 float observations) |
| **Total** | **97,263** | **100.00%** | **All 23 Production CSVs** | **Sum of all mechanisms reconciles exactly to 97,263** |

---

## 3. Discrepancy Investigations & Resolutions

### A. The 77,769 vs. 13,324 Discrepancy Resolved
- In the previous surgical audit, `77,769` was erroneously calculated as an unverified subtraction remainder: $97,263 - (18,451 + 24 + 995 + 24)$.
- In reality, `NO_VALID_PAIR` is exactly **58,611** cells. It consists of:
  1. **22,488 cells** across the 8 regime metric tables where target observations are zero.
  2. **184 cells** in `inversion_metrics_daily.csv` where 0 qualifying inversions existed on that day/regime.
  3. **35,939 cells** in derived pairwise flags (`argo_inversion` in candidates [19,110], `ARGO_supported` in profiles [16,805], and `argo_inversion` in case summary [24]) that require finite ARGO measurements at multiple levels.
- Direct metric audits confirmed: **100% of metric NULLs have `paired_count == 0`**. Zero metric NULLs exist where pairs > 0.

### B. The 293 vs. 294 Column Count Discrepancy Resolved
- The 23 production CSV deliverables contain exactly **284 columns**.
- Previous audits included `field_completeness_audit.csv` itself:
  - In Document Version 6.0, `field_completeness_audit.csv` had 10 columns ($284 + 10 = 294$).
  - In Document Version 7.0, `source` was merged into `verified_missing_reason`, giving 9 columns ($284 + 9 = 293$).
- The current accounting standardizes strictly on the **23 production CSV deliverables (284 production columns)**, documented in `exports/null_accounting_by_artifact.csv`.

---

## 4. Full-Grid Consistency & Provenance Verification

- **`reconstructions.csv` vs. Model Layer**: All 16,749,750 values (1,116,650 profiles $	imes$ 15 depths) verified with **0 mismatches** (Max diff: 0.000500°C).
- **Inversion Case Studies vs. Model Layer**: All 360 values verified with **0 mismatches** (Max diff: 0.000500°C).
- **GLORYS Provenance**: All 38,466 candidate endpoints checked against NetCDF `temperature_3d` with **0 mismatches**.
- **ARGO Provenance**: All 246 candidate endpoints checked against NetCDF `argo_temperature_3d` with **0 mismatches**.
- **Case Selection**: `deterministic chronological-regional sampling` (12 Arabian Sea, 12 Bay of Bengal).

---

---

## 5. Deliverables Inventory & Storage Footprint

Total export directory footprint: **`482.1 MB`** (well below the 1,024 MB Aiven / PostgreSQL storage budget).
All 23 production CSV files, master ledger, reconciliation tables, Parquet files, JSON audits, and 16 visualization PNGs are finalized in `exports/`.

---

## 6. Global Population & Holdout Count Reconciliation

- **Holdout Profiles Screened:** **1,116,650** vertical profiles ($22,333 \text{ active ocean cells} \times 50 \text{ holdout dates}$).
- **Holdout Model Depth Predictions:** **16,749,750** values ($1,116,650 \text{ profiles} \times 15 \text{ canonical depths}$).
- **GLORYS Valid Paired Count:** **16,064,650** wet ocean points above seafloor bathymetry (exactly **685,100** grid cells across the 50 holdout dates reside below local seafloor bathymetry where GLORYS is NaN).
- **ARGO Valid Paired Count:** **25,904** discrete in-situ float matchups (Arabian Sea, 4 discrete cycle dates).
- **Canonical Depth Levels:** Exactly 15 depths (`[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m).

---

## 7. Final Holdout Scientific Performance Benchmark

All metrics evaluated deterministically over the 50-day holdout (`2023-05-11` to `2023-06-29`) across all 22,333 ocean grid cells:

| Evaluation Scope | Benchmark Source | N Valid Observations | RMSE (°C) | MAE (°C) | Mean Bias (°C) | Median AE (°C) | P95 AE (°C) | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall GLORYS** | GLORYS12V1 Reanalysis | 16,064,650 | **1.4913** | **1.0519** | +0.7673 | 0.7260 | 3.2309 | **0.9869** |
| **Arabian Sea GLORYS** | GLORYS12V1 Reanalysis | 7,664,650 | **1.6688** | **1.1519** | +0.8662 | 0.7574 | 3.6767 | **0.9828** |
| **Bay of Bengal GLORYS**| GLORYS12V1 Reanalysis | 8,400,000 | **1.3084** | **0.9606** | +0.6771 | 0.7042 | 2.8416 | **0.9906** |
| **Overall ARGO** | Independent ARGO Floats | 25,904 | **1.8488** | **1.5654** | +1.5333 | 1.4684 | 3.3569 | **0.9876** |
| **Arabian Sea ARGO** | Independent ARGO Floats | 25,904 | **1.8488** | **1.5654** | +1.5333 | 1.4684 | 3.3569 | **0.9876** |
| **Bay of Bengal ARGO** | Independent ARGO Floats | 0 | — | — | — | — | — | *No float coverage* |

### Depth-Stratified Reconciled Benchmarks:
- **0–100m aggregate (8 depths):** GLORYS RMSE = **1.6055°C** ($n = 8,800,800$); ARGO RMSE = **2.0340°C** ($n = 18,668$).
- **125–300m aggregate (4 depths):** GLORYS RMSE = **1.6831°C** ($n = 4,250,250$); ARGO RMSE = **0.9758°C** ($n = 5,108$).
- **500–1000m aggregate (3 depths):** GLORYS RMSE = **0.5762°C** ($n = 3,013,600$); ARGO RMSE = **1.7414°C** ($n = 2,128$).
- **1000m Depth:** GLORYS RMSE = **0.6089°C** ($n = 987,600$); ARGO RMSE = **2.9288°C** ($n = 664$).
- *Authoritative lineage documented in `exports/headline_metric_lineage.csv`.*

---

## 8. Uncertainty Calibration Diagnostics

- **Correlation($\sigma$, $|e|$):** `0.2017` (Modest positive error tracking).
- **Empirical 1-$\sigma$ Coverage:** `12.92%` (Nominal Gaussian expectation: `68.27%`).
- **Empirical 2-$\sigma$ Coverage:** `24.84%` (Nominal Gaussian expectation: `95.45%`).
- **Calibration Status:** **`NOT DEMONSTRATED`** (Mandatory transparent caveat: $\sigma$ must be reported strictly as uncalibrated model-inferred variance).

---

## 9. Subsurface Thermal Inversion Summary

- **Total Holdout Profiles Screened:** 1,116,650 (22,333 ocean cells $\times$ 50 dates).
- **Profiles with Inversions ($\Delta T \ge 0.05^\circ\text{C}, z \le 100\text{ m}$):** **16,918** (1.52% overall frequency).
  - *Arabian Sea:* 12,465 profiles (2.24% regional frequency).
  - *Bay of Bengal:* 4,453 profiles (0.80% regional frequency).
- **Total Discrete Inversion Segments:** **19,233** segments.
  - $0.05 \le \Delta T < 0.10^\circ\text{C}$: **13,654** (70.99%)
  - $0.10 \le \Delta T < 0.20^\circ\text{C}$: **4,398** (22.87%)
  - $0.20 \le \Delta T < 0.30^\circ\text{C}$: **554** (2.88%)
  - $0.30 \le \Delta T < 0.50^\circ\text{C}$: **377** (1.96%)
  - $0.50 \le \Delta T < 1.00^\circ\text{C}$: **248** (1.29%)
  - $\Delta T \ge 1.00^\circ\text{C}$: **2** (0.01%, Max $\Delta T = +1.1318^\circ\text{C}$)
  - **Sum of Bins:** **19,233** (100.00% exact mathematical match).
- **Sampling Methodology:** `deterministic chronological-regional sampling` (12 AS, 12 BoB case studies).

---

## 10. Final Publication Readiness Verdict

**Verdict:** **`READY_WITH_TRANSPARENT_CAVEATS`**  
All code, models, data cubes, holdout partitions, NULL accounting ledgers, depth metrics, and inversion bins have passed strict anti-fabrication and numerical consistency audits. All unsupported causal physical assertions have been removed. The repository is certified ready for manuscript submission.

