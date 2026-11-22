# Ammonia-AI M1 Validation Report

## Overview
This report documents the validation of the screening-v3 baseline for N* adsorption energies.

## Metrics
- **MAE (clean rows):** 0.578 eV (5-fold, 200 iterations).
- **MAE (intermetallic class 0):** 0.464 eV.
- **Leave-element-out (54 elements):** 0.636 eV.

## Uncertainty Validation (A5 - Calibrated)
Full report available in `results/a5_final.json`.

| Adsorbate | Coverage (90% CI) | Spearman (Width vs |Err|) |
| :--- | :--- | :--- |
| N | 0.879 | 0.175 |
| N2 | 0.878 | 0.128 |
| NH | 0.885 | 0.175 |
| NH2 | 0.891 | 0.206 |
| NH3 | 0.895 | 0.135 |

**Overall Spearman Rho:** 0.224

## Limitations
1. Average global coverage hides per-adsorbate variation before per-adsorbate quantile calibration.
2. Conformal intervals assume exchangeability under chemical-system grouping, where out-of-group OOD domains can exhibit slight tail dispersion differences.
