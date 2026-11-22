# Ammonia-AI M1 Validation Report

## Overview
This report documents the validation of the screening-v3 baseline for N* adsorption energies.

## Metrics
- **MAE (clean rows):** 0.578 eV (5-fold, 200 iterations).
- **MAE (intermetallic class 0):** 0.464 eV.
- **Leave-element-out (54 elements):** 0.636 eV.

## Uncertainty Validation (A5 - Calibrated)
Full report available in `results/uncertainty_full.json`.
Protocol: 5-fold, 200 iterations, full protocol.

| Adsorbate | Coverage (90% CI) | Spearman (Width vs |Err|) |
| :--- | :--- | :--- |
| N | 0.883 | 0.184 |
| N2 | 0.872 | 0.120 |
| NH | 0.883 | 0.185 |
| NH2 | 0.888 | 0.204 |
| NH3 | 0.898 | 0.181 |

**Overall Spearman Rho:** 0.228

## Coverage per Reliability Tier
| Tier | Coverage (90% CI) | MAE (eV) |
| :--- | :--- | :--- |
| high | 0.885 | 0.565 |
| medium | 0.872 | 0.502 |
| low | 0.914 | 0.851 |

## Far-from-training rate
3.82% (target 3–7%).

## Limitations
1. Average global coverage hides per-adsorbate variation before per-adsorbate quantile calibration.
2. Conformal intervals assume exchangeability under chemical-system grouping, where out-of-group OOD domains can exhibit slight tail dispersion differences.
