# Ammonia-AI: High-Throughput Catalyst Screening Pipeline

Ammonia-AI is a machine learning pipeline that accelerates the discovery of heterogeneous catalysts for ammonia synthesis by predicting nitrogen adsorption energies (ΔE_ads) from metallic slab configurations obtained from the Open Catalyst 2020 (OC20) dataset. The pipeline evaluates thermodynamic efficiency using the Sabatier volcano approach based on the Hammer–Nørskov d-band model.

The current version (v1.0) uses a Gradient Boosting Regressor (GBR) with physics-informed descriptors, achieving competitive accuracy while maintaining a strict zero-leakage design.

---

## Pipeline Architecture

![Pipeline Architecture](figures/pipeline_architecture.png)

---

## Results

### Sabatier Volcano

![Sabatier Volcano](figures/sabatier_volcano.png)

Ru (ΔE = −0.22 eV) and Co (ΔE = −0.46 eV) sit closest to the Sabatier optimum (ΔE_opt ≈ −0.4 eV), consistent with experimental catalyst performance and the Hammer–Nørskov ordering.

### Parity Plot

![Parity Plot](figures/parity_plot.png)

### Feature Importances

![Feature Importances](figures/feature_importance.png)

Top predictive features: initial-frame GCN (~20%), local d-band centre (~16%), composition-weighted d-band centre (~12%).

---

## Model Performance (v1.0)

Evaluated on n = 5,047 metallic systems from the OC20 IS2RE *N dataset:

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | **0.778 eV** |
| Cross-validated R² | **0.557 ± 0.014** |
| Training systems | 5,047 metallic *N slabs |
| Dataset | OC20 IS2RE per-adsorbate *N (index 77) |

The low CV standard deviation (±0.014) confirms the model measures a reproducible physical signal rather than fitting noise.

---

## Key Features

- **Physics-Informed Descriptors** — 16 features including generalised coordination numbers (GCN), d-band centre theory-based metrics, and surface composition descriptors.
- **Zero-Leakage Design** — Pure GCN arithmetic is separated from ASE file I/O. Only unrelaxed initial frames (frame index 0) are used for inference, avoiding relaxation-induced leakage of the target variable into the features.
- **Automated Thermodynamics** — Proxy activity is computed from GBR predictions and plotted on a Sabatier volcano to identify optimal catalyst compositions.
- **CI-Ready Test Suite** — Two-layer pytest suite. Unit tests import `_gcn_from_neighbour_lists` directly from `src/gcn_utils.py` and pass numpy arrays; integration tests write temporary extxyz.xz files and call `calculate_site_gcn()` end-to-end. No large binary fixtures required.

---

## Installation

**1. Clone the repository**
```bash
git clone [https://gitlab.com/nosir.phy/ammonia-ml-catalyst-discovery.git](https://gitlab.com/nosir.phy/ammonia-ml-catalyst-discovery.git)
cd ammonia-ml-catalyst-discovery