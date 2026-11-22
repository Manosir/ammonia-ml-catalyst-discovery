# Ammonia-AI: Auditable ML-Assisted Catalyst Discovery

[![CI](https://github.com/Manosir/ammonia-ml-catalyst-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/Manosir/ammonia-ml-catalyst-discovery/actions/workflows/ci.yml)

Ammonia-AI is a reproducible materials-informatics workflow for exploring transition-metal surfaces for ammonia synthesis. The current release focuses on predicting nitrogen-atom (`N*`) adsorption-energy differences from Open Catalyst 2020 (OC20) IS2RE trajectories using physically motivated electronic, geometric, compositional, and surface descriptors.

The repository is designed as a foundation for a future autonomous scientific-agent system. Planned extensions include uncertainty-aware candidate selection, multi-objective optimization, experiment memory, automated validation, and human-approved first-principles calculations.

> **Current scope:** This release is a descriptor-based `N*` adsorption-energy surrogate. It is not yet a complete microkinetic model, an experimentally validated catalyst predictor, or an autonomous DFT agent.

## Status (2026-11-21): screening-v3 baseline

The repository now also contains a leak-free multi-adsorbate baseline (N*, N2*, NH*, NH2*, NH3*; thermal Haber-Bosch; unrelaxed slab + adsorbate at a known site), trained and evaluated on all 45,558 OC20 IS2RE trajectories with 64 element-free initial-frame features.

**Headline (clean rows, chemical-system grouped CV, untuned):** MAE 0.578 eV [0.564, 0.593] (5-fold, 200 iterations) against a floor of 0.984 eV; 0.464 eV on intermetallic (class 0) surfaces. Leave-element-out over 54 elements: 0.636 eV against a floor of 1.041 eV.

**Uncertainty baseline:** MAE 0.565 eV (high reliability), 0.502 eV (medium), 0.851 eV (low). Protocol: 5-fold, 200 iterations, full protocol (fast-mode draft: 3-fold, 100 iterations).

**Limitations:**
1. Average global coverage hides per-adsorbate variation before per-adsorbate quantile calibration.
2. Conformal intervals assume exchangeability under chemical-system grouping, where out-of-group OOD domains can exhibit slight tail dispersion differences.

Not yet done: a catalyst ranking (needs a barrier or microkinetic layer), a saved final model, and a re-placement test of the placement-sensitive features. Run order: `scripts/run_all.sh`.

## Repository structure

The active source of truth is the `ammonia_ai` package.

```text
ammonia-ml-catalyst-discovery/
├── README.md
├── src/ammonia_ai/
└── scripts/
```

(Full structure omitted for brevity; see original README for details.)

## Installation

(Installation instructions remain the same.)
