# Ammonia-AI Target Comparability Contract (v0.2)

This document defines the contract for comparing adsorption energies across different data sources, levels of theory, and computational setups.

## 1. Baseline Reference: OC20 Standard
All predictions from the `ammonia-ai` screening-v3 model are calibrated to the **OC20 IS2RE** reference state:
- **Functional:** RPBE (Revised Perdew-Burke-Ernzerhof)
- **Pseudopotentials:** PAW (Projector Augmented Wave)
- **Cutoff:** 350 eV
- **Spin:** Non-spin-polarized (Fixed for all systems in OC20)
- **Reference State:** Elemental N2 and H2 (gas phase) at 0K.

## 2. Client Levels of Theory (Commercial Targets)
Commercial triage typically involves comparisons against client-specific calculations, which often differ from the OC20 standard:
- **Functional:** PBE or PBEsol (standard for many industrial catalogs).
- **Spin:** Spin-polarized (necessary for magnetic transition metals: **Fe, Co, Ni**).
- **Cutoff:** Often higher (400-600 eV) for precision.

## 3. Comparability Rules and Offsets

Direct mixing of data from different sources is prohibited without calibration. The following flags identify where offsets are expected:

### A. Functional Offsets (e.g., RPBE vs. PBE)
RPBE generally binds adsorbates more weakly (higher energy) than PBE. 
- **Rule:** When comparing an OC20-trained prediction to a PBE-calculated value, a systematic shift must be applied.
- **Calibration:** Use common reference systems (e.g., N* on Ru(0001)) to determine the `delta_functional` offset.

### B. Magnetic/Spin Correction (The "Magnetic Flag")
OC20 is non-spin-polarized. For ferromagnetic elements (**Fe, Co, Ni**), non-spin-polarized calculations significantly overbind compared to spin-polarized ones.
- **Rule:** Predictions for surfaces containing **Fe, Co, or Ni** are flagged as `NEEDS_SPIN_CALIBRATION`.
- **Correction:** A per-element spin-polarization correction (derived from d-band shifts or empirical offsets) should be applied before triage ranking.

### C. Reference State Alignment
Ensure all energies use the same gas-phase reference state (1/2 N2 and 3/2 H2 for NH3*). 
- **Rule:** Catalysis-Hub or DTU RPA data must be shifted to the elemental reference basis if they use different conventions (e.g., atomic references).

## 4. Applicability Domain (AD) Definitions

- `IN_DOMAIN`: System contains elements and coordination environments well-represented in OC20 (Transition metals Sc-Pt).
- `OUT_OF_DOMAIN`: System contains elements not seen in training (e.g., f-block, alkali metals) or coordination numbers significantly outside the training range (e.g., isolated atoms on oxides).
- `NEEDS_SPIN_CALIBRATION`: System contains **Fe, Co, or Ni**; OC20-trained model will likely overbind.
