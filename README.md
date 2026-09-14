# Ammonia-AI: Auditable ML-Assisted Catalyst Discovery

[![CI](https://github.com/Manosir/ammonia-ml-catalyst-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/Manosir/ammonia-ml-catalyst-discovery/actions/workflows/ci.yml)

Ammonia-AI is a reproducible materials-informatics workflow for exploring transition-metal surfaces for ammonia synthesis. The current release focuses on predicting nitrogen-atom (`N*`) adsorption-energy differences from Open Catalyst 2020 (OC20) IS2RE trajectories using electronic, geometric, compositional, and surface descriptors.

The repository is designed as a foundation for a future autonomous scientific-agent system. Planned extensions include uncertainty-aware candidate selection, multi-objective optimization, experiment memory, automated validation, and human-approved first-principles calculations.

> **Current scope:** This release is a descriptor-based `N*` adsorption-energy surrogate. It is not yet a complete microkinetic model, an experimentally validated catalyst predictor, or an autonomous DFT agent.

## Current result

The current model-ready dataset contains **4,849 OC20 `N*` systems** after descriptor, energy, adsorbate, metallic-slab, and metadata filtering.

| Metric | Value |
|---|---:|
| Model-ready systems | 4,849 |
| Test MAE | 0.771 eV |
| Test R² | 0.583 |
| Five-fold random-CV R² | 0.589 ± 0.015 |
| Number of model features | 16 |
| Model | `StandardScaler` + `GradientBoostingRegressor` |

These metrics are preliminary. Random validation should later be supplemented by leave-one-metal-out, leave-one-facet-out, composition-family holdout, and genuinely external validation before making claims about transferability.

## Results overview

### Feature importance

![Feature importance](intermediates/n_adsorption/figures/feature_importance.png)

The feature-importance plot shows the relative contribution of the input descriptors to this fitted model. These values are model-specific associations, not causal physical percentages. In particular, the importance of `d_band_center_site` should not be interpreted as proof that this descriptor independently controls adsorption energy.

### Parity plot

![Parity plot](intermediates/n_adsorption/figures/parity_plot.png)

The parity plot compares predicted and reference adsorption-energy differences for the evaluated model-ready systems.

### Pipeline architecture

![Pipeline architecture](intermediates/n_adsorption/figures/pipeline_architecture.png)

The architecture diagram summarizes the separation between data acquisition, descriptor preprocessing, model training, diagnostics, prediction, and candidate screening.

### Sabatier-style volcano

![Sabatier volcano](intermediates/n_adsorption/figures/sabatier_volcano.png)

The volcano plot provides a screening-oriented view of predicted adsorption-energy trends. It is not a complete ammonia-synthesis activity model and should not be interpreted as a direct prediction of reaction rate.

## Scientific scope and limitations

The current target is the adsorption-energy difference:

```text
ΔE_ads = E_relaxed − E_reference
```

The reference and relaxed energies are read from the OC20 IS2RE trajectory data processed by the training workflow.

The model combines slab-level and site-local d-band descriptors, electronic and compositional descriptors, initial-frame generalized coordination number, relaxed-frame geometry and force descriptors, and OC20 surface metadata.

The model is best described as a **post-relaxation `N*` adsorption-energy surrogate** because `d_band_center_site`, `ads_height`, and `mean_ads_force` are derived from relaxed structures. The initial-frame GCN is computed before relaxation, but the complete 16-feature model is not yet a fully pre-DFT predictor.

Predictions should therefore be interpreted as computational screening hypotheses. The model does not establish catalyst stability, alloy segregation behavior, poisoning resistance, activity under operating conditions, selectivity, toxicity, lifetime, or complete ammonia production rates. Candidate predictions require structure-level validation, preferably with first-principles calculations and ultimately experiment.

## Repository structure

The active source of truth is the `ammonia_ai` package. Workflow scripts are kept under `intermediates/n_adsorption/scripts/` because they orchestrate dataset-specific preprocessing, training, prediction, and supplementary analyses.

```text
ammonia-ml-catalyst-discovery/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── environment.yml
├── pyproject.toml
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   └── ammonia_ai/
│       ├── __init__.py
│       ├── schema.py
│       └── features/
│           ├── __init__.py
│           ├── d_band.py
│           └── gcn.py
├── tests/
│   ├── test_d_band.py
│   ├── test_gcn.py
│   └── test_schema.py
└── intermediates/
    └── n_adsorption/
        ├── figures/
        │   ├── feature_importance.png
        │   ├── parity_plot.png
        │   ├── pipeline_architecture.png
        │   └── sabatier_volcano.png
        └── scripts/
            ├── download_is2re_n.py
            ├── download_oc20_is2re_n.py
            ├── precompute_initial_gcn.py
            ├── precompute_site_d_band.py
            ├── build_train_n_adsorption_model.py
            ├── predict_candidate_adsorption.py
            ├── screen_alloy_compositions.py
            ├── benchmark_literature_averaged.py
            ├── benchmark_oc20_literature.py
            └── diagnostics/
                └── evaluate_gcn_frame_leakage.py
```

Raw OC20 data, descriptor tables, trained models, caches, virtual environments, and temporary archives are generated locally and are not required for the public source repository.

## Installation

The package requires **Python 3.11 or newer**, as declared in `pyproject.toml`.

```bash
git clone https://github.com/Manosir/ammonia-ml-catalyst-discovery.git
cd ammonia-ml-catalyst-discovery
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

The editable installation installs the `ammonia_ai` package from `src/` and the test dependency. Runtime dependencies are declared in `pyproject.toml`.

An alternative installation is:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

The `pyproject.toml` installation is preferred because it installs the package and keeps the test configuration in one place.

## Minimal verification

Before downloading OC20 data, verify the package and test suite:

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q
python -m compileall -q src tests intermediates
```

The expected local test result is:

```text
30 passed
```

Verify the canonical package imports and feature count:

```bash
python - <<'PY'
from ammonia_ai import FEATURE_COLUMNS, TARGET_COLUMN

assert len(FEATURE_COLUMNS) == 16
assert TARGET_COLUMN == "delta_E_ads"
print("Canonical feature contract: OK")
PY
```

Check the workflow command-line interfaces without running data-intensive operations:

```bash
for script in intermediates/n_adsorption/scripts/*.py; do
  python "$script" --help >/dev/null || exit 1
done
```

## Canonical workflow

```text
OC20 IS2RE N* data
        │
        ├── acquire N* trajectories
        ├── validate OC20 surface metadata
        ├── compute initial-frame GCN
        ├── compute final-frame site-local d-band descriptor
        ├── build and train the 16-feature model
        ├── inspect schema and model compatibility
        ├── predict candidate adsorption energies
        ├── screen representative alloy compositions
        └── run supplementary benchmarks and diagnostics
```

The workflow is separated into deterministic preprocessing, model training, prediction, and evaluation steps. This separation makes descriptor provenance and potential train/test leakage easier to inspect.

## Data acquisition

The primary downloader is:

```text
intermediates/n_adsorption/scripts/download_is2re_n.py
```

Inspect its options before downloading:

```bash
python intermediates/n_adsorption/scripts/download_is2re_n.py --help
```

Run the downloader according to the supported local options:

```bash
python intermediates/n_adsorption/scripts/download_is2re_n.py --out-dir data/is2re_N
```

A second collector is available for an alternative OC20 IS2RE N* acquisition path:

```text
intermediates/n_adsorption/scripts/download_oc20_is2re_n.py
```

Do not run both downloaders blindly. Select one documented acquisition path, record the source, date, checksum, and resulting directory layout, and use the same layout for subsequent preprocessing.

Confirm that trajectory files exist:

```bash
find data/is2re_N -name '*.extxyz.xz' | head
find data/is2re_N -name 'system.txt' -print
```

The examples below assume that trajectories are located at:

```text
data/is2re_N/77/77/
```

If your local download produces `data/is2re_N/77/` instead, use that path consistently in every command.

## OC20 metadata

The training workflow requires an OC20 metadata table when metadata coverage is being validated. The expected local file is:

```text
data/oc20_n_metadata.csv
```

The table must provide complete identifiers and surface information for the systems used by training, including `system_id`, `miller_index`, and `shift` where required by the training script.

The public repository does not redistribute OC20 data. Generate or obtain the metadata table through your local OC20 data-preparation workflow, and record the source and checksum in `data/README.md`. Do not use a checksum bypass to hide a mapping-file mismatch; verify that local trajectory identifiers overlap with the mapping before training.

## Descriptor preprocessing

Compute the initial-frame GCN descriptor:

```bash
python intermediates/n_adsorption/scripts/precompute_initial_gcn.py \
  --data-dir data/is2re_N/77/77 \
  --out results/gcn_initial_frame.csv \
  --workers 4
```

Compute the site-local d-band descriptor from the final trajectory frame:

```bash
python intermediates/n_adsorption/scripts/precompute_site_d_band.py \
  --data-dir data/is2re_N/77/77 \
  --out results/site_dband.csv \
  --workers 4
```

Both descriptor tables use `system_id` as their identifier. Missing descriptor values are preserved during preprocessing and are subsequently handled by the model-building workflow’s coverage checks. Do not replace missing scientific descriptors with arbitrary constants without documenting and validating the resulting bias.

## Training and inspection

Run an inspection pass first. This validates the model-ready dataset without fitting the final model:

```bash
python intermediates/n_adsorption/scripts/build_train_n_adsorption_model.py \
  --data-dir data/is2re_N/77/77 \
  --gcn-csv results/gcn_initial_frame.csv \
  --site-dband-csv results/site_dband.csv \
  --out results/n_adsorption \
  --workers 4 \
  --inspect-only
```

Review the record count, missing descriptor counts, metadata coverage, target distribution, and per-metal statistics. Train only after confirming that the input tables and metadata correspond to the same system identifiers.

Run model training:

```bash
python intermediates/n_adsorption/scripts/build_train_n_adsorption_model.py \
  --data-dir data/is2re_N/77/77 \
  --gcn-csv results/gcn_initial_frame.csv \
  --site-dband-csv results/site_dband.csv \
  --out results/n_adsorption \
  --workers 4
```

Expected generated outputs include:

```text
results/n_adsorption/is2re_full_dataset.csv
results/n_adsorption/is2re_final_dataset.csv
results/n_adsorption/gbr_ammonia_ai_final.joblib
results/n_adsorption/model_manifest.json
results/n_adsorption/parity.png
results/n_adsorption/feature_importance.png
results/n_adsorption/sabatier_volcano.png
```

The model manifest should be retained with any model artifact distributed outside Git. It records the feature contract and reproducibility information needed to determine whether a model can safely be used with a candidate table.

## Canonical 16-feature contract

The model expects these features in exactly this order:

```text
d_band_center_slab
d_band_center_site
d_band_center_primary
metallic_radius
electronegativity
d_electrons
primary_metal_frac
tm_frac
n_distinct_TM
gcn_initial
ads_height
mean_ads_force
facet_roughness
shift
n_N_ads
n_surface_atoms
```

The authoritative definition is maintained in:

```text
src/ammonia_ai/schema.py
```

New code should import the contract rather than defining a second local feature list:

```python
from ammonia_ai.schema import FEATURE_COLUMNS, TARGET_COLUMN
```

The schema utilities support the transition between canonical `system_id` and legacy `sid` inputs. The canonical spelling for new files is `system_id`.

## Model diagnostics and prediction

Validate a trained model against its model-ready dataset:

```bash
python intermediates/n_adsorption/scripts/diagnostic.py \
  --model results/n_adsorption/gbr_ammonia_ai_final.joblib \
  --dataset results/n_adsorption/is2re_final_dataset.csv
```

Predict a representative candidate using the built-in catalogue:

```bash
python intermediates/n_adsorption/scripts/predict_candidate_adsorption.py \
  --model results/n_adsorption/gbr_ammonia_ai_final.joblib \
  --metal Fe
```

For a candidate table:

```bash
python intermediates/n_adsorption/scripts/predict_candidate_adsorption.py \
  --model results/n_adsorption/gbr_ammonia_ai_final.joblib \
  --input-csv candidates.csv
```

Each candidate row must contain a `label` and all 16 canonical feature columns. A valid CSV schema alone does not guarantee that a candidate is within the model’s domain of applicability.

## Alloy screening

The composition screener generates a representative exploratory ranking:

```bash
python intermediates/n_adsorption/scripts/screen_alloy_compositions.py \
  --model results/n_adsorption/gbr_ammonia_ai_final.joblib \
  --n 10000 \
  --top 50 \
  --output results/screened_alloys_16feat.csv
```

The current screener uses fixed representative geometry assumptions. It does not predict alloy stability, segregation, surface reconstruction, poisoning, toxicity, catalyst lifetime, or complete reaction rates. Its output should be treated as a candidate-generation list for subsequent structure-level validation.

## Supplementary benchmarks and leakage diagnostics

The supplementary benchmark scripts are not independent blind tests unless the compared systems were excluded from model training.

Run the averaged-feature sensitivity analysis:

```bash
python intermediates/n_adsorption/scripts/benchmark_literature_averaged.py \
  --model results/n_adsorption/gbr_ammonia_ai_final.joblib \
  --dataset results/n_adsorption/is2re_final_dataset.csv
```

Run the OC20 literature comparison:

```bash
python intermediates/n_adsorption/scripts/benchmark_oc20_literature.py \
  --model results/n_adsorption/gbr_ammonia_ai_final.joblib \
  --dataset results/n_adsorption/is2re_final_dataset.csv
```

The optional historical GCN diagnostic is located at:

```text
intermediates/n_adsorption/scripts/diagnostics/evaluate_gcn_frame_leakage.py
```

It should be interpreted as a diagnostic comparison, not as the canonical 16-feature production workflow. Leakage-sensitive evaluation should explicitly separate descriptors derived from initial and relaxed frames and report the split strategy used.

## Autonomous-agent vision

The long-term goal is an auditable closed-loop scientific agent:

```text
scientific objective
        ↓
validated candidate generation
        ↓
admissible feature construction
        ↓
uncertainty-aware surrogate prediction
        ↓
Pareto optimization under cost and stability constraints
        ↓
human-approved DFT validation
        ↓
experiment memory and model update
```

The language-model agent should coordinate plans, interpret outputs, maintain experiment memory, and explain decisions. Deterministic Python tools should calculate descriptors, predictions, uncertainty estimates, constraints, and metrics.

A minimum viable agent should contain:

| Component | Responsibility |
|---|---|
| Planner | Convert a scientific objective into a structured experiment plan. |
| Dataset validator | Check paths, schemas, identifiers, checksums, and coverage. |
| Surrogate tool | Load or train a versioned model and return predictions. |
| Uncertainty tool | Estimate confidence and detect out-of-domain candidates. |
| Optimizer | Select a Pareto-ranked batch under explicit constraints. |
| Critic | Reject duplicates, missing features, invalid structures, leakage, and unsupported claims. |
| Memory | Store plans, inputs, outputs, metrics, failures, and decisions. |
| Reporter | Produce reproducible Markdown and JSON experiment reports. |

The first autonomous demonstration should select a small batch of candidates using performance, uncertainty, and composition-cost objectives, and require human approval before any expensive DFT execution.

## Testing and continuous integration

Run the local test suite with:

```bash
python -m pytest -q
```

The tests cover the canonical 16-feature contract, missing and non-numeric feature values, current and legacy identifier schemas, duplicate identifiers, metadata completeness, model feature-count detection, GCN frame selection, and site-local d-band behavior.

GitHub Actions uses:

```text
.github/workflows/ci.yml
```

The workflow installs packages, compiles Python, runs pytest, and checks canonical imports. It does not download OC20 data or train the full model because those are large-data experiments rather than lightweight continuous-integration checks.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## References

1. Chanussot et al., “The Open Catalyst 2020 Dataset and Community Challenges,” [arXiv:2010.09990](https://arxiv.org/abs/2010.09990).
2. Abild-Pedersen et al., “Scaling Properties of Adsorption Energies for Hydrogen-Containing Molecules on Transition-Metal Surfaces,” [Physical Review Letters](https://doi.org/10.1103/PhysRevLett.99.016105).
3. Hammer and Nørskov, “Theoretical Surface Science and Catalysis—Calculations and Concepts,” [Advances in Catalysis](https://doi.org/10.1016/S0360-0564(08)60514-7).

## Author

**Mohamed Nosir** — Computational Materials Scientist.

Project repository: [github.com/Manosir/ammonia-ml-catalyst-discovery](https://github.com/Manosir/ammonia-ml-catalyst-discovery)
