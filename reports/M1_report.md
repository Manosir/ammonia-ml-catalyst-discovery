# Ammonia-AI M1: leak-free screening baseline (OC20 IS2RE; N*, N2*, NH*, NH2*, NH3*)

Draft of 2026-11-14. Contract `screening-v3-hb-unrelaxed-2026-11`. Models are untuned. Intervals are 95% bootstrap intervals over whole chemical systems and reflect test-set sampling only, not training variability.

## 1. Question and regime
Thermal Haber-Bosch screening with an unrelaxed slab plus adsorbate at a known site as input. Every feature comes from the initial frame. Last-frame, trajectory and energy quantities are audit-only and rejected by a registry gate and by name checks. The result is an adsorption-energy regressor, not a catalyst ranking: ranking needs a barrier layer (BEP or microkinetics) that OC20 does not provide.

## 2. Data and labels
- 45,558 OC20 IS2RE trajectories: N 12,123, NH 10,438, NH2 8,842, NH3 7,944, N2 6,211. The mapping pickle hash was verified, the join is one-to-one, and the folder adsorbate agrees with the mapping for every row.
- Label: E_ads = E_final - reference_energy (official OC20 reference, elemental N2/H2 basis).
- OC20 anomaly flags mark 11.7% of rows (N2 26.0%, NH3 27.9%). Regression is trained on the 40,208 clean rows. Anomaly classifiers and all-row metrics are reported separately.
- NH2 occurs only in the two OC20 "oos_ads" splits.

## 3. Features (64, element-free)
Adsorbate descriptors (numbers of N and H atoms, 2); Miller encodings (2); tabulated element properties (electronegativity, group, period) aggregated as mean/min/max/range over the bulk, the surface layer and the site atoms (36); transition-metal and non-metal fractions and element counts over the same three scopes (9); initial-frame geometry (7), topology (3) and coordination (5). Nine features are placement-sensitive. Element one-hot fractions were excluded: in the 12-element leave-element-out ablation they reached only 23% below the floor, against 42% for the property features.

## 4. Protocol
HistGradientBoosting, 200 iterations. Floor = (adsorbate, class) training mean. "Full" uses all 64 features, "light" the 55 that are not placement-sensitive. Evaluations: chemical-system grouped CV (3 repeats x 5 folds), the official OC20 split, true leave-element-out (54 elements; every bulk containing the element removed from training), NH2 held out of training.

## 5. Results (clean rows, MAE in eV)
| Adsorbate | Floor | Full [95% CI] | Light - full [CI] |
|---|---|---|---|
| All | 0.984 | 0.578 [0.564, 0.593] | +0.008 [+0.005, +0.010] |
| N | 1.386 | 0.694 [0.672, 0.718] | +0.013 [+0.008, +0.018] |
| N2 | 0.798 | 0.630 [0.605, 0.654] | +0.014 [+0.008, +0.021] |
| NH | 1.109 | 0.589 [0.571, 0.608] | +0.005 [+0.001, +0.009] |
| NH2 | 0.783 | 0.532 [0.513, 0.551] | +0.006 [+0.002, +0.010] |
| NH3 | 0.388 | 0.351 [0.335, 0.366] | -0.001 [-0.004, +0.003] |

- Class 0 (intermetallic) only: 0.464 [0.451, 0.479] against a floor of 0.927. By OC20 class (full): class 0 0.464, class 1 0.570, class 2 0.787, class 3 1.017 [0.907, 1.129].
- Spearman within adsorbate, class 0: N 0.89, NH 0.88, NH2 0.78, N2 0.68, NH3 0.56 (pooled over all surfaces and sites; not a catalyst ranking).
- Official split: val_is 0.610 [0.573, 0.654] (floor 1.062); val_oos_bulk, which shares no bulk with training, 0.582 [0.529, 0.650] (floor 1.033). NH2 is not in these splits.
- Leave-element-out (54 elements): mean 0.636 [0.591, 0.688] against a floor of 1.041; class 0 0.507 [0.465, 0.567] against 0.947; row-weighted 0.618 against 0.996. Class-0 values for Ru, Fe, Co, Os, Re, Mo: 0.383, 0.454, 0.433, 0.417, 0.424, 0.419 (floors 0.623, 0.790, 0.727, 0.608, 0.703, 0.746); these cover bulks containing the element, mostly alloys and compounds. The tested elements that fail are Sr (0.839 against 0.756) and Na (1.163 against 1.059); N* ranking is poor on bulks containing N, Cl, C, H or S.
- NH2 unseen (training without NH2): val_oos_ads 0.783 [0.756, 0.812] against a constant of 0.835, Spearman 0.734 [0.712, 0.754]; val_oos_ads_bulk 0.736 [0.700, 0.772] against 0.804, Spearman 0.696. Predictions are overspread (slope of truth on prediction 0.61-0.67).
- Anomaly classifiers (within-adsorbate AUC): desorbed N2 0.907, NH3 0.860; dissociated NH2 0.915, NH3 0.905, NH 0.920; surface-changed 0.84-0.89 for all adsorbates but with average precision of only 0.15-0.25. Dissociated N2 (33 rows) is not detectable (AUC 0.736 [0.635, 0.829], average precision 0.018).
- The clean-trained regressor on anomalous rows: better than the floor for dissociated (1.068 against 1.289) and desorbed (0.464 against 0.598) rows, worse than the floor for N2 surface changes (1.312 against 1.209). Its output should be trusted only for rows the classifier calls clean.
- Placement robustness: replacing the placement-sensitive features by medians gives 0.610 (light: 0.586). With random offsets of sigma = 0.25 A the full model scores 0.587 and at 0.5 A 0.612. Quote the light numbers (0.586; class 0 0.472) for candidates not placed by the OC20 recipe.

## 6. Verification
- Raw data, labels and joins audited (hash, coverage, adsorbate agreement); tests cover the schema, registry gate, readiness, candidate features and evaluation (including a test that audit columns in the table never reach the model).
- A recomputation of all 60 candidate columns with a different code path agreed on 28 systems (worst difference 7e-5, the 4-decimal storage precision); the typed element table agrees with pymatgen for 64 elements.

## 7. Limitations
- OC20 gives low-coverage RPBE adsorption energies: no barriers, coadsorbate effects or free-energy corrections. N2 dissociation, the key Haber-Bosch step, is essentially absent as a relaxation outcome (33 rows) and needs a BEP or microkinetic layer.
- Vacancies and doped surfaces are not in OC20; the data are perfect bulk truncations of 9,301 bulks. Coordination and site-element statistics are the only handle on under-coordination.
- A new candidate needs a site and a placement; OC20's placement recipe influences nine features.
- Label noise: 122 rows lie below -5 eV; slab restructuring is associated with the extremes (median maximum displacement 3.5 A for "surface changed" against 0.5 A otherwise), but displacement is not used as a filter.
- Untuned models; one run per element in leave-element-out; the 54 elements' test sets overlap.
- No retrodiction or volcano analysis yet.

## 8. Stage 1 (superseded for screening)
The earlier N*-only model (README: MAE 0.771 eV, R2 0.583) used final-frame features, random splits and a label-filtered 4,849-row subset. A reproduction with composition-grouped CV gives 0.786 with all features, 0.814 without the three final-frame ones and 0.823 with composition alone. Its site d-band feature was built without a cell, so it averaged over periodic images. These numbers are not comparable to Section 5.

## 9. Open items
run_all script and saved final model; moving legacy code; README note; a real re-placement test of the placement sensitivity; volcano ranking and retrodiction (M2).
