# Stage 1 (N*-only) pipeline: superseded for screening

Kept for provenance. Do not use it to rank candidates. Findings from the 2026-11 audit:

- **Final-frame features.** `d_band_center_site`, `ads_height` and `mean_ads_force` come from the relaxed structure and do not exist for an unrelaxed candidate. Removing the three costs about 0.03 eV (MAE 0.786 to 0.814, composition-grouped CV); `ads_height` and `mean_ads_force` add nothing.
- **Random splits** overstate transfer; grouped evaluation is in `reports/M1_report.md`.
- **Label-dependent filter.** 4,849 of 12,123 N* rows survive: rows are dropped by bounds on the target and a transition-metal fraction rule (about 58% removed in a 2,000-file sample).
- **Neighbour lists without a cell.** `compute_gcn` and `site_local_d_band` build `Atoms(..., pbc=True)` without a cell; ASE then uses a unit-cell box and the pair counts are 760-2,000x too large (203,042 against 100 on a 33-atom file). The stored `gcn_initial_frame.csv` appears to come from `precompute_initial_gcn.py`, which uses a 50 A dummy cell with `pbc=False`: no image inflation, but lateral periodic neighbours are ignored, so atoms near the cell boundary are under-coordinated (unquantified). `d_band_center_site` averaged over periodic images.
- **Unstable `primary_metal`.** It is the `max` over a set of element symbols; ties are resolved by hash-dependent order (about 16% of rows have a tied most-abundant transition metal), so the four primary-metal descriptors are partly arbitrary.
- **`ads_height`** has no periodic-image logic and is negative for 10.8% of rows.
- `predict_candidate_adsorption.py` uses hand-entered geometry values for the candidates (height 1.85-2.00, force 0.09-0.18), which carry no information about the real structures.
