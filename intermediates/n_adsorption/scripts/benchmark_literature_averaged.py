"""Supplementary pure-metal benchmark using actual OC20 feature rows."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
FEATURE_COLS=["d_band_center_slab","d_band_center_site","d_band_center_primary","metallic_radius","electronegativity","d_electrons","primary_metal_frac","tm_frac","n_distinct_TM","gcn_initial","ads_height","mean_ads_force","facet_roughness","shift","n_N_ads","n_surface_atoms"]
LITERATURE={"Sc":-1.30,"Ti":-1.10,"V":-0.80,"Cr":-0.90,"Mn":-0.75,"Fe":-0.86,"Co":-0.55,"Ni":-0.42,"Cu":-0.28,"Zr":-1.20,"Nb":-0.85,"Mo":-0.70,"Ru":-0.32,"Rh":-0.46,"Pd":-0.15,"Ag":-0.20,"Hf":-1.15,"Ta":-0.95,"W":-1.05,"Re":-0.95,"Os":-0.65,"Ir":-0.50,"Pt":-0.06}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('results/benchmark_averaged')); a=p.parse_args()
    import joblib
    model=joblib.load(a.model); n=getattr(model,'n_features_in_',None)
    if n != len(FEATURE_COLS): raise SystemExit(f'Model expects {n} features; benchmark requires {len(FEATURE_COLS)}')
    df=pd.read_csv(a.dataset); pure=df[(df.n_distinct_TM==1)&(df.primary_metal_frac>0.99)].copy(); rows=[]
    for metal, ref in LITERATURE.items():
        sub=pure[pure.primary_metal==metal].dropna(subset=FEATURE_COLS)
        if sub.empty: continue
        pred_each=model.predict(sub[FEATURE_COLS]); avg_features=sub[FEATURE_COLS].mean().to_frame().T; pred_at_mean=model.predict(avg_features)[0]
        rows.append({'metal':metal,'dft':ref,'mean_prediction':pred_each.mean(),'prediction_at_mean_features':pred_at_mean,'n_records':len(sub)})
    out=pd.DataFrame(rows)
    if out.empty: raise SystemExit('No complete pure-metal rows matched the literature table')
    out['abs_error_mean_prediction']=(out.mean_prediction-out.dft).abs(); out['abs_error_at_mean_features']=(out.prediction_at_mean_features-out.dft).abs(); a.out.mkdir(parents=True,exist_ok=True); out.to_csv(a.out/'benchmark_averaged.csv',index=False)
    print(f"n={len(out)} | MAE(mean predictions)={out.abs_error_mean_prediction.mean():.3f} eV | MAE(prediction at mean features)={out.abs_error_at_mean_features.mean():.3f} eV"); print(out.to_string(index=False))
if __name__=='__main__': main()
