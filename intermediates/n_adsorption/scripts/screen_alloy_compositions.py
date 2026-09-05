"""Screen composition hypotheses with the validated 16-feature N* model.

This is composition-space screening under fixed representative geometry. It is
not a structure-specific prediction and must be followed by DFT validation.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_TM = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zr","Nb","Mo","Ru","Rh","Pd","Ag","Hf","Ta","W","Re","Os","Ir","Pt"}
D_BAND_CENTER = {"Sc":-.50,"Ti":-.60,"V":-1.09,"Cr":-1.32,"Mn":-1.39,"Fe":-1.29,"Co":-1.17,"Ni":-1.29,"Cu":-2.67,"Zr":-.40,"Nb":-1.41,"Mo":-1.60,"Ru":-1.41,"Rh":-1.73,"Pd":-1.83,"Ag":-4.30,"Hf":-1.10,"Ta":-1.59,"W":-1.80,"Re":-1.60,"Os":-1.40,"Ir":-1.56,"Pt":-2.25}
METALLIC_RADII = {"Sc":1.62,"Ti":1.45,"V":1.34,"Cr":1.28,"Mn":1.29,"Fe":1.26,"Co":1.25,"Ni":1.24,"Cu":1.28,"Zr":1.60,"Nb":1.46,"Mo":1.39,"Ru":1.34,"Rh":1.34,"Pd":1.37,"Ag":1.44,"Hf":1.59,"Ta":1.46,"W":1.41,"Re":1.37,"Os":1.35,"Ir":1.36,"Pt":1.39}
ELECTRONEGATIVITY = {"Sc":1.36,"Ti":1.54,"V":1.63,"Cr":1.66,"Mn":1.55,"Fe":1.83,"Co":1.88,"Ni":1.91,"Cu":1.90,"Zr":1.33,"Nb":1.60,"Mo":2.16,"Ru":2.20,"Rh":2.28,"Pd":2.20,"Ag":1.93,"Hf":1.30,"Ta":1.50,"W":2.36,"Re":1.90,"Os":2.20,"Ir":2.20,"Pt":2.28}
D_ELECTRONS = {"Sc":1,"Ti":2,"V":3,"Cr":5,"Mn":5,"Fe":6,"Co":7,"Ni":8,"Cu":10,"Zr":2,"Nb":4,"Mo":5,"Ru":7,"Rh":8,"Pd":10,"Ag":10,"Hf":2,"Ta":3,"W":4,"Re":5,"Os":6,"Ir":7,"Pt":9}
FEATURE_COLS = ["d_band_center_slab","d_band_center_site","d_band_center_primary","metallic_radius","electronegativity","d_electrons","primary_metal_frac","tm_frac","n_distinct_TM","gcn_initial","ads_height","mean_ads_force","facet_roughness","shift","n_N_ads","n_surface_atoms"]
DEFAULT_GEOM = {"gcn_initial":6.8,"ads_height":1.95,"mean_ads_force":0.12,"facet_roughness":3,"shift":0.0,"n_N_ads":1,"n_surface_atoms":24}
E_OPT = -0.4

def compute_features(comp: dict[str,float]) -> dict[str,float]:
    if not comp or not all(e in TARGET_TM for e in comp): raise ValueError("Unsupported element")
    total = sum(comp.values())
    if total <= 0 or abs(total-1) > 1e-6: raise ValueError("Fractions must be positive and sum to 1")
    primary = max(comp, key=comp.get)
    db = sum(comp[e]*D_BAND_CENTER[e] for e in comp)
    return {"d_band_center_slab":db,"d_band_center_site":db,"d_band_center_primary":D_BAND_CENTER[primary],"metallic_radius":METALLIC_RADII[primary],"electronegativity":ELECTRONEGATIVITY[primary],"d_electrons":D_ELECTRONS[primary],"primary_metal_frac":comp[primary],"tm_frac":1.0,"n_distinct_TM":len(comp),**DEFAULT_GEOM}

def generate(compound_elements):
    n = random.randint(1, min(4,len(compound_elements)))
    elems = random.sample(sorted(compound_elements), n)
    fracs = np.random.dirichlet(np.ones(n))
    return dict(zip(elems, fracs))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--n",type=int,default=10000)
    p.add_argument("--top",type=int,default=50)
    p.add_argument("--output",type=Path,default=Path("results/screened_alloys_16feat.csv"))
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    import joblib
    model=joblib.load(args.model)
    n_model=getattr(model,"n_features_in_",None)
    if n_model != len(FEATURE_COLS): raise SystemExit(f"Model expects {n_model} features; this screener supplies {len(FEATURE_COLS)}")
    rows=[]
    for _ in range(args.n):
        comp=generate(TARGET_TM); feats=compute_features(comp)
        rows.append({"composition":"-".join(f"{e}{comp[e]:.4f}" for e in sorted(comp)),"n_components":len(comp),"primary":max(comp,key=comp.get),**feats})
    df=pd.DataFrame(rows); df["delta_E_pred"]=model.predict(df[FEATURE_COLS]); df["sabatier_distance"]=abs(df["delta_E_pred"]-E_OPT); df["geometry_assumption"]="representative (111) hollow-site values"
    out=df.sort_values("sabatier_distance").reset_index(drop=True); args.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False); out.head(args.top).to_csv(args.output.parent/"top_candidates_16feat.csv",index=False)
    print(out.head(args.top)[["composition","primary","delta_E_pred","sabatier_distance","n_components"]].to_string(index=False)); print(f"Saved {len(out)} candidates to {args.output}")
if __name__ == "__main__": main()
