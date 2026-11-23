"""Pure-metal leave-one-metal-out N*: most-stable-site label vs min prediction."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HIGHLIGHT = ["Fe", "Ru", "Os", "Co", "Mo", "Rh"]
LO, HI = -4.0, 5.0          # label-outlier window (policy: flag, report with and without)
L, P = "adsorption_energy_eV", "predicted_E_ads_eV"

def per_metal(d):
    rows = []
    for m, g in d.groupby("bulk_chemsys"):
        il, ip = g[L].idxmin(), g[P].idxmin()
        rows.append(dict(metal=m, n=len(g), label=g.loc[il, L], label_facet=g.loc[il, "miller_index"],
                         pred=g.loc[ip, P], pred_facet=g.loc[ip, "miller_index"]))
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

df = pd.read_csv("results/m3/pure_metal_oof.csv")
df = df[(df["adsorbate"] == "N") & (df["is_clean_primary_regression_row"] == True)]
raw = per_metal(df)
T = per_metal(df[(df[L] >= LO) & (df[L] <= HI)])
ALL = T.copy()
TM = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag",
      "Hf","Ta","W","Re","Os","Ir","Pt","Au"}
T = T[T.metal.isin(TM)].reset_index(drop=True)

print(f"N rows: {len(df)}, metals: {df.bulk_chemsys.nunique()}")
print("\n== WITH label outliers ==");    print(raw.round(3).to_string(index=False))
print("\n== WITHOUT outliers (figure) =="); print(T.round(3).to_string(index=False))

s = pd.read_csv("results/m3/most_stable_sites.csv")
s = s[s["adsorbate"] == "N"].groupby("bulk_chemsys").agg(sites_label=("min_dft_E_ads_eV", "min"),
                                                         sites_pred=("min_pred_E_ads_eV", "min"))
print("\n== cross-check vs most_stable_sites.csv (raw) ==")
print(raw.set_index("metal")[["label", "pred"]].join(s).round(3).to_string())

print(f"all {len(ALL)} elements: Spearman {spearmanr(ALL.label, ALL.pred)[0]:.3f}, MAE {(ALL.pred-ALL.label).abs().mean():.3f} eV")
rho = spearmanr(T["label"], T["pred"])[0]
mae = (T["pred"] - T["label"]).abs().mean()
print(f"\nSpearman {rho:.3f}, MAE {mae:.3f} eV, n={len(T)} metals")

lim = [min(T.label.min(), T.pred.min()) - 0.2, max(T.label.max(), T.pred.max()) + 0.2]
fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
ax.plot(lim, lim, "--", color="grey", lw=1)
hl = T.metal.isin(HIGHLIGHT)
ax.scatter(T.label[~hl], T.pred[~hl], s=40, color="#9aa5b1")
ax.scatter(T.label[hl], T.pred[hl], s=60, color="#c2410c")
for _, r in T.iterrows():
    ax.annotate(r.metal, (r.label, r.pred), xytext=(4, 4), textcoords="offset points", fontsize=9,
                fontweight="bold" if r.metal in HIGHLIGHT else "normal")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Most-stable-site label E_ads (eV)"); ax.set_ylabel("Minimum predicted E_ads (eV)")
ax.set_title(f"N*, d-block pure metals (groups 3-11), leave-one-metal-out\nn={len(T)} metals, Spearman {rho:.2f}, MAE {mae:.2f} eV, "
             f"labels outside [{LO:g}, {HI:g}] eV removed", fontsize=9)
fig.tight_layout()
Path("docs/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("docs/figures/pure_metal_lomo_N.png")
print("saved docs/figures/pure_metal_lomo_N.png")

G610 = {"Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pd","Pt"}
H = T[T.metal.isin(G610)]
print(f"groups 6-10 ({len(H)} metals): Spearman {spearmanr(H.label, H.pred)[0]:.3f}, "
      f"MAE {(H.pred - H.label).abs().mean():.3f} eV")
