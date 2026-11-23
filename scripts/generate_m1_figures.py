import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def main():
    # Load A5 results
    res_path = Path("results/a5_final.json")
    if not res_path.exists():
        print("Error: results/a5_final.json not found")
        return

    with open(res_path) as f:
        res = json.load(f)

    # Fig 1: Spearman Heatmap (Width vs |Error|) per adsorbate
    plt.figure(figsize=(8, 6))
    ads_rhos = {ads: info['rho'] for ads, info in res['spearman_by_adsorbate'].items()}
    rho_df = pd.DataFrame(list(ads_rhos.items()), columns=['Adsorbate', 'Spearman Rho'])
    sns.barplot(data=rho_df, x='Adsorbate', y='Spearman Rho', palette='viridis')
    plt.title('Uncertainty Sharpness: Spearman correlation (Width vs |Error|)')
    plt.ylim(0, 0.4)
    plt.savefig('docs/figures/spearman_heatmap.png')
    plt.close()

    # Fig 2: Ru Facet Comparison (using the coverage table)
    ru_path = Path("results/m3/ru_n_coverage.csv")
    if ru_path.exists():
        ru_df = pd.read_csv(ru_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=ru_df, x='coverage_ML', y='E_ads_eV', hue='facet', style='supercell', s=100)
        plt.title('Ru(pure) N Adsorption Energy vs Coverage')
        plt.xlabel('Coverage (ML)')
        plt.ylabel('E_ads (eV)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('docs/figures/ru_facet_comparison.png')
        plt.close()

    # Fig 3: Coverage per Tier/Adsorbate (Bar chart)
    plt.figure(figsize=(10, 6))
    tier_cov = res['coverage_by_tier']
    tiers = ['high', 'medium', 'low']
    vals = [tier_cov.get(t, 0) for t in tiers]
    sns.barplot(x=tiers, y=vals, palette='rocket')
    plt.axhline(0.9, color='red', linestyle='--', label='Target 90%')
    plt.title('90% Prediction Interval Coverage per Reliability Tier')
    plt.ylabel('Actual Coverage')
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.savefig('docs/figures/coverage_tiers.png')
    plt.close()

    print("Generated 3 figures in docs/figures/")

if __name__ == "__main__":
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    main()