import joblib
import pandas as pd

# Load your trained model
model = joblib.load('results/is2re/gbr_is2re_n.joblib')

# Create a DataFrame with the exact columns your model expects
new_catalyst = pd.DataFrame([{
    'd_band_center_weighted': -1.428,
    'd_band_center_primary': -1.60,
    'metallic_radius': 1.39,
    'electronegativity': 2.16,
    'd_electrons': 5,
    'n_N_ads': 1,
    'n_surface_atoms': 36, # estimate based on your typical slabs
    'n_distinct_TM': 2,
    'primary_metal_frac': 0.6,
    'tm_frac': 1.0,
    'ads_height': 1.5,     # typical N-metal bond distance
    'mean_ads_force': 0.0  # assume relaxed state
}])

# Predict!
predicted_energy = model.predict(new_catalyst)
print(f"Predicted Adsorption Energy: {predicted_energy[0]:.2f} eV")