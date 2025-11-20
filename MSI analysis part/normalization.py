import pandas as pd
import numpy as np
from pathlib import Path

# ==== CONFIG: your exact CSV path ====
INPUT_CSV  = Path(r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\DataMatrix\data_matrix.csv")
OUTPUT_CSV = INPUT_CSV.with_name(INPUT_CSV.stem + "_minmax.csv")

# ==== LOAD ====
df = pd.read_csv(INPUT_CSV)

# First 13 columns are wavelengths (features); 14th is 'label'
X = df.iloc[:, :13].astype(float).copy()
y = df.iloc[:, 13].copy()   # assumes this column is named 'label'

# ==== MIN–MAX NORMALIZATION to [0,1] (per column) ====
col_min = X.min(axis=0)
col_max = X.max(axis=0)
denom = (col_max - col_min).replace(0, 1.0)  # avoid division by zero for constant columns

X_norm = (X - col_min) / denom
# Optional: clip tiny numerical spillover
X_norm = X_norm.clip(lower=0.0, upper=1.0)

# ==== RECOMBINE & SAVE ====
df_out = pd.concat([X_norm, y.rename('label')], axis=1)
df_out.to_csv(OUTPUT_CSV, index=False)

# ==== (Optional) Quick sanity checks printed to console ====
print(f"Saved normalized matrix to: {OUTPUT_CSV}")
print("Per-column min (should be ~0):")
print(X_norm.min().round(6))
print("Per-column max (should be ~1):")
print(X_norm.max().round(6))
