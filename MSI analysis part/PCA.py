import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# ================== PATHS ==================
INPUT_CSV = Path(r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\DataMatrix\data_matrix_minmax.csv")
OUT_DIR   = Path(r"C:\Users\irana\OneDrive\Desktop\Chlorophyll MSI\PCA_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ================== SELECT WAVELENGTHS ==================
# Option A (recommended): select by COLUMN NAMES exactly as they appear in your CSV header
#   Example if your headers are plain numbers like 365, 405, ..., 940 they will load as strings by pandas.
SELECT_BY_NAME = True
WAVELENGTH_COLUMNS = ['365nm', '405nm', '473nm', '530nm', '575nm', '621nm']   # <-- change to your wavelength column names

# Option B: select by POSITIONS (0-based) within the FIRST 13 feature columns
#   e.g., [1, 7, 8] means use the 2nd, 8th, and 9th of the first 13 columns
WAVELENGTH_IDX = []  # leave empty if using names

# ================== LOAD ==================
df = pd.read_csv(INPUT_CSV)

# Separate features (first 13) and label (14th)
all_feat_df = df.iloc[:, :13].copy()
labels = df.iloc[:, 13]

# Normalize column-name types to strings for robust matching
all_feat_df.columns = [str(c) for c in all_feat_df.columns]
all_feature_names = all_feat_df.columns.tolist()

# --- Build the selection ---
if SELECT_BY_NAME:
    missing = [c for c in WAVELENGTH_COLUMNS if c not in all_feature_names]
    if missing:
        raise ValueError(f"The following requested columns were not found among the first 13 features: {missing}\n"
                         f"Available: {all_feature_names}")
    feat_df = all_feat_df[WAVELENGTH_COLUMNS].astype(float)
    sel_names = WAVELENGTH_COLUMNS
else:
    if not WAVELENGTH_IDX:
        raise ValueError("You chose SELECT_BY_NAME=False but WAVELENGTH_IDX is empty.")
    if any((i < 0 or i >= 13) for i in WAVELENGTH_IDX):
        raise ValueError("Indices in WAVELENGTH_IDX must be within [0,12] for the first 13 features.")
    sel_names = [all_feature_names[i] for i in WAVELENGTH_IDX]
    feat_df = all_feat_df.iloc[:, WAVELENGTH_IDX].astype(float)

X = feat_df.to_numpy()
feature_names = sel_names  # for plots/loadings

m, d = X.shape
if d < 1:
    raise ValueError("No features selected. Please choose at least one wavelength.")

# ================== PCA (from equations) ==================
# mean & center
mu = X.mean(axis=0)
Xc = X - mu

# covariance
C = (Xc.T @ Xc) / (m - 1) if m > 1 else np.zeros((d, d))

# eigen-decomp
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# explained variance
total_var = eigvals.sum() if eigvals.sum() > 0 else 1.0
evr = eigvals / total_var
cev = np.cumsum(evr)

# choose k for 95% variance (but not exceeding d)
tau = 0.95
k = int(np.searchsorted(cev, tau) + 1)
k = max(1, min(k, d))

# project
Vk = eigvecs[:, :k]
Y  = Xc @ Vk

# ================== SAVE TABLES ==================
subset_tag = "_subset_" + "_".join(feature_names)
safe_tag = "".join([c if c.isalnum() or c in "._-" else "_" for c in subset_tag])

pd.DataFrame(Y, columns=[f"PC{i+1}" for i in range(k)]).assign(label=labels).to_csv(
    OUT_DIR / f"pca_scores{safe_tag}.csv", index=False
)
pd.DataFrame({"eigenvalue": eigvals, "explained_variance_ratio": evr, "cumulative_evr": cev}).to_csv(
    OUT_DIR / f"pca_evr_all{safe_tag}.csv", index=False
)
pd.DataFrame(Vk, index=feature_names, columns=[f"PC{i+1}" for i in range(k)]).to_csv(
    OUT_DIR / f"pca_components_loadings{safe_tag}.csv"
)
pd.DataFrame({"feature": feature_names, "mean": mu}).to_csv(
    OUT_DIR / f"pca_mean_vector{safe_tag}.csv", index=False
)
with open(OUT_DIR / f"pca_metadata{safe_tag}.json", "w", encoding="utf-8") as f:
    json.dump({"m": int(m), "d": int(d), "chosen_k": int(k), "tau": tau, "features": feature_names}, f, indent=2)

# ================== PLOTS ==================
plt.rcParams.update({"figure.dpi": 140})

def plot_scree(evr, cev, out_path):
    x = np.arange(1, len(evr)+1)
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.bar(x, evr, label="Explained variance (per PC)")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(x, cev, marker="o", linewidth=1.5, color="red", label="Cumulative EVR")
    ax2.set_ylabel("Cumulative EVR", color="red")
    ax2.set_ylim(0, 1.05)

    lines, labels_leg = [], []
    for a in (ax1, ax2):
        lns, lbs = a.get_legend_handles_labels()
        lines += lns; labels_leg += lbs
    fig.legend(lines, labels_leg, loc="upper right")
    plt.title("PCA Scree Plot (subset)")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def label_to_int(series):
    codes = pd.Categorical(series).codes
    return codes, pd.Categorical(series).categories.tolist()

def plot_scores_2d(Y, codes, cats, pcx, pcy, out_path):
    if Y.shape[1] < max(pcx, pcy):
        return
    x = Y[:, pcx-1]
    y = Y[:, pcy-1]
    fig, ax = plt.subplots(figsize=(5.5,5))
    for i, cat in enumerate(cats):
        mask = (codes == i)
        ax.scatter(x[mask], y[mask], s=28, edgecolor="k", linewidth=0.5, label=str(cat), alpha=0.9)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(f"PC{pcx} ({evr[pcx-1]*100:.1f}% var)")
    ax.set_ylabel(f"PC{pcy} ({evr[pcy-1]*100:.1f}% var)")
    ax.set_title(f"PCA Scores: PC{pcx} vs PC{pcy} (subset)")
    ax.legend(title="Label", fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def plot_loadings_bar(loadings, feature_names, pc_index, out_path):
    if loadings.shape[1] < pc_index:
        return
    w = loadings[:, pc_index-1]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(np.arange(len(feature_names)), w)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Loading")
    ax.set_title(f"PC{pc_index} Loadings (subset)")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# Make plots
plot_scree(evr, cev, OUT_DIR / f"plot_scree_evr_cumulative{safe_tag}.png")

codes, cats = label_to_int(labels)
plot_scores_2d(Y, codes, cats, 1, 2, OUT_DIR / f"plot_scores_PC1_PC2{safe_tag}.png")
plot_scores_2d(Y, codes, cats, 1, 3, OUT_DIR / f"plot_scores_PC1_PC3{safe_tag}.png")
plot_loadings_bar(Vk, feature_names, 1, OUT_DIR / f"plot_loadings_PC1{safe_tag}.png")
plot_loadings_bar(Vk, feature_names, 2, OUT_DIR / f"plot_loadings_PC2{safe_tag}.png")

print("Saved PCA (subset) to:", OUT_DIR)
print("Selected features:", feature_names)
