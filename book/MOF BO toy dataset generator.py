# MOF BO toy dataset generator (full-factorial 20,000 experiments)
# --------------------------------------------------------------------------------
# DATASET DESCRIPTION
# Name: MOF Bayesian Optimization Toy Dataset
# Purpose: Provide a synthetic benchmark for Gaussian-process BO with active learning.
# Variables:
#   1) temperature (°C): 10 discrete levels -> [25, 40, 55, 70, 85, 100, 115, 130, 155, 160]
#   2) time_h (hours): 10 discrete levels -> [12, 24, ..., 120]
#   3) concentration_M (M): 10 discrete levels -> [0.05, 0.10, ..., 0.50]
#   4) solvent_DMF: one-hot binary {0=H2O, 1=DMF}
#   5) linker (10 choices) with descriptors: MW, logP, TPSA, n_rings, smiles, and family
#
# Yield model:
#   - Smooth family-dependent response vs temperature, time, concentration.
#   - Solvent bias by family.
#   - Descriptor effects (MW, logP, TPSA, n_rings) and interactions to induce clusters.
#   - Add noise (Gaussian, sd=0.03).
#   - Inject real-world quirks:
#       * 10% random failures -> yield = 0
#       * 10% at 70% of expected (before noise)
#       * 15% boosted by +0.10 but clipped at 0.99
#
# Purity model (black box):
#   - Nonlinear with discontinuities and hash-based jitter.
#   - If yield is high, only ~30% of those cases land in high purity; ~70% skew low.
#   - Depends on linker, a hidden pseudo-metal class, temperature, concentration, solvent.
#   - High temperature and high concentration tend to reduce purity.
#
# Reproducibility model (black box, 4-class):
#   - Values in {0.25, 0.5, 0.75, 1.0}.
#   - Strongly dependent on linker, lightly on other factors.
#   - Linker rules:
#       * H2BDC, Fumaric acid -> 80% 1.0, 20% 0.75
#       * H2BDC-NH2, H2BDC-F  -> 70% 0.25, 30% 0.5
#       * Others              -> 50% 0.75, 20% 1.0, 30% 0.5
#   - Temperature sometimes mutates the selected class:
#       * T > 140: 35% chance to drop one class
#       * 120 < T <= 140: 20% drop chance
#       * T < 55: 15% chance to drop one class
#       * Otherwise: 5% random flip up or down
#   - Solvent and concentration add small noise to the mutation probability.
#
# Output:
#   - CSV with 20,000 rows and columns:
#       [linker, family, smiles, MW, logP, TPSA, n_rings,
#        temperature, time_h, concentration_M, solvent_DMF,
#        expected_yield, yield, purity, reproducibility]
#   - Plots:
#       * PCA and t-SNE of feature space
#       * Combined figure with PCA, t-SNE, and a purity heat map
# Notes:
#   - PCA runs on all rows; t-SNE runs on a subset for speed.
#   - Random seeds fixed for reproducibility.
# --------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import product

rng = np.random.default_rng(123)

# Factor levels
temperatures = np.array([25, 40, 55, 70, 85, 100, 115, 130, 155, 160])
times = np.arange(12, 121, 12)  # 12,24,...,120
concs = np.round(np.arange(0.05, 0.51, 0.05), 2)  # 0.05...0.50
solvents = np.array([0, 1])  # 0=H2O, 1=DMF

# Linkers table (10 entries)
linkers = pd.DataFrame([
    ("H2BDC", "O=C(O)c1ccc(cc1)C(=O)O", 166.13, 1.2, 75.0, 1, "BDC"),
    ("Fumaric acid", "O=C(O)/C=C/C(=O)O", 116.07, -0.6, 75.0, 0, "Aliphatic diacid"),
    ("2-methylimidazole", "Cc1ncc[nH]1", 82.10, 0.2, 29.0, 1, "Azole"),
    ("Trimesic acid", "O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1", 210.14, 1.4, 112.0, 1, "Triacid"),
    ("H2BDC-NH2", "Nc1cc(C(=O)O)ccc1C(=O)O", 181.15, 1.0, 101.0, 1, "BDC"),
    ("H2BDC-F", "O=C(O)c1cc(F)ccc1C(=O)O", 184.13, 1.4, 75.0, 1, "BDC"),
    ("1,4-NDC", "O=C(O)c1ccc2cccc(C(=O)O)c2c1", 216.19, 2.0, 75.0, 2, "Naphthalene diacid"),
    ("2,6-NDC", "O=C(O)c1cccc2c1ccc(C(=O)O)c2", 216.19, 2.0, 75.0, 2, "Naphthalene diacid"),
    ("4,4'-BPDC", "O=C(O)c1ccc(cc1)-c2ccc(cc2)C(=O)O", 242.23, 2.5, 75.0, 2, "Biphenyl diacid"),
    ("Benzimidazole", "c1ccc2[nH]cnc2c1", 118.14, 1.3, 25.0, 2, "Azole"),
], columns=["linker", "smiles", "MW", "logP", "TPSA", "n_rings", "family"])

# Family response settings
family_params = {
    "BDC":                 {"T_opt": 130, "t_opt": 48,  "c_opt": 0.25, "solv_bias": 0.10},
    "Aliphatic diacid":    {"T_opt": 85,  "t_opt": 36,  "c_opt": 0.20, "solv_bias": -0.05},
    "Azole":               {"T_opt": 70,  "t_opt": 24,  "c_opt": 0.15, "solv_bias": -0.10},
    "Triacid":             {"T_opt": 115, "t_opt": 72,  "c_opt": 0.30, "solv_bias": 0.05},
    "Naphthalene diacid":  {"T_opt": 140, "t_opt": 60,  "c_opt": 0.30, "solv_bias": 0.12},
    "Biphenyl diacid":     {"T_opt": 145, "t_opt": 72,  "c_opt": 0.30, "solv_bias": 0.15},
}

def gaussian_penalty(x, mu, width):
    return np.exp(-((x - mu) ** 2) / (2 * (width ** 2)))

def expected_yield_row(row):
    params = family_params[row["family"]]
    pT = gaussian_penalty(row["temperature"], params["T_opt"], 20)
    pt = gaussian_penalty(row["time_h"], params["t_opt"], 20)
    pc = gaussian_penalty(row["concentration_M"], params["c_opt"], 0.08)
    solv_term = params["solv_bias"] * (1 if row["solvent_DMF"] == 1 else -1)

    mw_term = np.clip(1.0 - (row["MW"] - 80) / 200.0, 0.0, 1.0)
    tpsa_term = np.clip((row["TPSA"] / 120.0), 0.0, 1.0)
    rings_term = 0.05 * row["n_rings"]
    logp_term = np.clip(1 - abs(row["logP"] - (0.8 + 0.8*row["solvent_DMF"])) / 2.5, 0.0, 1.0)

    fam_hash = hash(row["family"]) % 3
    if fam_hash == 0:
        inter = 0.15 * pc * logp_term + 0.10 * pT * tpsa_term
    elif fam_hash == 1:
        inter = 0.12 * pt * mw_term + 0.08 * pc * rings_term
    else:
        inter = 0.10 * pT * pc + 0.10 * pt * logp_term

    base = 0.15 + 0.45 * (0.5 * pT + 0.3 * pt + 0.2 * pc) + 0.15 * (0.4 * mw_term + 0.4 * tpsa_term + 0.2 * logp_term) + solv_term + inter
    return np.clip(base, 0.02, 0.95)

# Build full-factorial grid of conditions
cond_grid = pd.DataFrame(list(product(temperatures, times, concs, solvents)),
                         columns=["temperature", "time_h", "concentration_M", "solvent_DMF"])

# Cross with linkers to get 20,000 rows
df = cond_grid.merge(linkers, how="cross")
# Reorder columns
df = df[["linker", "family", "smiles", "MW", "logP", "TPSA", "n_rings",
         "temperature", "time_h", "concentration_M", "solvent_DMF"]].copy()

# Expected yield and noise
df["expected_yield"] = df.apply(expected_yield_row, axis=1)
noise = rng.normal(0, 0.03, size=df.shape[0])
df["yield_raw"] = np.clip(df["expected_yield"] + noise, 0.0, 0.99)

# Inject quirks
N = df.shape[0]  # should be 20,000
idx = np.arange(N)
rng.shuffle(idx)
n_fail = int(0.10 * N)
n_low  = int(0.10 * N)
n_high = int(0.15 * N)

fail_idx = idx[:n_fail]
low_idx = idx[n_fail:n_fail+n_low]
high_idx = idx[n_fail+n_low:n_fail+n_low+n_high]

y = df["yield_raw"].to_numpy()
y[fail_idx] = 0.0
y[low_idx]  = np.clip(0.7 * df.loc[low_idx, "expected_yield"].to_numpy(), 0.0, 0.99)
y[high_idx] = np.clip(df.loc[high_idx, "expected_yield"].to_numpy() + 0.10, 0.0, 0.99)
df["yield"] = y

# ---------------------------
# Black-box purity generator
# ---------------------------
def _hash_noise(*tokens, scale=1.0):
    s = "|".join(map(str, tokens))
    h = hash(s)
    u = ((h ^ (h >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF
    return scale * (u - 0.5)

def purity_black_box(row):
    # Hidden pseudo-metal class derived from linker + family, 0..2
    metal_class = (hash(row["linker"] + row["family"]) % 3)
    base_map = {0: 0.60, 1: 0.45, 2: 0.70}
    base = base_map[metal_class]

    T = row["temperature"]
    if T > 140:
        base -= 0.20
    elif T > 120:
        base -= 0.12
    elif 70 <= T <= 110:
        base += 0.03

    c = row["concentration_M"]
    base -= 0.25 * (c - 0.15)

    fam = row["family"]
    solv = row["solvent_DMF"]
    if fam in {"Azole"} and solv == 1:
        base -= 0.05
    if fam in {"BDC", "Biphenyl diacid"} and solv == 1:
        base += 0.04

    osc = 0.08 * np.sin(0.15 * T + 7.0 * c + (hash(row["linker"]) % 5))

    y = row["yield"]
    if y >= 0.70:
        u = (hash((row["linker"], T, row["time_h"], c, solv, "gate")) & 1023) / 1023.0
        branch = (0.25 + 0.40 * (y - 0.70)) if u < 0.30 else (-0.20 - 0.25 * (y - 0.70))
    else:
        branch = -0.05 + 0.10 * y

    jitter = (
        _hash_noise(row["linker"], T, c, solv, fam, scale=0.10)
        + _hash_noise("kink", int(T // 15), int(c*100), scale=0.06)
    )

    purity = base + osc + branch + jitter
    purity = 1 / (1 + np.exp(-4*(purity - 0.5)))
    return float(np.clip(purity, 0.0, 1.0))

df["purity"] = df.apply(purity_black_box, axis=1)

# ---------------------------
# Black-box reproducibility generator (4-class)
# ---------------------------
_repro_levels = [0.25, 0.5, 0.75, 1.0]

def _pick_from_probs(levels, probs, key):
    # deterministic draw in [0,1) from key
    u = ((hash(key) ^ (hash(key) >> 17)) & 0xFFFFFFFF) / 0x100000000
    cum = np.cumsum(probs)
    idx = int(np.searchsorted(cum, u))
    return levels[min(idx, len(levels)-1)]

def _mutate_level(level, down=True):
    order = [0.25, 0.5, 0.75, 1.0]
    i = order.index(level)
    if down:
        return order[max(0, i-1)]
    else:
        return order[min(len(order)-1, i+1)]

def reproducibility_black_box(row):
    L = row["linker"]
    T = row["temperature"]
    c = row["concentration_M"]
    solv = row["solvent_DMF"]

    # base distribution by linker
    if L in {"H2BDC", "Fumaric acid"}:
        base = _pick_from_probs(_repro_levels, [0.0, 0.0, 0.20, 0.80], key=(L, T, c, solv, "base"))
    elif L in {"H2BDC-NH2", "H2BDC-F"}:
        base = _pick_from_probs(_repro_levels, [0.70, 0.30, 0.0, 0.0], key=(L, T, c, solv, "base"))
    else:
        base = _pick_from_probs(_repro_levels, [0.0, 0.30, 0.50, 0.20], key=(L, T, c, solv, "base"))

    # temperature-driven mutation policy
    # high T penalizes reproducibility; low T mild penalty; otherwise small random flips
    if T > 140:
        p_drop = 0.35
    elif T > 120:
        p_drop = 0.20
    elif T < 55:
        p_drop = 0.15
    else:
        p_drop = 0.05

    # small wobble from solvent and concentration
    p_drop = np.clip(p_drop + 0.03*(solv==1) + 0.02*(c > 0.30) - 0.01*(c < 0.12), 0.0, 0.5)

    # decide mutation
    u = ((hash((L, T, c, solv, "mut")) ^ 0x9E3779B9) & 0xFFFFFFFF) / 0x100000000
    if u < p_drop:
        return _mutate_level(base, down=True)
    elif u > 1 - 0.03:  # rare upwards flip
        return _mutate_level(base, down=False)
    else:
        return base

df["reproducibility"] = df.apply(reproducibility_black_box, axis=1).astype(float)

# Output CSV
out_cols = ["linker", "family", "smiles", "MW", "logP", "TPSA", "n_rings",
            "temperature", "time_h", "concentration_M", "solvent_DMF",
            "expected_yield", "yield", "purity", "reproducibility"]
csv_path = "mof_yield_dataset.csv"
df[out_cols].to_csv(csv_path, index=False)

# ---------------------------
# Dimensionality reduction plots
# ---------------------------
feat_cols = ["MW", "logP", "TPSA", "n_rings",
             "temperature", "time_h", "concentration_M", "solvent_DMF"]
X = df[feat_cols].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=123)
pca_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.scatter(pca_2d[:,0], pca_2d[:,1], s=3, alpha=0.4, c="gray")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of MOF features (all points, no grouping)")
plt.show()

families = df["family"].astype("category")
colors = families.cat.codes
plt.figure(figsize=(7,6))
scatter = plt.scatter(pca_2d[:,0], pca_2d[:,1],
                      c=colors, cmap="tab10", s=3, alpha=0.5)
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, families.cat.categories,
           title="Family", bbox_to_anchor=(1.05,1), loc="upper left")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of MOF features (colored by linker family)")
plt.show()

# t-SNE on subset
n_tsne = min(2000, X_scaled.shape[0])
sub_idx = np.random.default_rng(1).choice(np.arange(X_scaled.shape[0]), size=n_tsne, replace=False)
tsne = TSNE(n_components=2, learning_rate="auto", init="pca",
            perplexity=35, random_state=1)
tsne_2d = tsne.fit_transform(X_scaled[sub_idx])

plt.figure(figsize=(7,6))
plt.scatter(tsne_2d[:,0], tsne_2d[:,1], s=4, alpha=0.5, c="gray")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of MOF features (subset, no grouping)")
plt.show()

sub_families = families.iloc[sub_idx]
sub_colors = sub_families.cat.codes
plt.figure(figsize=(7,6))
scatter = plt.scatter(tsne_2d[:,0], tsne_2d[:,1],
                      c=sub_colors, cmap="tab10", s=4, alpha=0.5)
handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
plt.legend(handles, sub_families.cat.categories,
           title="Family", bbox_to_anchor=(1.05,1), loc="upper left")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of MOF features (subset, colored by linker family)")
plt.show()

# ---------------------------
# Combined figure: PCA, t-SNE, and heat map together
# ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(18,5))

# PCA colored by family
sc0 = axes[0].scatter(pca_2d[:,0], pca_2d[:,1], c=colors, cmap="tab10", s=3, alpha=0.6)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("PCA by family")

# t-SNE colored by family (subset)
sc1 = axes[1].scatter(tsne_2d[:,0], tsne_2d[:,1],
                      c=sub_colors, cmap="tab10", s=4, alpha=0.7)
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].set_title("t-SNE by family")

# Heat map of mean purity for a chosen linker and solvent across T x conc
hm_linker = "H2BDC"
hm_solvent = 1  # DMF
subset = df[(df["linker"] == hm_linker) & (df["solvent_DMF"] == hm_solvent)]
pivot = subset.pivot_table(index="temperature", columns="concentration_M", values="purity", aggfunc="mean")
im = axes[2].imshow(pivot.values, aspect="auto", origin="lower")
axes[2].set_xticks(np.arange(len(pivot.columns)))
axes[2].set_xticklabels([str(c) for c in pivot.columns], rotation=90)
axes[2].set_yticks(np.arange(len(pivot.index)))
axes[2].set_yticklabels([str(t) for t in pivot.index])
axes[2].set_xlabel("concentration_M")
axes[2].set_ylabel("temperature")
axes[2].set_title(f"Mean purity heat map\n{hm_linker}, solvent=DMF")
fig.colorbar(im, ax=axes[2], shrink=0.8, label="purity")

plt.tight_layout()
plt.show()
