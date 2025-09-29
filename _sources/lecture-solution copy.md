---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---



## 7. Solution

### Q1

```{code-cell} ipython3
# Q1 - Solution

# 10-descriptor table
desc10_20 = df_20["SMILES"].apply(calc_descriptors10)
df10_20 = pd.concat([df_20[["Compound Name","SMILES","Group"]], desc10_20], axis=1)

# 64-bit Morgan fingerprints
df_fp_20 = df_20.copy()
df_fp_20["Fingerprint"] = df_fp_20["SMILES"].apply(lambda s: morgan_bits(s, n_bits=64, radius=2))

print("10-descriptor table (head):")
display(df10_20.head().round(3))

print("\nFingerprint preview (first 24 bits):")
fp_preview = df_fp_20[["Compound Name","Group","Fingerprint"]].copy()
fp_preview["Fingerprint"] = fp_preview["Fingerprint"].str.slice(0, 24) + "..."
display(fp_preview)

print("\nDescriptor stats:")
display(df10_20.drop(columns=["Compound Name","SMILES","Group"]).agg(["mean","std"]).round(3))

```

### Q2
```{code-cell} ipython3
# Q2 - Solution

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

desc_cols = ["MolWt","LogP","TPSA","NumRings",
             "NumHAcceptors","NumHDonors","NumRotatableBonds",
             "HeavyAtomCount","FractionCSP3","NumAromaticRings"]

X_desc = df10_20[desc_cols].to_numpy()
X_desc_std = StandardScaler().fit_transform(X_desc)

pca_desc_full = PCA().fit(X_desc_std)
Z_desc = pca_desc_full.transform(X_desc_std)[:, :2]

# color map by group
group_to_color = {"A_hydrophobic_aromatics":"tab:orange",
                  "B_polar_alc_acid":"tab:blue",
                  "C_N_hetero_amines":"tab:green"}
colors = df10_20["Group"].map(group_to_color).to_numpy()

# explained variance
plt.figure(figsize=(4,3))
plt.plot(np.cumsum(pca_desc_full.explained_variance_ratio_), marker="o")
plt.xlabel("Number of PCs"); plt.ylabel("Cumulative EVR"); plt.tight_layout(); plt.show()

# 2D scatter with labels
plt.figure(figsize=(7,6))
plt.scatter(Z_desc[:,0], Z_desc[:,1], s=80, c=colors, alpha=0.9)
for i, name in enumerate(df10_20["Compound Name"]):
    plt.text(Z_desc[i,0]+0.03, Z_desc[i,1], name, fontsize=8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA on 10 descriptors (n=20)")
# legend
for g, col in group_to_color.items():
    plt.scatter([], [], c=col, label=g)
plt.legend(title="Group", loc="best")
plt.tight_layout(); plt.show()

```

### Q3

```{code-cell} ipython3
# Q3 - Solution

from sklearn.manifold import TSNE

tsne_desc = TSNE(n_components=2, perplexity=2, learning_rate="auto",
                 init="pca", n_iter=1000, random_state=0)
Y_desc = tsne_desc.fit_transform(X_desc_std)

plt.figure(figsize=(7,6))
plt.scatter(Y_desc[:,0], Y_desc[:,1], s=80, c=colors, alpha=0.9)
for i, name in enumerate(df10_20["Compound Name"]):
    plt.text(Y_desc[i,0]+0.5, Y_desc[i,1], name, fontsize=8)
plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2"); plt.title("t-SNE on 10 descriptors (n=20)")
for g, col in group_to_color.items():
    plt.scatter([], [], c=col, label=g)
plt.legend(title="Group", loc="best")
plt.tight_layout(); plt.show()

```

### Q4
```{code-cell} ipython3
# Q4 - Solution

# 0/1 matrix from bitstrings
X_fp = np.array([[int(ch) for ch in s] for s in df_fp_20["Fingerprint"]], dtype=float)

# PCA on fingerprint bits
pca_fp = PCA(n_components=2, random_state=0).fit(X_fp)
Z_fp_pca = pca_fp.transform(X_fp)

plt.figure(figsize=(7,6))
plt.scatter(Z_fp_pca[:,0], Z_fp_pca[:,1], s=80, c=colors, alpha=0.9)
for i, name in enumerate(df_fp_20["Compound Name"]):
    plt.text(Z_fp_pca[i,0]+0.03, Z_fp_pca[i,1], name, fontsize=8)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA on Morgan-64 (n=20)")
for g, col in group_to_color.items():
    plt.scatter([], [], c=col, label=g)
plt.legend(title="Group", loc="best")
plt.tight_layout(); plt.show()

# t-SNE on fingerprint bits
tsne_fp = TSNE(n_components=2, perplexity=5, learning_rate="auto",
               init="pca", n_iter=1000, random_state=0)
Y_fp = tsne_fp.fit_transform(X_fp)

plt.figure(figsize=(7,6))
plt.scatter(Y_fp[:,0], Y_fp[:,1], s=80, c=colors, alpha=0.9)
for i, name in enumerate(df_fp_20["Compound Name"]):
    plt.text(Y_fp[i,0]+0.5, Y_fp[i,1], name, fontsize=8)
plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2"); plt.title("t-SNE on Morgan-64 (n=20)")
for g, col in group_to_color.items():
    plt.scatter([], [], c=col, label=g)
plt.legend(title="Group", loc="best")
plt.tight_layout(); plt.show()

```

### Q5

```{code-cell} ipython3
# Q5 - Solution

from sklearn.metrics import pairwise_distances
from rdkit import DataStructs

# choose a query row
q = 3  # change during class to test different queries

# 1) Euclidean on standardized 10D descriptors
D_eu = pairwise_distances(X_desc_std, metric="euclidean")
order_eu = np.argsort(D_eu[q])
nn_eu_idx = [i for i in order_eu if i != q][:5]

print(f"Query: {df10_20.loc[q,'Compound Name']}  |  Group: {df10_20.loc[q,'Group']}")
print("\nNearest 5 by Euclidean (10 descriptors):")
for j in nn_eu_idx:
    print(f"  {df10_20.loc[j,'Compound Name']:24s}  group={df10_20.loc[j,'Group']:22s}  d_eu={D_eu[q,j]:.3f}")

# 2) Tanimoto on 64-bit fingerprints
def fp_string_to_bvect(fp_str):
    arr = [int(ch) for ch in fp_str]
    bv = DataStructs.ExplicitBitVect(len(arr))
    for k, b in enumerate(arr):
        if b: bv.SetBit(k)
    return bv

fps = [fp_string_to_bvect(s) for s in df_fp_20["Fingerprint"]]
S_tan = np.zeros((len(fps), len(fps)))
for i in range(len(fps)):
    S_tan[i, :] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
D_tan = 1.0 - S_tan

order_tan = np.argsort(D_tan[q])
nn_tan_idx = [i for i in order_tan if i != q][:5]

print("\nNearest 5 by Tanimoto (Morgan-64):")
for j in nn_tan_idx:
    print(f"  {df10_20.loc[j,'Compound Name']:24s}  group={df10_20.loc[j,'Group']:22s}  d_tan={D_tan[q,j]:.3f}")

```


---

Lecture 12
---


## 8. Solutions

**Q1**

```{code-cell} ipython3
# Q1. t-SNE on 10 descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assumes df_raw is already loaded and RDKit imports exist:
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

def calc_descriptors10(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    return pd.Series({
        "MolWt": Descriptors.MolWt(m),
        "LogP": Crippen.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "NumRings": rdMolDescriptors.CalcNumRings(m),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(m),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(m),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(m),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(m),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(m),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(m)
    })

# 10 descriptors
desc10 = df_raw["SMILES"].apply(calc_descriptors10)
df10 = pd.concat([df_raw.reset_index(drop=True), desc10.reset_index(drop=True)], axis=1)

cols10 = [
    "MolWt","LogP","TPSA","NumRings","NumHAcceptors",
    "NumHDonors","NumRotatableBonds","HeavyAtomCount",
    "FractionCSP3","NumAromaticRings"
]
scaler10 = StandardScaler().fit(df10[cols10])
X10 = scaler10.transform(df10[cols10])

# t-SNE embedding for descriptors (reused in Q4 and Q5)
tsne10 = TSNE(n_components=2, perplexity=30, learning_rate="auto",
              init="pca", metric="euclidean", random_state=0)
Z10 = tsne10.fit_transform(X10)

plt.figure(figsize=(5,4))
plt.scatter(Z10[:,0], Z10[:,1], s=12, alpha=0.85)
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.title("t-SNE on 10 descriptors")
plt.tight_layout()
plt.show()

```

**Q2**

```{code-cell} ipython3
# Q2. Elbow on KMeans (k=2..9)

ks = range(2, 10)
inertias = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X10)
    inertias.append(km.inertia_)

plt.figure(figsize=(5,4))
plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.title("Elbow on 10-descriptor KMeans")
plt.grid(True); plt.tight_layout()
plt.show()

pd.DataFrame({"k": list(ks), "inertia": inertias}).round(3)
```


**Q3**

```{code-cell} ipython3
# Q3. Silhouette on KMeans (k=2..9)
sil_scores = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X10)
    sil_scores.append(silhouette_score(X10, km.labels_))

plt.figure(figsize=(5,4))
plt.plot(list(ks), sil_scores, marker="o")
plt.xlabel("k"); plt.ylabel("Silhouette")
plt.title("Silhouette vs k on 10-descriptor KMeans")
plt.grid(True); plt.tight_layout()
plt.show()

best_k_sil = list(ks)[int(np.argmax(sil_scores))]
print("Best k by silhouette:", best_k_sil)
pd.DataFrame({"k": list(ks), "silhouette": np.round(sil_scores, 3)})

```

**Q4**

```{code-cell} ipython3
# Q4. Agglomerative sweep with plots
sil_agg = []
for k in ks:
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels_agg = agg.fit_predict(X10)
    sil_agg.append(silhouette_score(X10, labels_agg))

plt.figure(figsize=(5,4))
plt.plot(list(ks), sil_agg, marker="o")
plt.xlabel("k"); plt.ylabel("Silhouette")
plt.title("Agglomerative (ward) silhouette vs k on 10 descriptors")
plt.grid(True); plt.tight_layout()
plt.show()

best_k_agg = list(ks)[int(np.argmax(sil_agg))]
print("Best k for Agglomerative by silhouette:", best_k_agg)

# Fit best k and plot clusters on t-SNE plane (Z10)
agg_best = AgglomerativeClustering(n_clusters=best_k_agg, linkage="ward")
labels_agg_best = agg_best.fit_predict(X10)

plt.figure(figsize=(5,4))
for c in np.unique(labels_agg_best):
    idx = labels_agg_best == c
    plt.scatter(Z10[idx,0], Z10[idx,1], s=12, alpha=0.9, label=f"c{c}")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.title(f"Agglomerative (ward) on t-SNE, k={best_k_agg}")
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.show()
```





