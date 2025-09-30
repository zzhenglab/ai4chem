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





