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

# Lecture 12 - Self-supervised Learning

```{contents}
:local:
:depth: 1
```

## Learning goals

- Build **clustering** workflows: pick features, scale, fit, visualize.
- Choose and justify distance metrics for descriptors vs fingerprints.
- Select k and model type using elbow and silhouette.
<<<<<<< HEAD

[![Colab](https://img.shields.io/badge/Open-Colab-orange)](https://colab.research.google.com/drive/1CGznPlVhSet10f820k7TyPvk3kcBdHkC?usp=sharing)

=======
- Diagnose and improve clusters with simple augmentation, dimensionality reduction, and iteration.
- Train a small **contrastive model** with data augmentation.
>>>>>>> 6cda9aa (0929-1)

## 1. Setup

```{code-cell} ipython3
<<<<<<< HEAD
:tags: [hide-input]
=======
>>>>>>> 6cda9aa (0929-1)
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors, AllChem, rdFingerprintGenerator, DataStructs
    RD = True
except Exception:
    try:
        %pip install rdkit
        from rdkit import Chem
        from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors, AllChem, rdFingerprintGenerator, DataStructs
        RD = True
    except Exception as e:
        print("RDKit is not available in this environment. Drawing and descriptors will be skipped.")
        RD = False
        Chem = None

<<<<<<< HEAD
# UMAP install guard
try:
    import umap
    from umap import UMAP
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

=======
>>>>>>> 6cda9aa (0929-1)
import warnings
warnings.filterwarnings("ignore")
```

<<<<<<< HEAD
## 2. Data Loading
=======
## 2. From dimension reduction to clustering
>>>>>>> 6cda9aa (0929-1)

Similar to we we did before, let's first make small helper to compute 4 quick descriptors and a compact fingerprint.

```{code-cell} ipython3
def calc_desc4(smiles: str):
    if not RD:
        return pd.Series({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan})
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return pd.Series({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan})
    return pd.Series({
        "MolWt": Descriptors.MolWt(m),
        "LogP": Crippen.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "NumRings": rdMolDescriptors.CalcNumRings(m),
    })

def morgan_bits(smiles: str, n_bits: int = 128, radius: int = 2):
    if not RD:
        return np.zeros(n_bits, dtype=int)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=int)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(m)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
```

Load the same C-H oxidation dataset used in Lectures 7 and 11.

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)
df_raw.head(3)
```

Compute features we will use today and keep the `Reactivity` column for later evaluation.

```{code-cell} ipython3
desc = df_raw["SMILES"].apply(calc_desc4)
df = pd.concat([df_raw, desc], axis=1)

# fingerprint matrix as 0 or 1
FP_BITS = 128
fp_mat = np.vstack(df_raw["SMILES"].apply(lambda s: morgan_bits(s, n_bits=FP_BITS, radius=2)).values)

# a tidy feature table that has descriptors and the label of interest
cols_x = ["MolWt", "LogP", "TPSA", "NumRings"]
keep = ["Compound Name", "SMILES", "Reactivity"] + cols_x
frame = pd.concat([df[keep].reset_index(drop=True), pd.DataFrame(fp_mat, columns=[f"fp_{i}" for i in range(FP_BITS)])], axis=1)
frame.head()
```

We will standardize the small descriptor block to avoid scale dominance.

```{code-cell} ipython3
scaler = StandardScaler().fit(frame[cols_x])
X_desc = scaler.transform(frame[cols_x])          # shape: n x 4
X_fp   = frame[[c for c in frame.columns if c.startswith("fp_")]].to_numpy().astype(float)  # n x 128
y_reac = frame["Reactivity"].astype(str)          # strings such as "low", "medium", "high" if available
X_desc[:2], X_fp.shape, y_reac.value_counts().to_dict()
```


<<<<<<< HEAD
=======
---

## 2. From dimension reduction to clustering
>>>>>>> 6cda9aa (0929-1)

In Lecture 11 we mapped high dimensional features to 2D for plots using PCA, t-SNE, and UMAP. Today we take the next step: form clusters that group similar molecules together. We will start simple and add checks.

Key ideas:

- Clustering uses only $X$. No label $y$ during fit.
- Distance matters. For descriptors we use Euclidean on standardized columns. For binary fingerprints many chemists prefer Tanimoto similarity, with distance $d_{\text{tan}} = 1 - s_{\text{tan}}$ where
  $
  s_{\text{tan}}(i,j) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.
  $

First let's do a quick 2D descriptor map to build intuition.

```{code-cell} ipython3
pca = PCA(n_components=2, random_state=0).fit(X_desc)
Z = pca.transform(X_desc)

plt.scatter(Z[:,0], Z[:,1], s=12, alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA on 4 descriptors - preview")
plt.show()
```


---



<<<<<<< HEAD

=======
## 3. KMeans on clustering
>>>>>>> 6cda9aa (0929-1)

## 3. KMeans on clustering

We start with **KMeans** because it is one of the simplest and most widely used clustering algorithms. It gives us a baseline understanding of how groups may form in our dataset.

---

### 3.1 Fit KMeans for a single $k$

In clustering, $k$ is the number of groups we want to divide our data into.  
KMeans works by alternating between two steps:

1. **Assignment step**  
   Each data point $x_i$ is assigned to the nearest cluster center $c_j$ according to Euclidean distance:  

   $
   \text{assign}(x_i) = \arg \min_j \| x_i - c_j \|^2
   $

2. **Update step**  
   Each cluster center $c_j$ is updated to be the mean of all points assigned to it:  

   $
   c_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i
   $

The process repeats until the centers stabilize or a maximum number of iterations is reached.

The optimization goal is to minimize the **within-cluster sum of squares** (WCSS):

$
\min_{c_1, \dots, c_k} \sum_{j=1}^k \sum_{x_i \in S_j} \| x_i - c_j \|^2
$


Our descriptor space has 4 dimensions. To plot, we project the standardized descriptors into **2D PCA space**.  
This does not affect clustering (which is run in the original scaled space), but helps us visualize the results.


```{code-cell} ipython3
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X_desc)
labels_km = kmeans.labels_

pd.Series(labels_km).value_counts().sort_index()
```

Map clusters on our PCA plane.

```{code-cell} ipython3
plt.figure(figsize=(5,4))
for lab in np.unique(labels_km):
    idx = labels_km == lab
    plt.scatter(Z[idx,0], Z[idx,1], s=14, alpha=0.8, label=f"cluster {lab}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("KMeans clusters (k=3) on PCA(Descriptors)")
plt.legend()
plt.show()
```
```{admonition} ⏰ **Exercise**
Try `k=2` and `k=4`.

```


Peek at cluster centers in the original descriptor space. We inverse transform to original units.

```{code-cell} ipython3
centroids_std = kmeans.cluster_centers_                           # in z space
centroids_orig = scaler.inverse_transform(centroids_std)          # back to original units
cent_tab = pd.DataFrame(centroids_orig, columns=cols_x)
cent_tab
```
```{code-cell} ipython3
:tags: [hide-input]
# Plot KMeans clusters with centroids as stars
plt.figure(figsize=(5,4))
for c in np.unique(labels_km):
    plt.scatter(Z[labels_km==c,0], Z[labels_km==c,1], s=14, alpha=0.7, label=f"c{c}")
cent_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(cent_pca[:,0], cent_pca[:,1], s=500, marker="*", c="red", edgecolor="w", label="centroid")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("KMeans clusters with centroids")
plt.legend(); plt.show()

```
### 3.2 What does a single sample look like

Sometimes it helps to print one row to see the scaled numbers and the assigned cluster.

```{code-cell} ipython3
i = 7
print("Compound:", frame.loc[i, "Compound Name"])
print("Original desc:", frame.loc[i, cols_x].to_dict())
print("Scaled desc:", dict(zip(cols_x, np.round(X_desc[i], 3))))
print("Cluster:", labels_km[i])
```
### 3.3 KMeans on t-SNE and UMAP
Let's also take a look at how kmeans can be applied to t-SNE and UMAP we learned during the last lecture.

```{code-cell} ipython3
from sklearn.manifold import TSNE

<<<<<<< HEAD
=======
# UMAP install guard
try:
    import umap
    from umap import UMAP
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

>>>>>>> 6cda9aa (0929-1)
def embed_descriptors(X, use_umap=True, random_state=0):
    if use_umap and HAVE_UMAP:
        reducer = UMAP(n_neighbors=15, min_dist=0.10, metric="euclidean", random_state=random_state)
        Z = reducer.fit_transform(X)
        name = "UMAP(desc)"
    else:
        reducer = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                       init="pca", metric="euclidean", random_state=random_state)
        Z = reducer.fit_transform(X)
        name = "t-SNE(desc)"
    return Z, name

# 1) Embed descriptors
Z_desc, name_desc = embed_descriptors(X_desc, use_umap=True, random_state=0)

# 2) KMeans on the 2D embedding
k = 3
km_desc = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z_desc)
labs_desc = km_desc.labels_

# 3) Plot clusters
plt.figure(figsize=(5,4))
for c in np.unique(labs_desc):
    idx = labs_desc == c
    plt.scatter(Z_desc[idx,0], Z_desc[idx,1], s=14, alpha=0.9, label=f"c{c}")
plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.title(f"{name_desc} + KMeans(k={k}) on descriptors")
plt.legend()
plt.show()

```


```{admonition} ⏰ **Exercise**

Change `use_umap = True` to `False` and see the change by using t-sne.

Also try to use k = `2`, `3`, and `4`.

Run the cell below to see the difference.
```
```{code-cell} ipython3
:tags: [hide-input]
def pick_two_per_cluster(labels):
    picks = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        picks[c] = idx[:2] if len(idx) >= 2 else idx
    return picks

def show_molecules(indices, smiles_col="SMILES", names_col="Compound Name"):
    if not RD:
        print("RDKit not available, cannot draw molecules.")
        return
    mols, legends = [], []
    for i in indices:
        smi = frame.loc[i, smiles_col]
        m = Chem.MolFromSmiles(smi) if smi is not None else None
        mols.append(m)
        legends.append(f"{frame.loc[i, names_col]}  (idx {i})")
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(250,250), legends=legends, returnPNG=True)
    display(img)

# 1) pick two per cluster
picks_desc = pick_two_per_cluster(labs_desc)

# 2) draw molecules for each cluster
for c, idxs in picks_desc.items():
    print(f"\nDescriptors {name_desc} cluster c{c}: showing up to 2 molecules")
    show_molecules(idxs)

# 3) star their positions on the existing embedding
plt.figure(figsize=(5,4))
for c in np.unique(labs_desc):
    idx = labs_desc == c
    plt.scatter(Z_desc[idx,0], Z_desc[idx,1], s=14, alpha=0.7, label=f"c{c}")

for c, idxs in picks_desc.items():
    if len(idxs) == 0: 
        continue
    plt.scatter(Z_desc[idxs,0], Z_desc[idxs,1], s=140, marker="*", edgecolor="k", linewidth=0.7, label=f"c{c} picks")

plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.title(f"{name_desc} + KMeans(k=3) on descriptors with picked molecules")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```
Instead of using descritors, we also try using fingerprints.

Different from **Tanimoto similarity** we used before, this time we demonstrated **Jaccard similarity**. For binary molecular fingerprints (0/1 bits), **Jaccard** and **Tanimoto** are mathematically the same.

**Jaccard similarity**:  
$$
s_J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Tanimoto similarity**:  
$$
s_T(A, B) = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

Since $|A \cup B| = |A| + |B| - |A \cap B|$, we have $s_J = s_T$ for binary fingerprints.

```{code-cell} ipython3
def embed_fingerprints(X_bits_bool, use_umap=True, random_state=0):
    if use_umap and HAVE_UMAP:
        reducer = UMAP(n_neighbors=15, min_dist=0.10, metric="jaccard", random_state=random_state)
        Z = reducer.fit_transform(X_bits_bool)
        name = "UMAP(fp, Jaccard)"
    else:
        D_jac = pairwise_distances(X_bits_bool, metric="jaccard")
        reducer = TSNE(n_components=2, metric="precomputed", perplexity=30,
                       learning_rate="auto", init="random", random_state=random_state)
        Z = reducer.fit_transform(D_jac)
        name = "t-SNE(fp, Jaccard)"
    return Z, name

X_fp_bool = X_fp.astype(bool)
Z_fp, name_fp = embed_fingerprints(X_fp_bool, use_umap=True, random_state=0)

k = 3
km_fp = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Z_fp)
labs_fp = km_fp.labels_

plt.figure(figsize=(5,4))
for c in np.unique(labs_fp):
    idx = labs_fp == c
    plt.scatter(Z_fp[idx,0], Z_fp[idx,1], s=14, alpha=0.9, label=f"c{c}")
plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.title(f"{name_fp} + KMeans(k={k}) on fingerprints")
plt.legend()
plt.show()

```
```{code-cell} ipython3
:tags: [hide-input]
# 1) pick two per cluster
picks_fp = pick_two_per_cluster(labs_fp)

# 2) draw molecules for each cluster
for c, idxs in picks_fp.items():
    print(f"\nFingerprints {name_fp} cluster c{c}: showing up to 2 molecules")
    show_molecules(idxs)

# 3) star their positions on the existing embedding
plt.figure(figsize=(5,4))
for c in np.unique(labs_fp):
    idx = labs_fp == c
    plt.scatter(Z_fp[idx,0], Z_fp[idx,1], s=14, alpha=0.7, label=f"c{c}")

for c, idxs in picks_fp.items():
    if len(idxs) == 0:
        continue
    plt.scatter(Z_fp[idxs,0], Z_fp[idxs,1], s=140, marker="*", edgecolor="k", linewidth=0.7, label=f"c{c} picks")

plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.title(f"{name_fp} + KMeans(k=3) on fingerprints with picked molecules")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

```


---
## 4. Picking k and validating clusters

Choosing the right number of clusters $k$ is a central step in KMeans.  
If $k$ is too small, distinct groups may be forced together.  
If $k$ is too large, the algorithm may split natural groups unnecessarily.



### 4.1 Elbow plot

KMeans optimizes the **within-cluster sum of squares** (WCSS), often denoted $W_k$:

$
W_k = \sum_{j=1}^k \sum_{x_i \in S_j} \| x_i - c_j \|^2
$

- $S_j$ = the set of points in cluster $j$  
- $c_j$ = the centroid of cluster $j$  
- $\| x_i - c_j \|^2$ = squared Euclidean distance

As $k$ increases, $W_k$ always decreases, since more clusters mean smaller groups.  
But the **rate of decrease slows down**. The "elbow" of the curve marks a good trade-off:  
beyond this point, adding clusters yields little gain.
```{code-cell} ipython3
ks = range(2, 10)
inertias = []
for kk in ks:
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(X_desc)
    inertias.append(km.inertia_)

plt.plot(list(ks), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow sweep on descriptors")
plt.grid(True)
plt.show()
```

### 4.2 Silhouette score

Another way to decide $k$ is the **silhouette score**, which measures how well each point fits within its cluster compared to others.

For a point $i$:

1. Let $a(i)$ = average distance of $i$ to all other points in the **same cluster** (intra-cluster distance).  
2. Let $b(i)$ = the minimum average distance of $i$ to points in **any other cluster** (nearest-cluster distance).  

Then the silhouette of point $i$ is:

$
s(i) = \frac{b(i) - a(i)}{\max \{ a(i), b(i) \}}
$

In particular:

- $s(i)$ ranges in $[-1, 1]$.  
- $s(i) \approx 1$: point is well clustered (much closer to its own cluster than others).  
- $s(i) \approx 0$: point is on a boundary between clusters.  
- $s(i) < 0$: point may be misclassified (closer to another cluster).  

Take home messgae: higher $S$ the better.

```{code-cell} ipython3
sil = []
for kk in ks:
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(X_desc)
    sil.append(silhouette_score(X_desc, km.labels_))

pd.DataFrame({"k": list(ks), "silhouette": np.round(sil,3)})
```

Visualize the silhouette for a chosen k.

```{code-cell} ipython3
def plot_silhouette(X, labels):
    s = silhouette_samples(X, labels)
    order = np.argsort(labels)
    s_sorted = s[order]
    lbl_sorted = np.array(labels)[order]
    plt.figure(figsize=(5,3))
    y0 = 0
    for c in np.unique(lbl_sorted):
        vals = s_sorted[lbl_sorted == c]
        y1 = y0 + len(vals)
        plt.barh(np.arange(y0, y1), vals, edgecolor="none")
        plt.text(0.02, (y0+y1)/2, f"c{c}", va="center")
        y0 = y1
    plt.xlabel("silhouette")
    plt.ylabel("samples (grouped by cluster)")
    plt.title("Silhouette plot")
    plt.axvline(np.mean(s), color="k", linestyle="--")
    plt.show()

plot_silhouette(X_desc, labels_km)
```

```{admonition} ⏰ **Exercise 4**

In section 3, we also have `Z_desc` from either t-sne or umap, try to use `Z_desc` to calcuate elbow and silhouette.

We also have `Z_fp`, try to use it to make the plot as well.

```

---

## 5. Alternative clustering methods


KMeans assumes clusters are roughly spherical and similar in size.  
But real chemical or molecular data may not follow this pattern.  
That is why it is useful to explore **alternative clustering methods** that can adapt to different cluster shapes.



### 5.1 Agglomerative (Ward)

Agglomerative clustering is a **hierarchical method**.  
It does not start with predefined cluster centers. Instead, it begins by treating each point as its own cluster and then **merges clusters step by step** until only $k$ remain. Here are the steps:

1. **Initialization**: each point is its own cluster.  
2. **Iteration**: repeatedly merge the two clusters that are closest.  
3. **Stop** when exactly $k$ clusters remain.


Different definitions of "closest" give different results.  
The **Ward method** merges the two clusters that cause the smallest increase in within-cluster variance.

Formally, at each step it chooses the merge that minimizes the increase of:

$
\sum_{j=1}^{k} \sum_{x_i \in S_j} \| x_i - c_j \|^2
$

This is similar to the KMeans objective, but built hierarchically rather than iteratively reassigning points.

```{code-cell} ipython3
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
lab_agg = agg.fit_predict(X_desc)

plt.figure(figsize=(5,4))
for lab in np.unique(lab_agg):
    idx = lab_agg == lab
    plt.scatter(Z[idx,0], Z[idx,1], s=14, alpha=0.8, label=f"agg {lab}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Agglomerative on descriptors")
plt.legend(); plt.show()

if y_reac.nunique() > 1:
    print("ARI:", round(adjusted_rand_score(y_reac, lab_agg), 3),
          "NMI:", round(normalized_mutual_info_score(y_reac, lab_agg), 3))
```

### 5.2 DBSCAN

Unlike KMeans or Agglomerative, **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) does not require $k$ in advance.  
Instead, it groups points based on density and labels sparse points as noise The key ideas are:

- A point is a core point if at least `min_samples` neighbors fall within radius $\varepsilon$ (epsilon).  
- Points within $\varepsilon$ of a core point are part of the same cluster.  
- Clusters expand outward from core points.  
- Points not reachable from any core point are labeled noise ($-1$).

So, why DBSCAN is useful? It can find **non-spherical** clusters (e.g., moons, rings). As we will see for the example below. Also it handles noise explicitly and does not force every point into a cluster.  

But: results depend strongly on $\varepsilon$ and `min_samples`.

```{code-cell} ipython3
db = DBSCAN(eps=0.65, min_samples=5).fit(X_desc)
lab_db = db.labels_
np.unique(lab_db, return_counts=True)
```

```{code-cell} ipython3
plt.figure(figsize=(5,4))
for lab in np.unique(lab_db):
    idx = lab_db == lab
    name = "noise" if lab == -1 else f"db {lab}"
    plt.scatter(Z[idx,0], Z[idx,1], s=14, alpha=0.8, label=name)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("DBSCAN on descriptors")
plt.legend(); plt.show()
```

Tip: tune `eps` by scanning a small grid.

```{code-cell} ipython3
eps_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
rows = []
for e in eps_grid:
    db = DBSCAN(eps=e, min_samples=8).fit(X_desc)
    labs = db.labels_
    n_noise = np.sum(labs == -1)
    n_clu = len(np.unique(labs[labs!=-1]))
    rows.append({"eps": e, "n_clusters": n_clu, "n_noise": int(n_noise)})
pd.DataFrame(rows)
```
Below are some examples for you to get a better idea on their difference.

```{code-cell} ipython3
:tags: [hide-input]
"""
- Column 1: KMeans. Pick k by highest silhouette.
- Column 2: Agglomerative (Ward). Pick k by highest silhouette.
- Column 3: DBSCAN. Pick eps and min_samples by highest silhouette.
"""

import time
import warnings
from itertools import cycle, islice

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph

# ---------------------------
# 1) Build toy datasets with different structure
# ---------------------------
n_samples = 1500
rng_seed = 170

noisy_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=rng_seed)
noisy_moons   = make_moons(n_samples=n_samples, noise=0.05, random_state=rng_seed)
blobs         = make_blobs(n_samples=n_samples, random_state=8)

# Anisotropic blobs via linear transform
X_aniso, y_aniso = make_blobs(n_samples=n_samples, random_state=rng_seed)
transformation = np.array([[0.6, -0.6],
                           [-0.4, 0.8]])
X_aniso = X_aniso @ transformation
aniso = (X_aniso, y_aniso)

# Blobs with varied variances
varied = make_blobs(n_samples=n_samples,
                    cluster_std=[1.0, 2.5, 0.5],
                    random_state=rng_seed)

# No structure cloud
rs = np.random.RandomState(rng_seed)
no_structure = (rs.rand(n_samples, 2), None)

datasets = [
    ("noisy_circles", noisy_circles),
    ("noisy_moons",   noisy_moons),
    ("varied_blobs",  varied),
    ("aniso_blobs",   aniso),
    ("blobs",         blobs),
    ("no_structure",  no_structure),
]

# ---------------------------
# 2) Color helper
# ---------------------------
base_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22"
]

def colors_for_labels(y_pred, noise_color="#000000"):
    """
    Map cluster labels to colors.
    DBSCAN noise points labeled as -1 get black.
    """
    y_pred = np.asarray(y_pred).ravel()
    pos = y_pred[y_pred >= 0]
    n_labels = int(pos.max()) + 1 if pos.size else 1
    palette = np.array(list(islice(cycle(base_colors), n_labels)))
    col = palette[np.clip(y_pred, 0, n_labels - 1)]
    col[y_pred == -1] = noise_color
    return col

# ---------------------------
# 3) Scoring utility
# ---------------------------
def safe_silhouette(X, labels):
    """
    Return silhouette score if valid. Otherwise a very low score.
    Valid means at least 2 clusters and no single-cluster assignment.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    # Remove noise for the check when all are noise
    if len(uniq) == 1:
        return -1e9
    # If DBSCAN produced only noise and one cluster, also invalid
    if np.all(labels == -1):
        return -1e9
    # Check at least 2 non-noise clusters
    nn = labels[labels != -1]
    if nn.size == 0 or np.unique(nn).size < 2:
        return -1e9
    try:
        return silhouette_score(X, labels)
    except Exception:
        return -1e9

# ---------------------------
# 4) Model selection per method
# ---------------------------
def select_kmeans(X, k_range=range(2, 5)):
    best = {"score": -1e9, "k": None, "labels": None, "fit_time": 0.0}
    for k in k_range:
        km = cluster.KMeans(n_clusters=k, random_state=42, n_init=10)
        t0 = time.time()
        labels = km.fit_predict(X)
        fit_t = time.time() - t0
        score = safe_silhouette(X, labels)
        if score > best["score"]:
            best = {"score": score, "k": k, "labels": labels, "fit_time": fit_t}
    return best

def select_ward(X, k_range=range(2, 5), n_neighbors=10):
    # Build a sparse graph to encourage local merges
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    best = {"score": -1e9, "k": None, "labels": None, "fit_time": 0.0}
    for k in k_range:
        ward = cluster.AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=connectivity)
        t0 = time.time()
        labels = ward.fit_predict(X)
        fit_t = time.time() - t0
        score = safe_silhouette(X, labels)
        if score > best["score"]:
            best = {"score": score, "k": k, "labels": labels, "fit_time": fit_t}
    return best

def select_dbscan(X, eps_grid=None, min_samples_grid=(3, 5, 8)):
    # If eps grid not given, make a small sweep adapted to standardized data
    if eps_grid is None:
        eps_grid = np.linspace(0.05, 0.5, 10)
    best = {"score": -1e9, "eps": None, "min_samples": None, "labels": None, "fit_time": 0.0}
    for eps in eps_grid:
        for ms in min_samples_grid:
            db = cluster.DBSCAN(eps=eps, min_samples=ms)
            t0 = time.time()
            labels = db.fit_predict(X)
            fit_t = time.time() - t0
            score = safe_silhouette(X, labels)
            if score > best["score"]:
                best = {"score": score, "eps": eps, "min_samples": ms, "labels": labels, "fit_time": fit_t}
    return best

# ---------------------------
# 5) Figure layout: 6 rows x 3 columns
# ---------------------------
n_rows = len(datasets)
n_cols = 3
fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 2.6))
plt.subplots_adjust(left=.04, right=.99, bottom=.04, top=.92, wspace=.07, hspace=.18)

col_titles = ["KMeans (best k)", "Agglomerative Ward (best k)", "DBSCAN (best eps, ms)"]

plot_idx = 1
for row_i, (ds_name, ds) in enumerate(datasets):
    X, y = ds
    # Standardize for fair distance use
    X = StandardScaler().fit_transform(X)

    # KMeans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_km = select_kmeans(X)
    ax = plt.subplot(n_rows, n_cols, plot_idx); plot_idx += 1
    cols = colors_for_labels(best_km["labels"])
    ax.scatter(X[:, 0], X[:, 1], s=8, c=cols, linewidths=0, alpha=0.95)
    if row_i == 0:
        ax.set_title(col_titles[0], fontsize=11, pad=8)
    ax.set_ylabel(ds_name, fontsize=11) if (plot_idx - 2) % 3 == 0 else None
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98, f"k={best_km['k']}  sil={best_km['score']:.2f}\n{best_km['fit_time']:.2f}s",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"))

    # Ward
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_wd = select_ward(X)
    ax = plt.subplot(n_rows, n_cols, plot_idx); plot_idx += 1
    cols = colors_for_labels(best_wd["labels"])
    ax.scatter(X[:, 0], X[:, 1], s=8, c=cols, linewidths=0, alpha=0.95)
    if row_i == 0:
        ax.set_title(col_titles[1], fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98, f"k={best_wd['k']}  sil={best_wd['score']:.2f}\n{best_wd['fit_time']:.2f}s",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"))

    # DBSCAN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_db = select_dbscan(X)
    ax = plt.subplot(n_rows, n_cols, plot_idx); plot_idx += 1
    cols = colors_for_labels(best_db["labels"])
    ax.scatter(X[:, 0], X[:, 1], s=8, c=cols, linewidths=0, alpha=0.95)
    if row_i == 0:
        ax.set_title(col_titles[2], fontsize=11, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    label = f"eps={best_db['eps']:.2f}, ms={best_db['min_samples']}"
    ax.text(0.02, 0.98, f"{label}\nsil={best_db['score']:.2f}\n{best_db['fit_time']:.2f}s",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"))

fig.suptitle("Clustering gallery with automatic selection per dataset", fontsize=14)
plt.show()

```
---

<<<<<<< HEAD

---
### 6. Glossary

```{glossary}

clustering
    Group samples using only $X$, no labels during fit.

KMeans
    A clustering algorithm that assigns points to k centroids, updating assignments and centroids to minimize within-cluster variance.

Agglomerative clustering
    A hierarchical algorithm that starts with each point as its own cluster and merges clusters step by step until k remain.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    A density-based clustering method that groups points when they have enough neighbors within a radius. Points that don’t belong to any dense region are labeled noise.

elbow method
    A heuristic for selecting the number of clusters in KMeans by plotting inertia vs k. The "elbow" marks diminishing returns from adding more clusters.

silhouette score
    A metric for clustering quality. For each point, compares average distance to its own cluster vs the nearest other cluster. Ranges from -1 (bad) to +1 (good).

Tanimoto / Jaccard similarity
    For binary fingerprints, both measure overlap divided by union. Often used to compare molecular fingerprints in chemoinformatics.


```
---
## 7. In-class activity

### Q1. t-SNE on 10 descriptors
Rebuild features using 10 descriptors, embed with t-SNE, plot in 2D.

Hint: copy and use function `def calc_descriptors10(smiles: str)` from Lecture 11 if you forget how to define the 10 descriptor.
```python
# TO DO
```

### Q2. Elbow on KMeans
Choose a reasonable k by inertia.

1) Fit `KMeans` for k in {2..9}.
2) Record `inertia_` for each k.
3) Plot `inertia` vs `k` and inspect the elbow.
```python
# TO DO
```



### Q3. Silhouette on KMeans
Choose k by separation vs compactness.

1) Fit `KMeans` for k in {2..9}.
2) Compute `silhouette_score()` for each k.
3) Plot `silhouette` vs `k`. Report the best k by this metric.
=======
## 6. Do clusters track the `Reactivity` column

We do not use `Reactivity` during fit, but we can check alignment after the fact.

```{code-cell} ipython3
def report_against_reactivity(labels, name="model"):
    ct = pd.crosstab(labels, y_reac)
    print(name, "cross tab")
    display(ct)
    if y_reac.nunique() > 1:
        print("ARI:", round(adjusted_rand_score(y_reac, labels), 3))
        print("NMI:", round(normalized_mutual_info_score(y_reac, labels), 3))

report_against_reactivity(labels_km, "KMeans")
report_against_reactivity(lab_agg,  "Agglomerative")
if (lab_db != -1).any():
    report_against_reactivity(lab_db,  "DBSCAN")
```

Sometimes fingerprints carry scaffold patterns that make groups easier.

We will cluster the fingerprint matrix with KMeans on cosine distance. Cosine on bits is a rough proxy. For a closer chem view you can swap in Tanimoto with methods that allow precomputed distances.

```{code-cell} ipython3
# normalize for cosine KMeans trick
norm = np.linalg.norm(X_fp, axis=1, keepdims=True) + 1e-9
X_fp_unit = X_fp / norm

kmeans_fp = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X_fp_unit)
report_against_reactivity(kmeans_fp.labels_, "KMeans on FP (cosine via unit vectors)")
```

Project FP to 2D for inspection.

```{code-cell} ipython3
p_fp = PCA(n_components=2, random_state=0).fit_transform(X_fp)
plt.figure(figsize=(5,4))
for lab in np.unique(kmeans_fp.labels_):
    idx = kmeans_fp.labels_ == lab
    plt.scatter(p_fp[idx,0], p_fp[idx,1], s=12, alpha=0.8, label=f"c{lab}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("KMeans on fingerprints - PCA view")
plt.legend(); plt.show()
```

---

## 7. Data augmentation for chemistry tables

Self-supervised methods often need two augmented views per sample. We keep it simple and transparent.

For descriptors (continuous):
- **Gaussian jitter**: $x' = x + \epsilon$, with $\epsilon \sim \mathcal{N}(0, \sigma^2)$ after standardization.
- **Feature dropout**: set one coordinate to 0 with small probability $p$.

For fingerprints (binary):
- **Bit dropout**: randomly set a small fraction of on bits to 0.
- **Substructure noise**: flip a tiny fraction of bits both ways, but keep the flip rate very small.

We will build two simple augmenters.

```{code-cell} ipython3
rng = np.random.default_rng(0)

def aug_desc(x, sigma=0.1, drop_p=0.1):
    z = x + rng.normal(0, sigma, size=x.shape)
    mask = rng.random(size=x.shape) < drop_p
    z = z.copy()
    z[mask] = 0.0
    return z

def aug_fp(bits, p_drop=0.05, p_flip=0.01):
    b = bits.copy().astype(float)
    on_idx = np.where(b > 0)[0]
    off_idx = np.where(b == 0)[0]
    # drop some on bits
    if len(on_idx) > 0:
        drop_n = max(0, int(p_drop * len(on_idx)))
        sel = rng.choice(on_idx, size=drop_n, replace=False)
        b[sel] = 0.0
    # flip a tiny fraction either way
    flip_n = max(0, int(p_flip * len(b)))
    sel = rng.choice(np.arange(len(b)), size=flip_n, replace=False)
    b[sel] = 1.0 - b[sel]
    return b
```

Inspect a few rows to see the effect.

```{code-cell} ipython3
i = 3
print("Original desc (z):", np.round(X_desc[i], 3))
print("Aug desc (z):     ", np.round(aug_desc(X_desc[i]), 3))

print("\nOriginal FP bits sum:", int(X_fp[i].sum()))
print("Aug FP bits sum:", int(aug_fp(X_fp[i]).sum()))
```

```{admonition} Tip
Augmentation should be mild. If you distort too much, two views no longer represent the same molecule and the method will struggle.
```

---

## 8. Contrastive learning - a gentle SimCLR style demo

Goal: learn an embedding $f_\theta(x)$ so that two augmented views of the same molecule are close and different molecules are apart. We will do a small, clean implementation that runs on CPU with minimal code.

We use a linear projection on top of standardized descriptors:
- Input: 4 z-scored descriptors.
- Hidden: small 32 unit layer with ReLU.
- Projection: 16 unit head used in the loss.

The InfoNCE loss for a batch with pairs $(i, i')$ is
$
\ell = -\frac{1}{N}\sum_{i=1}^{N}\log \frac{\exp(\mathrm{sim}(z_i, z_{i'})/\tau)}{\sum_{j=1}^{2N}\mathbb{1}[j \ne i]\exp(\mathrm{sim}(z_i, z_j)/\tau)}
$
with cosine similarity and temperature $\tau$.

We implement a small training loop in PyTorch if available. If not, we show a simple NumPy fallback that optimizes a cosine triplet style loss.

```{code-cell} ipython3
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False
TORCH_OK
```

### 8.1 PyTorch path

```{code-cell} ipython3
if TORCH_OK:
    class TinySimCLR(nn.Module):
        def __init__(self, d_in=4, d_hidden=32, d_proj=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_proj)
            )

        def forward(self, x):
            z = self.net(x)
            # normalize to unit length for cosine
            z = z / (z.norm(dim=1, keepdim=True) + 1e-9)
            return z

    def info_nce_loss(z1, z2, tau=0.2):
        # z1, z2: [N, d], already unit norm
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                       # [2N, d]
        sim = z @ z.t()                                      # cosine sim if z is normalized
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim = sim / tau

        # for each i in 0..2N-1, the positive index is i^1 (flip last bit)
        targets = torch.arange(2*N, device=z.device)
        targets = targets ^ 1

        # log-softmax over rows, exclude self with mask
        sim = sim.masked_fill(mask, -1e9)
        log_prob = sim.log_softmax(dim=1)
        loss = -log_prob[torch.arange(2*N, device=z.device), targets].mean()
        return loss

    # make small training set from descriptors
    X = torch.tensor(X_desc, dtype=torch.float32)
    model = TinySimCLR(d_in=4, d_hidden=32, d_proj=16)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # simple loader that creates on the fly augmentations
    def batch_indices(n, bs, rng=np.random.default_rng(0)):
        idx = np.arange(n)
        rng.shuffle(idx)
        for i in range(0, n, bs):
            yield idx[i:i+bs]

    losses = []
    model.train()
    for epoch in range(10):
        for idx in batch_indices(len(X_desc), bs=128):
            x_np = X_desc[idx]
            x1 = np.vstack([aug_desc(x) for x in x_np])
            x2 = np.vstack([aug_desc(x) for x in x_np])
            x1 = torch.tensor(x1, dtype=torch.float32)
            x2 = torch.tensor(x2, dtype=torch.float32)

            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce_loss(z1, z2, tau=0.2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
    print("Last loss:", round(losses[-1], 4))
```

Plot the training loss curve.

```{code-cell} ipython3
if TORCH_OK:
    plt.plot(losses)
    plt.xlabel("update"); plt.ylabel("loss")
    plt.title("Contrastive training - loss")
    plt.show()
```

Get the final embeddings and visualize in 2D with PCA for a quick look. Color by `Reactivity` only for reading the plot.

```{code-cell} ipython3
if TORCH_OK:
    model.eval()
    with torch.no_grad():
        Z_embed = model(torch.tensor(X_desc, dtype=torch.float32)).numpy()
    Z2 = PCA(n_components=2, random_state=0).fit_transform(Z_embed)

    colors = pd.Categorical(y_reac).codes
    plt.scatter(Z2[:,0], Z2[:,1], s=12, c=colors, cmap="tab10", alpha=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Learned embedding - projected to 2D")
    plt.show()

    # cluster in the learned space
    kmeans_embed = KMeans(n_clusters=3, random_state=0, n_init=10).fit(Z_embed)
    report_against_reactivity(kmeans_embed.labels_, "KMeans on learned embedding")
```

### 8.2 NumPy fallback (tiny triplet style)

If PyTorch is not available, we do a small projection matrix and optimize a margin loss on cosine similarities.

```{code-cell} ipython3
if not TORCH_OK:
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.1, size=(4, 16))  # projection
    lr = 1e-2

    def proj(x):
        z = x @ W
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        return z

    def triplet_step(Xz, m=0.2, bs=128):
        idx = rng.choice(len(Xz), size=bs, replace=False)
        x = Xz[idx]
        x_pos = np.vstack([aug_desc(v) for v in x])
        x_neg = Xz[rng.choice(len(Xz), size=bs, replace=False)]

        z = proj(x)
        zp = proj(x_pos)
        zn = proj(x_neg)

        # cosine sim
        s_pos = np.sum(z * zp, axis=1)
        s_neg = np.sum(z * zn, axis=1)
        # hinge
        loss = np.maximum(0.0, m - s_pos + s_neg).mean()

        # simple gradient approximation by finite diff on W for clarity in class
        # not efficient, but readable for students
        eps = 1e-3
        grad = np.zeros_like(W)
        for a in range(W.shape[0]):
            for b in range(W.shape[1]):
                W[a,b] += eps
                lp = np.maximum(0.0, m - np.sum(proj(x)*proj(x_pos), axis=1) + np.sum(proj(x)*proj(x_neg), axis=1)).mean()
                W[a,b] -= 2*eps
                lm = np.maximum(0.0, m - np.sum(proj(x)*proj(x_pos), axis=1) + np.sum(proj(x)*proj(x_neg), axis=1)).mean()
                W[a,b] += eps
                grad[a,b] = (lp - lm) / (2*eps)

        return loss, grad

    Xz = X_desc.copy()
    curve = []
    for it in range(20):
        loss, g = triplet_step(Xz, m=0.2, bs=96)
        W -= lr * g
        curve.append(loss)
    print("Last loss:", round(curve[-1], 4))

    plt.plot(curve); plt.xlabel("iter"); plt.ylabel("loss")
    plt.title("NumPy triplet training - loss")
    plt.show()

    Z_embed = proj(X_desc)
    Z2 = PCA(n_components=2, random_state=0).fit_transform(Z_embed)
    colors = pd.Categorical(y_reac).codes
    plt.scatter(Z2[:,0], Z2[:,1], s=12, c=colors, cmap="tab10", alpha=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("Learned embedding - projected to 2D")
    plt.show()

    kmeans_embed = KMeans(n_clusters=3, random_state=0, n_init=10).fit(Z_embed)
    report_against_reactivity(kmeans_embed.labels_, "KMeans on learned embedding")
```

```{admonition} What to expect
- The tiny model is simple, so gains will be modest. That is fine for a first hands-on.
- If clusters on the learned space align a bit better with `Reactivity` than raw descriptors, the signal is working.
- If not, try a slightly larger hidden size, more epochs, or gentler augmentation.
```

---
### 9 Glossary

```{glossary}
clustering
  Group samples using only $X$, no labels during fit.

silhouette score
  For each point compare average distance to its own cluster vs nearest other cluster. Ranges in $[-1,1]$.

ARI
  Adjusted Rand Index. Compares two partitions. Corrects for random agreement.

NMI
  Normalized mutual information between two partitions. Ranges in $[0,1]$.

augmentation
  A small random transform that keeps identity. Used to create two views of the same sample.

contrastive learning
  Learn embeddings so positive pairs are close and negatives are apart.

InfoNCE
  A softmax loss over similarities. Temperature $\tau$ controls sharpness.

Tanimoto similarity
  Overlap over union for bit sets, common for fingerprints.
```
---
## 10. In-class activity

You will work in pairs. Use small edits to code you already ran.

**Q1. Descriptor vs fingerprint clustering**
- Run KMeans with k in {2, 3, 4} on `X_desc` and on `X_fp_unit`.
- For each run, record ARI and NMI vs `Reactivity`.
- Which feature family did better for this dataset on your machine?
>>>>>>> 6cda9aa (0929-1)

```python
# TO DO
```

<<<<<<< HEAD


### Q4. Agglomerative sweep with plots
Compare another clustering model and visualize the chosen solution.

1) For k in {2..9}, fit `AgglomerativeClustering(linkage="ward")`.
2) Compute the silhouette for each k and plot `silhouette` vs `k`.
3) Pick the best k by silhouette, refit, then plot the cluster assignments on the same t-SNE plane from Q1.
=======
**Q2. DBSCAN tuning**
- Fix descriptors. Sweep `eps` in `[0.6, 0.8, 1.0, 1.2]` and `min_samples` in `[5, 8, 12]`.
- For each pair, record number of clusters and noise count.
- Pick one setting that gives at least 2 clusters and less than 20 percent noise. Show PCA plot with those labels.
>>>>>>> 6cda9aa (0929-1)

```python
# TO DO
```

<<<<<<< HEAD
=======
**Q3. Elbow vs silhouette**
- For descriptors, compute inertia and silhouette for k from 2 to 8.
- Choose k by elbow and also by silhouette.
- Are the two choices the same? If not, which do you trust here and why?

```python
# TO DO
```

**Q4. Augmentation sensitivity**
- In the contrastive block, change descriptor jitter `sigma` from 0.05 to 0.2 and dropout `drop_p` from 0.05 to 0.2.
- Rerun for 5 epochs and plot the loss curve.
- Does the loss get unstable when noise is too strong?

```python
# TO DO
```

**Q5. Downstream check with a tiny classifier**
- Take the learned embedding `Z_embed` and split into 80 or 20.
- Train a simple logistic regression on `Z_embed` to predict a binarized `Reactivity` like high vs not high.
- Compare accuracy to the same classifier trained on raw descriptors.

```python
# TO DO
```

---

## 11. Solutions

**Q1**

```{code-cell} ipython3
def sweep_kmeans(X, ks=(2,3,4), title=""):
    rows = []
    for kk in ks:
        km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(X)
        ari = adjusted_rand_score(y_reac, km.labels_) if y_reac.nunique()>1 else np.nan
        nmi = normalized_mutual_info_score(y_reac, km.labels_) if y_reac.nunique()>1 else np.nan
        rows.append({"k": kk, "ARI": ari, "NMI": nmi})
    out = pd.DataFrame(rows).round(3)
    print(title); display(out); return out

tab_desc = sweep_kmeans(X_desc, title="Descriptors")
tab_fp   = sweep_kmeans(X_fp_unit, title="Fingerprints (unit, cosine trick)")
```

**Q2**

```{code-cell} ipython3
rows = []
for eps in [0.6, 0.8, 1.0, 1.2]:
    for ms in [5, 8, 12]:
        db = DBSCAN(eps=eps, min_samples=ms).fit(X_desc)
        labs = db.labels_
        n_noise = int((labs == -1).sum())
        n_cluster = len(np.unique(labs[labs!=-1]))
        rows.append({"eps": eps, "min_samples": ms, "n_clusters": n_cluster, "n_noise": n_noise})
pd.DataFrame(rows)
```

Plot with one selection.

```{code-cell} ipython3
eps_opt, ms_opt = 0.8, 8
db = DBSCAN(eps=eps_opt, min_samples=ms_opt).fit(X_desc)
labs = db.labels_
plt.figure(figsize=(5,4))
for lab in np.unique(labs):
    idx = labs == lab
    name = "noise" if lab == -1 else f"c{lab}"
    plt.scatter(Z[idx,0], Z[idx,1], s=14, alpha=0.8, label=name)
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(f"DBSCAN eps={eps_opt} ms={ms_opt}")
plt.show()
```

**Q3**

```{code-cell} ipython3
ks = range(2,9)
inertia, sil = [], []
for kk in ks:
    km = KMeans(n_clusters=kk, random_state=0, n_init=10).fit(X_desc)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_desc, km.labels_))

display(pd.DataFrame({"k": list(ks), "inertia": inertia, "silhouette": sil}).round(3))
```

**Q4**

```{code-cell} ipython3
def aug_desc_custom(x, sigma=0.05, drop_p=0.05):
    z = x + rng.normal(0, sigma, size=x.shape)
    mask = rng.random(size=x.shape) < drop_p
    z = z.copy()
    z[mask] = 0.0
    return z

if TORCH_OK:
    model2 = TinySimCLR(d_in=4, d_hidden=32, d_proj=16)
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)
    losses2 = []
    for epoch in range(5):
        for idx in batch_indices(len(X_desc), bs=128):
            x_np = X_desc[idx]
            x1 = np.vstack([aug_desc_custom(x, sigma=0.2, drop_p=0.2) for x in x_np])
            x2 = np.vstack([aug_desc_custom(x, sigma=0.2, drop_p=0.2) for x in x_np])
            x1 = torch.tensor(x1, dtype=torch.float32)
            x2 = torch.tensor(x2, dtype=torch.float32)
            z1 = model2(x1); z2 = model2(x2)
            loss = info_nce_loss(z1, z2, tau=0.2)
            opt2.zero_grad(); loss.backward(); opt2.step()
            losses2.append(float(loss.item()))
    plt.plot(losses2); plt.xlabel("update"); plt.ylabel("loss")
    plt.title("High noise training - loss")
    plt.show()
```

**Q5**

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# binarize reactivity if it has multiple levels: high vs not high
yr = (y_reac.str.lower().str.contains("high")).astype(int)

# raw descriptors
Xd_tr, Xd_te, yr_tr, yr_te = train_test_split(X_desc, yr, test_size=0.2, random_state=0, stratify=yr)
clf_d = LogisticRegression(max_iter=1000, random_state=0).fit(Xd_tr, yr_tr)
acc_d = accuracy_score(yr_te, clf_d.predict(Xd_te))

# learned embedding if available
if TORCH_OK:
    with torch.no_grad():
        Z_emb = model(torch.tensor(X_desc, dtype=torch.float32)).numpy()
else:
    Z_emb = Z_embed

Ze_tr, Ze_te, yr_tr2, yr_te2 = train_test_split(Z_emb, yr, test_size=0.2, random_state=0, stratify=yr)
clf_e = LogisticRegression(max_iter=1000, random_state=0).fit(Ze_tr, yr_tr2)
acc_e = accuracy_score(yr_te2, clf_e.predict(Ze_te))

pd.DataFrame({"Model": ["LogReg on descriptors", "LogReg on learned embed"], "Accuracy": [acc_d, acc_e]}).round(3)
```





>>>>>>> 6cda9aa (0929-1)
