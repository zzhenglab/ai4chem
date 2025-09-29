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

# Lecture 12 — Self-supervised Learning
 
## 0. Setup for sections 7–8

This file is self-contained. It installs packages if needed, loads the dataset, and prepares descriptor tables used below.

```{code-cell} ipython3
# Core
import sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# RDKit (install if missing)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs
except Exception as e:
    print("Installing rdkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"])
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import DataStructs

import warnings
warnings.filterwarnings("ignore")
```

**Load the C–H oxidation dataset**

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)
print(df_raw.shape)
df_raw.head(3)
```

**Descriptor functions**

- `calc_descriptors10` returns 10 quick descriptors per SMILES.
- `morgan_bits` returns a compact bitstring for Morgan fingerprints.

```{code-cell} ipython3
def calc_descriptors10(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return pd.Series({
            "MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan,
            "NumHAcceptors": np.nan, "NumHDonors": np.nan, "NumRotatableBonds": np.nan,
            "HeavyAtomCount": np.nan, "FractionCSP3": np.nan, "NumAromaticRings": np.nan
        })
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

def morgan_bits(smiles: str, n_bits: int = 64, radius: int = 2):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.nan
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(m)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return "".join(map(str, arr))

desc_cols = ["MolWt","LogP","TPSA","NumRings",
             "NumHAcceptors","NumHDonors","NumRotatableBonds",
             "HeavyAtomCount","FractionCSP3","NumAromaticRings"]
```

**Build descriptor tables used in Sections 7 and 8**

```{code-cell} ipython3
# 10-descriptor table
desc10 = df_raw["SMILES"].apply(calc_descriptors10)
df10 = pd.concat([df_raw[["Compound Name","SMILES","Toxicity"]], desc10], axis=1).dropna(subset=desc_cols)
print("df10 shape:", df10.shape)
df10.head(3)
```

---

## 7. In-class activities

We practice three light self-supervised ideas that fit experimental chemistry. Each activity is short and shows intermediate structures.

### 7.1 Masked descriptor imputation (pretext task)

**Idea**  
Randomly hide one descriptor and predict it from the other 9. This is a pretext task that learns relationships inside $x$ without labels $y$.

**Prepare the matrix**

```{code-cell} ipython3
X_desc_all = df10[desc_cols].copy()
X = X_desc_all.to_numpy().astype(float)
X.shape
```

**Loop over columns and score $R^2$ with LinearRegression**

```{code-cell} ipython3
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

col_scores = {}
for j, col in enumerate(desc_cols):
    X_tr, X_te = train_test_split(X, test_size=0.25, random_state=42)
    y_tr, y_te = X_tr[:, j].copy(), X_te[:, j].copy()

    X_trm = X_tr.copy(); X_trm[:, j] = np.nan
    X_tem = X_te.copy(); X_tem[:, j] = np.nan

    imp = SimpleImputer(strategy="mean").fit(X_trm)
    X_trf = imp.transform(X_trm)
    X_tef = imp.transform(X_tem)

    X_tr_use = np.delete(X_trf, j, axis=1)
    X_te_use = np.delete(X_tef, j, axis=1)

    reg = LinearRegression().fit(X_tr_use, y_tr)
    y_hat = reg.predict(X_te_use)
    col_scores[col] = r2_score(y_te, y_hat)

pd.Series(col_scores).sort_values(ascending=False).round(3)
```

We just learned which descriptors are most predictable from the rest. That signals redundancy in the feature space.

**Visualize parity for the best column**

```{code-cell} ipython3
best_col = max(col_scores, key=col_scores.get)
j = desc_cols.index(best_col)

# repeat fit for plotting
X_tr, X_te = train_test_split(X, test_size=0.25, random_state=42)
y_tr, y_te = X_tr[:, j].copy(), X_te[:, j].copy()
X_trm = X_tr.copy(); X_trm[:, j] = np.nan
X_tem = X_te.copy(); X_tem[:, j] = np.nan

imp = SimpleImputer(strategy="mean").fit(X_trm)
X_trf = imp.transform(X_trm); X_tef = imp.transform(X_tem)

X_tr_use = np.delete(X_trf, j, axis=1)
X_te_use = np.delete(X_tef, j, axis=1)

reg = LinearRegression().fit(X_tr_use, y_tr)
y_hat = reg.predict(X_te_use)

print(f"Best masked column: {best_col}")
print(f"R2 on test: {r2_score(y_te, y_hat):.3f}")
```

```{code-cell} ipython3
plt.scatter(y_te, y_hat, alpha=0.6)
lims = [min(y_te.min(), y_hat.min()), max(y_te.max(), y_hat.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title(f"Masked imputation parity for {best_col}")
plt.show()
```

```{admonition} Tip
If a descriptor has low $R^2$ here, it carries information that is less redundant with the rest. That can be useful when you decide which features to keep.
```

**⏰ Exercise 7.1**

- Replace LinearRegression with `Ridge(alpha=1.0)` and repeat the loop. Compare per-column $R^2$.  
- For one weaker column, plot residuals $r = y - \hat y$ vs `MolWt` and comment on the pattern.

```python
# TO DO
```

---

### 7.2 Character language model for SMILES

Train an $n$-gram model that predicts the next character in SMILES. This mirrors next-token or masked-token pretraining at a tiny scale.

**Collect SMILES and build a character set**

```{code-cell} ipython3
smiles = df10["SMILES"].dropna().astype(str).tolist()
from collections import Counter
all_text = "".join(smiles)
chars = sorted(set(all_text))
len(chars), chars[:20]
```

**Count trigrams ($n=3$)**

```{code-cell} ipython3
from collections import defaultdict, Counter

n = 3
counts = defaultdict(Counter)

for s in smiles:
    s2 = "^"*(n-1) + s + "$"
    for i in range(len(s2)-(n-1)):
        context = s2[i:i+(n-1)]
        nxt = s2[i+(n-1)]
        counts[context][nxt] += 1

len(counts)
```

**Turn counts into probabilities with add-$\alpha$ smoothing**

```{code-cell} ipython3
alpha = 0.5
vocab_plus = ["^","$"] + chars

def context_probs(context):
    c = counts[context]
    total = sum(c.values()) + alpha*len(vocab_plus)
    return {ch: (c.get(ch,0) + alpha)/total for ch in vocab_plus}

# peek baseline context
context_probs("^"*(n-1))
```

**Sequence negative log likelihood**

```{code-cell} ipython3
import math

def sequence_nll(s):
    s2 = "^"*(n-1) + s + "$"
    nll = 0.0
    for i in range(len(s2)-(n-1)):
        ctx = s2[i:i+(n-1)]
        nxt = s2[i+(n-1)]
        p = context_probs(ctx).get(nxt, 1e-12)
        nll += -math.log(p + 1e-12)
    return nll

vals = pd.Series([sequence_nll(s) for s in smiles[:10]], index=smiles[:10])
vals.round(2)
```

**Top-k next-token suggestions**

```{code-cell} ipython3
def topk_next(context, k=5):
    pr = context_probs(context)
    return sorted(pr.items(), key=lambda x: -x[1])[:k]

topk_next("^C", k=8)
```

**Sample a few SMILES candidates**

```{code-cell} ipython3
import random
def sample_smiles(rng=None, max_len=120):
    if rng is None:
        rng = random.Random(0)
    s = "^"*(n-1)
    out = []
    for _ in range(max_len):
        pr = context_probs(s[-(n-1):])
        items, probs = zip(*pr.items())
        cum = np.cumsum(probs)
        u = rng.random() * cum[-1]
        j = int(np.searchsorted(cum, u))
        token = items[j]
        if token == "$":
            break
        if token not in ["^"]:
            out.append(token)
        s += token
    return "".join(out)

[sample_smiles(random.Random(i)) for i in range(5)]
```

```{admonition} Note
Many sampled strings will not be valid molecules. The goal is to see how a self-supervised signal teaches token statistics without property labels.
```

**⏰ Exercise 7.2**

- Rebuild the model with $n=4$ and compare mean sequence NLL on a set of 100 molecules.  
- Create a masked character task: pick one position (not the first), hide it, and check if the true char appears in the model’s top-5 guesses. Report hit@5.

```python
# TO DO
```

---

### 7.3 Linear autoencoder via PCA on descriptors

Treat PCA as an encoder-decoder. Choose $k$ by explained variance and study reconstruction error per column and per molecule.

**Standardize and fit PCA**

```{code-cell} ipython3
scaler = StandardScaler().fit(X_desc_all)
Xz = scaler.transform(X_desc_all)

pca = PCA().fit(Xz)
evr = np.cumsum(pca.explained_variance_ratio_)
pd.Series(evr[:10]).round(3)
```

**Pick $k$ where cumulative EVR $\ge 0.95$**

```{code-cell} ipython3
k = int(np.argmax(evr >= 0.95) + 1)
k
```

**Encode and decode**

```{code-cell} ipython3
Z = pca.transform(Xz)[:, :k]
Xz_hat = Z @ pca.components_[:k, :]
X_hat = scaler.inverse_transform(Xz_hat)

recon = pd.DataFrame(X_hat, columns=desc_cols, index=X_desc_all.index)
recon.head(3)
```

**RMSE per descriptor and per molecule**

```{code-cell} ipython3
col_rmse = {c: mean_squared_error(X_desc_all[c], recon[c], squared=False) for c in desc_cols}
pd.Series(col_rmse).sort_values().round(3)
```

```{code-cell} ipython3
per_mol_err = np.sqrt(((X_desc_all.values - recon.values)**2).mean(axis=1))
per_mol_err.describe()
```

```{code-cell} ipython3
plt.hist(per_mol_err, bins=30, alpha=0.7)
plt.xlabel("Per-molecule RMSE (10 desc)")
plt.ylabel("Count")
plt.title(f"PCA autoencoder error, k={k}")
plt.show()
```

**⏰ Exercise 7.3**

- Fix $k=3$ and recompute RMSE per descriptor. Which columns are hardest to reconstruct  
- Plot per-molecule error vs `MolWt` and comment on any trend.

```python
# TO DO
```

---

### 7.4 Contrast of two descriptor views with cosine

Make two simple views of the same molecule: standardized descriptors and a noisy copy. Check that each molecule matches its own view by cosine similarity.

**Build two views**

```{code-cell} ipython3
rng = np.random.RandomState(0)
Xz = scaler.transform(X_desc_all.values)
Xz_noisy = Xz + rng.normal(0, 0.05, size=Xz.shape)

Xz[:2], Xz_noisy[:2]
```

**Cosine similarity and top-1 hit rate**

```{code-cell} ipython3
S = cosine_similarity(Xz, Xz_noisy)
row_argmax = S.argmax(axis=1)
top1 = np.mean(row_argmax == np.arange(S.shape[0]))
print(f"Top-1 match rate for own noisy view: {top1:.3f}")
```

**⏰ Exercise 7.4**

- Increase the noise std to `0.2` and compute the new top-1 rate.  
- Replace cosine with similarity $1/(1+d)$ where $d$ is Euclidean distance and compare.

```python
# TO DO
```

---

### 7.5 Optional: map the encoder codes

Plot the first two principal components as a quick 2D map of the encoder codes.

```{code-cell} ipython3
Z2 = Z[:, :2]
plt.scatter(Z2[:,0], Z2[:,1], s=12, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Encoder codes (first 2 PCs)")
plt.show()
```

---

## 8. Solutions

Short reference solutions that match the activities above.

### Solution 7.1 Ridge and residuals

```{code-cell} ipython3
ridge_scores = {}
for j, col in enumerate(desc_cols):
    X_tr, X_te = train_test_split(X, test_size=0.25, random_state=42)
    y_tr, y_te = X_tr[:, j].copy(), X_te[:, j].copy()
    X_trm, X_tem = X_tr.copy(), X_te.copy()
    X_trm[:, j] = np.nan; X_tem[:, j] = np.nan

    imp = SimpleImputer(strategy="mean").fit(X_trm)
    X_trf = imp.transform(X_trm); X_tef = imp.transform(X_tem)

    X_tr_use = np.delete(X_trf, j, axis=1)
    X_te_use = np.delete(X_tef, j, axis=1)

    rr = Ridge(alpha=1.0).fit(X_tr_use, y_tr)
    ridge_scores[col] = r2_score(y_te, rr.predict(X_te_use))

pd.DataFrame({"Linear": pd.Series(col_scores), "Ridge": pd.Series(ridge_scores)}).round(3).sort_values("Linear", ascending=False)
```

```{code-cell} ipython3
# residuals vs MolWt for a weaker column
weak_col = min(col_scores, key=col_scores.get)
j = desc_cols.index(weak_col)

X_tr, X_te = train_test_split(X, test_size=0.25, random_state=42)
y_tr, y_te = X_tr[:, j].copy(), X_te[:, j].copy()
X_trm, X_tem = X_tr.copy(), X_te.copy()
X_trm[:, j] = np.nan; X_tem[:, j] = np.nan

imp = SimpleImputer(strategy="mean").fit(X_trm)
X_trf = imp.transform(X_trm); X_tef = imp.transform(X_tem)
X_tr_use = np.delete(X_trf, j, axis=1); X_te_use = np.delete(X_tef, j, axis=1)

reg = LinearRegression().fit(X_tr_use, y_tr)
y_hat = reg.predict(X_te_use)
res = y_te - y_hat
molwt_te = X_te[:, desc_cols.index("MolWt")]

plt.scatter(molwt_te, res, alpha=0.6)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("MolWt")
plt.ylabel("Residual")
plt.title(f"Residuals vs MolWt for {weak_col}")
plt.show()
```

---

### Solution 7.2 $n=4$ and masked hit@5

```{code-cell} ipython3
# rebuild counts with n=4
n = 4
counts = defaultdict(Counter)
for s in smiles:
    s2 = "^"*(n-1) + s + "$"
    for i in range(len(s2)-(n-1)):
        ctx = s2[i:i+(n-1)]; nxt = s2[i+(n-1)]
        counts[ctx][nxt] += 1

alpha = 0.5
vocab_plus = ["^","$"] + chars

def context_probs(context):
    c = counts[context]
    total = sum(c.values()) + alpha*len(vocab_plus)
    return {ch: (c.get(ch,0) + alpha)/total for ch in vocab_plus}

def sequence_nll(s):
    s2 = "^"*(n-1) + s + "$"
    nll = 0.0
    for i in range(len(s2)-(n-1)):
        ctx = s2[i:i+(n-1)]; nxt = s2[i+(n-1)]
        p = context_probs(ctx).get(nxt, 1e-12)
        nll += -np.log(p + 1e-12)
    return nll

subset = smiles[:100]
mean_nll_n4 = float(np.mean([sequence_nll(s) for s in subset]))
mean_nll_n4
```

```{code-cell} ipython3
def topk_next(context, k=5):
    pr = context_probs(context)
    return [ch for ch,_ in sorted(pr.items(), key=lambda x: -x[1])[:k]]

rng = np.random.RandomState(0)
def masked_hit_at_5(s):
    if len(s) < n: 
        return None
    pos = rng.randint(n-1, len(s))  # choose a position with context available
    ctx = ("^"*(n-1) + s)[pos-(n-1):pos]
    true_char = s[pos] if pos < len(s) else "$"
    preds = topk_next(ctx, k=5)
    return 1 if true_char in preds else 0

hits = [h for s in subset if (h := masked_hit_at_5(s)) is not None]
hit_at_5 = float(np.mean(hits))
hit_at_5
```

---

### Solution 7.3 $k=3$ and error vs MolWt

```{code-cell} ipython3
k = 3
Z3 = pca.transform(Xz)[:, :k]
Xz_hat3 = Z3 @ pca.components_[:k, :]
X_hat3 = scaler.inverse_transform(Xz_hat3)
recon3 = pd.DataFrame(X_hat3, columns=desc_cols, index=X_desc_all.index)

rmse3 = {c: mean_squared_error(X_desc_all[c], recon3[c], squared=False) for c in desc_cols}
pd.Series(rmse3).sort_values().round(3)
```

```{code-cell} ipython3
per_mol_err3 = np.sqrt(((X_desc_all.values - recon3.values)**2).mean(axis=1))
molwt = X_desc_all["MolWt"].values
plt.scatter(molwt, per_mol_err3, alpha=0.6)
plt.xlabel("MolWt")
plt.ylabel("Per-molecule RMSE (k=3)")
plt.title("Reconstruction error vs MolWt")
plt.show()
```

---

### Solution 7.4 Noise sweep and alt similarity

```{code-cell} ipython3
from sklearn.metrics.pairwise import euclidean_distances

def top1_rate_cosine(std_noise):
    rng = np.random.RandomState(0)
    Xz_noisy = Xz + rng.normal(0, std_noise, size=Xz.shape)
    S = cosine_similarity(Xz, Xz_noisy)
    return float(np.mean(S.argmax(axis=1) == np.arange(S.shape[0])))

rates = {s: top1_rate_cosine(s) for s in [0.05, 0.1, 0.2, 0.3]}
pd.Series(rates).round(3)
```

```{code-cell} ipython3
def top1_rate_euclid_like(std_noise):
    rng = np.random.RandomState(0)
    Xz_noisy = Xz + rng.normal(0, std_noise, size=Xz.shape)
    D = euclidean_distances(Xz, Xz_noisy)
    S = 1.0/(1.0 + D)
    return float(np.mean(S.argmax(axis=1) == np.arange(S.shape[0])))

rates_e = {s: top1_rate_euclid_like(s) for s in [0.05, 0.1, 0.2, 0.3]}
pd.Series(rates_e).round(3)
```

---

### Solution 7.5 Code map

```{code-cell} ipython3
Z2 = pca.transform(Xz)[:, :2]
plt.scatter(Z2[:,0], Z2[:,1], s=12, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Encoder codes (first 2 PCs)")
plt.show()
```
