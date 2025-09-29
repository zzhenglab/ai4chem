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
 
## Sections 6 and 7: Data augmentation and contrastive pretraining for MOF synthesis
 
This handout walks through a compact self-supervised pipeline tailored to metal–organic framework (MOF) synthesis records. We construct chemistry-aware and process-aware augmentations, train a small encoder with a contrastive objective, then evaluate with a frozen linear probe for success prediction.
 
**Design choices**
- Work on 100 rows so everything runs quickly on CPU. The same code scales to the full dataset by removing the sampling and tuning epochs and batch size.
- Two complementary *views* of each experiment: a **process** view and a **chemistry** view. We apply different perturbations that are plausible under lab variability.
- Use a SimCLR-style objective (NT-Xent), which is an instance of InfoNCE. It encourages invariances that match our augmentations.
 
**Notation**
- A single experiment is a vector $x \in \mathbb{R}^d$ composed of process and chemistry features.
- An encoder $f_\theta$ maps input to an embedding $h = f_\theta(x) \in \mathbb{R}^{d_h}$ and a projection head $g_\phi$ outputs $z = g_\phi(h) \in \mathbb{R}^{d_z}$ for the contrastive loss.
- For each row $i$ we sample two stochastic augmentations to create views $x_i^{(1)}, x_i^{(2)}$ and their projected embeddings $z_i^{(1)}, z_i^{(2)}$.
 
---
 
## 6. Data augmentation for MOF synthesis (tabular + molecular)
 
**Goal.** Build label-free augmentations that keep both **chemistry** and **process** context. We pursue two objectives:
 
1) Create paired *views* of the same experiment for contrastive learning.  
2) Create simple synthetic variations for robustness when we later train a small classifier.
 
### 6.1 Load and subset the MOF dataset
 
We take a random subset of 100 rows to keep the runtime short. If your environment blocks network access, replace the URL with a local path.
 
```python
import numpy as np
import pandas as pd
 
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/mof_yield_dataset.csv"
df_full = pd.read_csv(url)
 
# Keep only the columns we need and create a success label for a later linear probe
need = ["linker_smiles", "temperature", "time_h", "concentration_M", "solvent_DMF", "yield"]
df_full = df_full[need].copy()
df_full = df_full.rename(columns={"yield": "yield_pct"})
 
# Binary success target for later probe: success = (yield > 20)
df_full["success"] = (df_full["yield_pct"] > 20).astype(int)
 
# Fixed seed for reproducible selection
rng = np.random.default_rng(7)
idx = rng.choice(len(df_full), size=100, replace=False)
df_small = df_full.iloc[idx].reset_index(drop=True)
df_small.head()
```
 
We save the 100-row subset for reuse.
 
```python
small_url_hint = "mof_yield_dataset_100.csv"
df_small.to_csv(small_url_hint, index=False)
small_url_hint
```
 
### 6.2 Descriptor and fingerprint features
 
We compute four light descriptors and a 256-bit Morgan fingerprint from the linker SMILES. If RDKit is not present we fall back to NaN descriptors and zero fingerprints so the notebook is still runnable.
 
```python
import warnings
warnings.filterwarnings("ignore")
 
# RDKit guarded import
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    RD = True
except Exception as e:
    RD = False
    Chem = None
    print("RDKit not available in this environment. Molecular descriptors will be set to NaN and fingerprints to zeros.")
 
def calc_desc4(smiles: str):
    if not RD or smiles is None:
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
 
def morgan_bits(smiles: str, n_bits: int = 256, radius: int = 2):
    if not RD or smiles is None:
        return np.zeros(n_bits, dtype=np.int8)
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.int8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(m)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
 
desc = df_small["linker_smiles"].apply(calc_desc4)
FP_BITS = 256
fps = np.vstack(df_small["linker_smiles"].apply(lambda s: morgan_bits(s, n_bits=FP_BITS, radius=2)).values)
 
feat = pd.concat([df_small.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)
for j in range(FP_BITS):
    feat[f"fp_{j}"] = fps[:, j].astype(np.int8)
 
feat.head(3)
```
 
### 6.3 Split into process and chemistry views
 
For augmentation we split features into two views:
- **Process view**: `temperature`, `time_h`, `concentration_M`, `solvent_DMF`  
- **Chemistry view**: `[MolWt, LogP, TPSA, NumRings]` plus 256-bit fingerprint
 
We standardize the numeric parts only. Bits stay binary.
 
```python
from sklearn.preprocessing import StandardScaler
 
proc_cols = ["temperature", "time_h", "concentration_M", "solvent_DMF"]
chem_cols = ["MolWt", "LogP", "TPSA", "NumRings"] + [f"fp_{i}" for i in range(FP_BITS)]
 
X_proc_raw = feat[proc_cols].astype(float).to_numpy()
X_chem_raw = feat[chem_cols].astype(float).to_numpy()
 
sc_proc = StandardScaler().fit(X_proc_raw)
X_proc = sc_proc.transform(X_proc_raw)
 
# For chemistry we standardize only the descriptors, not the binary bits
sc_chem_desc = StandardScaler().fit(feat[["MolWt", "LogP", "TPSA", "NumRings"]])
X_desc4 = sc_chem_desc.transform(feat[["MolWt", "LogP", "TPSA", "NumRings"]])
X_bits  = feat[[f"fp_{i}" for i in range(FP_BITS)]].to_numpy(dtype=float)
 
X_chem = np.hstack([X_desc4, X_bits])
X_proc.shape, X_chem.shape
```
 
### 6.4 Why these augmentations
 
We want augmentations that do not change the underlying identity of the experiment yet still add variability. Think of an invariance set $\mathcal{T}$ of transformations where $t \in \mathcal{T}$ should preserve labels while changing nuisance factors. Contrastive learning tries to make the encoder $f_\theta$ invariant to such $t$ by pulling the two augmented views together.
 
- **Gaussian jitter (process)**: after standardization, add zero-mean noise to simulate setpoint or measurement variance. If $x$ is standardized, $x' = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.
- **Mask-and-impute (process)**: randomly set one standardized dimension to zero (mean in original space), which mimics missing logs.
- **Descriptor jitter (chemistry)**: add tiny noise to `[MolWt, LogP, TPSA, NumRings]` to model computational variance or wet-lab tolerance.
- **Fingerprint dropout (chemistry)**: independently drop a small fraction of bits to represent imperfect perception and force robustness. For bits $b \in \{0,1\}$, with dropout prob $p$, we set $\tilde{b} = b \cdot \mathbb{1}\{u > p\}$ where $u \sim \text{Uniform}(0,1)$.
- **MixUp (for supervised probe)**: for tabular vectors $x_i, x_j$ we synthesize $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, with $\lambda \sim \text{Beta}(\alpha, \alpha)$. This is not used to create positive pairs since it blends identities.
 
```python
from dataclasses import dataclass
 
@dataclass
class AugmentParams:
    proc_noise_std: float = 0.05         # standard dev for Gaussian jitter in z-space
    proc_mask_p: float = 0.05            # chance to mask a column and impute
    chem_desc_noise_std: float = 0.02    # tiny noise on 4 descriptors
    fp_dropout_p: float = 0.02           # independent bit dropout prob
 
P_WEAK = AugmentParams(proc_noise_std=0.03, proc_mask_p=0.02, chem_desc_noise_std=0.01, fp_dropout_p=0.01)
P_STRONG = AugmentParams(proc_noise_std=0.10, proc_mask_p=0.10, chem_desc_noise_std=0.05, fp_dropout_p=0.05)
P_DEFAULT = P_WEAK
```
 
### 6.5 Augmentation operators
 
We provide two operators:
- `augment_for_contrastive`: create two correlated views of the same row  
- `mixup_rows`: optional synthetic rows for supervised robustness
 
```python
def aug_proc_once(x_proc_z, params: AugmentParams, rng: np.random.Generator):
    z = x_proc_z.copy()
    # Gaussian jitter
    z = z + rng.normal(0.0, params.proc_noise_std, size=z.shape)
    # Mask-and-impute
    if rng.uniform() < params.proc_mask_p:
        j = rng.integers(low=0, high=z.shape[0])
        z[j] = 0.0   # zero in standardized space equals mean in original space
    return z
 
def aug_chem_once(x_chem, params: AugmentParams, rng: np.random.Generator):
    x = x_chem.copy()
    # first 4 are descriptors in z-space, remainder are 0/1 fingerprint bits
    x[:4] = x[:4] + rng.normal(0.0, params.chem_desc_noise_std, size=4)
    bits = x[4:].copy()
    if params.fp_dropout_p > 0:
        mask = rng.uniform(size=bits.shape[0]) < params.fp_dropout_p
        bits[mask] = 0.0
    x[4:] = bits
    return x
 
def augment_for_contrastive(X_proc_z, X_chem_full, params=P_DEFAULT, seed=0):
    rng = np.random.default_rng(seed)
    n = X_proc_z.shape[0]
    # produce two views v1 and v2 for each i
    v1_proc = np.vstack([aug_proc_once(X_proc_z[i], params, rng) for i in range(n)])
    v2_proc = np.vstack([aug_proc_once(X_proc_z[i], params, rng) for i in range(n)])
    v1_chem = np.vstack([aug_chem_once(X_chem_full[i], params, rng) for i in range(n)])
    v2_chem = np.vstack([aug_chem_once(X_chem_full[i], params, rng) for i in range(n)])
    # concatenate process and chemistry views
    v1 = np.hstack([v1_proc, v1_chem])
    v2 = np.hstack([v2_proc, v2_chem])
    return v1, v2
 
def mixup_rows(X, y=None, alpha=0.4, n_new=100, seed=0):
    rng = np.random.default_rng(seed)
    m, d = X.shape
    rows = []
    ys = []
    for _ in range(n_new):
        i, j = rng.integers(0, m), rng.integers(0, m)
        lam = rng.beta(alpha, alpha)
        rows.append(lam * X[i] + (1.0 - lam) * X[j])
        if y is not None:
            ys.append(lam * y[i] + (1.0 - lam) * y[j])
    if y is None:
        return np.vstack(rows)
    return np.vstack(rows), np.array(ys)
```
 
### 6.6 Inspect the effect of augmentation
 
A quick histogram check shows that augmented standardized values remain close to the base distribution.
 
```python
import matplotlib.pyplot as plt
 
v1, v2 = augment_for_contrastive(X_proc, X_chem, params=P_DEFAULT, seed=123)
 
plt.figure()
plt.hist(X_proc[:, 0], bins=20, alpha=0.6)
plt.hist(v1[:, 0], bins=20, alpha=0.6)
plt.title("Process feature 0: original vs augmented")
plt.xlabel("z value")
plt.ylabel("count")
plt.show()
```
 
We also compare PCA (or t-SNE) embeddings of originals vs one augmented view. Colors indicate success vs failure of the original row.
 
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
 
def embed_2d(X, use_tsne=False, random_state=0):
    if use_tsne:
        reducer = TSNE(n_components=2, perplexity=20, learning_rate="auto",
                       init="pca", random_state=random_state)
    else:
        reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(X)
 
# Build a single augmented view for comparison
v1, v2 = augment_for_contrastive(X_proc, X_chem, params=P_DEFAULT, seed=123)
 
X_orig = np.hstack([X_proc, X_chem]).astype(float)
X_aug  = v1.astype(float)
 
use_tsne = False  # True to use t-SNE instead of PCA
Z0 = embed_2d(X_orig, use_tsne=use_tsne, random_state=0)
Z1 = embed_2d(X_aug,  use_tsne=use_tsne, random_state=0)
 
# Success/failure masks for originals (same order as df_small)
y = feat["success"].to_numpy().astype(int)
m_succ = (y == 1)
m_fail = (y == 0)
 
plt.figure()
# original
plt.scatter(Z0[m_succ,0], Z0[m_succ,1], s=18, color= "lightblue", alpha=0.8, marker="o", label="orig success")
plt.scatter(Z0[m_fail,0], Z0[m_fail,1], s=18, color= "blue", alpha=0.6, marker="x", label="orig failure")
# augmented single view (labels align with originals)
plt.scatter(Z1[m_succ,0], Z1[m_succ,1], s=16, color= "yellow", alpha=0.8, marker="^", label="aug success")
plt.scatter(Z1[m_fail,0], Z1[m_fail,1], s=16, color= "red",alpha=0.6, marker="s", label="aug failure")
 
plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.title(("t-SNE" if use_tsne else "PCA") + " of original vs augmented (success vs failure)")
plt.legend()
plt.show()
```
 
We can make a larger augmented pool per row and visualize its geometry relative to the originals.
 
```python
def make_augmented_pool(Xp, Xc, params=P_DEFAULT, n_aug_per_row=3, seed=0):
    # Create n_aug_per_row augmented samples for each original row.
    # Returns:
    #   X_aug_pool: [n * n_aug_per_row, d] augmented features
    #   src_idx:    [n * n_aug_per_row] index of the original row for each augmented sample
    rng = np.random.default_rng(seed)
    n = Xp.shape[0]
    rows = []
    src = []
    for i in range(n):
        for _ in range(n_aug_per_row):
            xpp = aug_proc_once(Xp[i], params, rng)
            xcc = aug_chem_once(Xc[i], params, rng)
            rows.append(np.hstack([xpp, xcc]))
            src.append(i)
    return np.vstack(rows).astype(np.float32), np.array(src, dtype=int)
 
# Example: build 5 augmented copies per row and visualize with PCA
X_aug5, src5 = make_augmented_pool(X_proc, X_chem, params=P_DEFAULT, n_aug_per_row=5, seed=7)
Z5 = embed_2d(X_aug5, use_tsne=False, random_state=0)
 
# Labels for originals and the 5x pool
y = feat["success"].to_numpy().astype(int)
m_succ = (y == 1)
m_fail = (y == 0)
 
y5 = y[src5]              # labels for each augmented sample via its source row
m5_succ = (y5 == 1)
m5_fail = (y5 == 0)
 
plt.figure()
# original
plt.scatter(Z0[m_succ,0], Z0[m_succ,1], s=18, color= "lightblue", alpha=0.8, marker="o", label="orig success")
plt.scatter(Z0[m_fail,0], Z0[m_fail,1], s=18, color= "blue",alpha=0.6, marker="x", label="orig failure")
# 5× augmented pool
plt.scatter(Z5[m5_succ,0], Z5[m5_succ,1], s=10, color= "yellow", alpha=0.45, marker="^", label="aug x5 success")
plt.scatter(Z5[m5_fail,0], Z5[m5_fail,1], s=10, color= "red", alpha=0.45, marker="s", label="aug x5 failure")
 
plt.xlabel("dim 1"); plt.ylabel("dim 2")
plt.title("PCA of original vs 5× augmented pool (success vs failure)")
plt.legend()
plt.show()
```
 
### 6.7 Math for contrastive pairs
 
For each example $x_i$ we sample two views $x_i^{(1)} = t_1(x_i)$ and $x_i^{(2)} = t_2(x_i)$ with $t_1, t_2 \sim \mathcal{T}$. A positive pair is $\big(x_i^{(1)}, x_i^{(2)}\big)$. In a batch of size $B$ we get $2B$ projected embeddings
\[
z_i^{(1)} = g_\phi\!\left(f_\theta\!\big(x_i^{(1)}\big)\right), \quad
z_i^{(2)} = g_\phi\!\left(f_\theta\!\big(x_i^{(2)}\big)\right).
\]
We use cosine similarity
\[
\operatorname{sim}(u, v) \;=\; \frac{u^\top v}{\|u\|\;\|v\|}
\]
and the NT-Xent loss (InfoNCE) with temperature $\tau > 0$:
\[
\ell(u, v) \;=\; -\log \frac{\exp\!\big(\operatorname{sim}(u,v)/\tau\big)}
{\sum\limits_{w \in \mathcal{A}(u)} \exp\!\big(\operatorname{sim}(u,w)/\tau\big)}.
\]
$\mathcal{A}(u)$ collects all other embeddings in the batch excluding $u$. The symmetric objective sums over anchors in both views. With sufficient negatives, this loss increases a lower bound on mutual information between the two views and encourages $f_\theta$ to learn invariances matched to $\mathcal{T}$.
 
---
 
## 7. Contrastive pretraining and a linear probe
 
We train a tiny MLP encoder $f_\theta$ on augmented pairs using NT-Xent. We then **freeze** the encoder and fit a logistic regression probe on the frozen features $H$ to predict **success** (yield > 20). The probe is a small supervised head that tests whether the representation transfers to the downstream task.
 
### 7.1 Torch setup and data wrappers
 
```python
import math, sys, os, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Pack the base matrix that the encoder consumes
X_base = np.hstack([X_proc, X_chem]).astype(np.float32)
y_success = feat["success"].to_numpy().astype(np.int64)
X_base.shape, y_success.shape
```
 
```python
class ContrastivePairs(Dataset):
    def __init__(self, X_proc, X_chem, params: AugmentParams, seed=0):
        self.Xp = X_proc.astype(np.float32)
        self.Xc = X_chem.astype(np.float32)
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
 
    def __len__(self):
        return self.Xp.shape[0]
 
    def __getitem__(self, idx):
        # on-the-fly augmentations for a single index
        x1p = aug_proc_once(self.Xp[idx], self.params, self.rng)
        x2p = aug_proc_once(self.Xp[idx], self.params, self.rng)
        x1c = aug_chem_once(self.Xc[idx], self.params, self.rng)
        x2c = aug_chem_once(self.Xc[idx], self.params, self.rng)
        x1 = np.hstack([x1p, x1c]).astype(np.float32)
        x2 = np.hstack([x2p, x2c]).astype(np.float32)
        return torch.from_numpy(x1), torch.from_numpy(x2)
 
params_train = P_DEFAULT
train_set = ContrastivePairs(X_proc, X_chem, params_train, seed=13)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
len(train_set)
```
 
### 7.2 Encoder and projection head
 
The encoder maps the concatenated feature vector to a latent space. The projection head is used for the contrastive objective only. For the linear probe we use the encoder output before the head.
 
```python
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hid=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out),
        )
 
    def forward(self, x):
        return self.net(x)
 
class Projector(nn.Module):
    def __init__(self, in_dim, hid=256, out=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, out),
            )
    def forward(self, x):
        return self.net(x)
 
in_dim = X_base.shape[1]
enc = MLPEncoder(in_dim, hid=256, out=128).to(device)
proj = Projector(128, hid=128, out=64).to(device)
 
sum(p.numel() for p in enc.parameters()), sum(p.numel() for p in proj.parameters())
```
 
### 7.3 NT-Xent loss
 
```python
def nt_xent(z1, z2, temp=0.2, eps=1e-8):
    """
    z1, z2: tensors of shape [B, D]
    returns scalar loss
    """
    B, D = z1.shape
    # L2 normalize
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + eps)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + eps)
 
    reps = torch.cat([z1, z2], dim=0)                              # [2B, D]
    sim = torch.mm(reps, reps.t())                                 # cosine since normalized
    # mask self-similarity
    mask = torch.eye(2*B, dtype=torch.bool, device=reps.device)
    sim = sim.masked_fill(mask, -9e15)
 
    # positives: for i in [0..B-1], pos for i is i+B; for i+B, pos is i
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(reps.device)
    logits = sim / temp
    labels = pos
 
    # cross entropy over rows
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
```
 
### 7.4 Train the contrastive model
 
We keep epochs small to stay CPU friendly. Watch the training curve for stability.
 
```python
opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-5)
 
EPOCHS = 20
loss_hist = []
 
enc.train(); proj.train()
for ep in range(EPOCHS):
    run = []
    for (x1, x2) in train_loader:
        x1 = x1.to(device); x2 = x2.to(device)
        h1 = enc(x1); h2 = enc(x2)
        z1 = proj(h1); z2 = proj(h2)
        loss = nt_xent(z1, z2, temp=0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run.append(loss.item())
    loss_hist.append(np.mean(run))
 
import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_hist, marker="o")
plt.xlabel("epoch")
plt.ylabel("contrastive loss")
plt.title("NT-Xent training loss")
plt.grid(True)
plt.show()
```
 
### 7.5 Build representations and visualize
 
We compute embeddings using the encoder output before the projection head. Then we use PCA for a quick 2D visualization.
 
```python
enc.eval()
with torch.no_grad():
    H = enc(torch.from_numpy(X_base).to(device)).cpu().numpy()
 
from sklearn.decomposition import PCA
Z = PCA(n_components=2, random_state=0).fit_transform(H)
 
plt.figure()
plt.scatter(Z[:,0], Z[:,1], s=20, alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Encoder representations (PCA 2D)")
plt.show()
```
 
### 7.6 Linear probe: success vs failure
 
Freeze the encoder, then train a logistic regression on the frozen features $H$ using 5-fold stratified splits. We report accuracy, F1, and ROC AUC.
 
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs, f1s, aucs = [], [], []
 
for tr, te in skf.split(H, y_success):
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(H[tr], y_success[tr])
    p = clf.predict(H[te])
    proba = clf.predict_proba(H[te])[:,1]
    accs.append(accuracy_score(y_success[te], p))
    f1s.append(f1_score(y_success[te], p))
    try:
        aucs.append(roc_auc_score(y_success[te], proba))
    except Exception:
        aucs.append(np.nan)
 
print("Linear probe on frozen encoder")
print("acc mean:", round(np.mean(accs), 3), "±", round(np.std(accs), 3))
print("f1  mean:", round(np.mean(f1s), 3), "±", round(np.std(f1s), 3))
print("auc mean:", round(np.nanmean(aucs), 3))
```
 
### 7.7 Baseline without pretraining
 
We compare against a logistic regression trained directly on the raw standardized features. This clarifies whether contrastive pretraining helps in small data.
 
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
X_raw = X_base.copy()
 
accs0, f1s0, aucs0 = [], [], []
for tr, te in skf.split(X_raw, y_success):
    base_clf = LogisticRegression(max_iter=200, solver="lbfgs")
    base_clf.fit(X_raw[tr], y_success[tr])
    p0 = base_clf.predict(X_raw[te])
    proba0 = base_clf.predict_proba(X_raw[te])[:,1]
    accs0.append(accuracy_score(y_success[te], p0))
    f1s0.append(f1_score(y_success[te], p0))
    try:
        aucs0.append(roc_auc_score(y_success[te], proba0))
    except Exception:
        aucs0.append(np.nan)
 
print("Baseline logistic on raw features")
print("acc mean:", round(np.mean(accs0), 3), "±", round(np.std(accs0), 3))
print("f1  mean:", round(np.mean(f1s0), 3), "±", round(np.std(f1s0), 3))
print("auc mean:", round(np.nanmean(aucs0), 3))
```
 
### 7.8 Optional: training-time augmentation multiplicity for the probe
 
Sometimes repeating augmented copies of the training rows while fitting the probe improves stability. We keep the encoder frozen and only enlarge the probe training set with encoded augmentations of the training fold.
 
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import torch
 
def eval_probe_with_augmult(enc, X_proc, X_chem, y, n_aug_per_row=1, params=P_DEFAULT, n_splits=5, seed=0):
    """
    Compare a linear probe trained with augmented training data.
    - enc: frozen encoder (torch nn.Module)
    - X_proc, X_chem: base inputs used to build augmentations
    - y: binary labels (success)
    - n_aug_per_row: how many augmented copies per training row (0 means no augmentation)
    Returns mean and std of acc, f1, auc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    A, F, U = [], [], []
    with torch.no_grad():
        X_base = np.hstack([X_proc, X_chem]).astype(np.float32)
        H_all = enc(torch.from_numpy(X_base).to(device)).cpu().numpy()
 
    for tr, te in skf.split(H_all, y):
        Htr, ytr = H_all[tr], y[tr]
        Hte, yte = H_all[te], y[te]
 
        if n_aug_per_row > 0:
            # Build augmented pool *only* from the training rows
            Xp_tr = X_proc[tr]
            Xc_tr = X_chem[tr]
            X_aug_pool, src_idx = make_augmented_pool(Xp_tr, Xc_tr, params=params, n_aug_per_row=n_aug_per_row, seed=seed+7)
            # Encode the augmented rows
            with torch.no_grad():
                H_aug = enc(torch.from_numpy(X_aug_pool).to(device)).cpu().numpy()
            y_aug = np.repeat(ytr, n_aug_per_row)
            Htr = np.vstack([Htr, H_aug])
            ytr = np.concatenate([ytr, y_aug])
 
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Htr, ytr)
        p = clf.predict(Hte)
        proba = clf.predict_proba(Hte)[:,1]
        A.append(accuracy_score(yte, p))
        F.append(f1_score(yte, p))
        try:
            U.append(roc_auc_score(yte, proba))
        except Exception:
            U.append(np.nan)
    return (np.mean(A), np.std(A)), (np.mean(F), np.std(F)), np.nanmean(U)
 
# Baseline: no augmented rows, then 1x and 5x
enc.eval()
baseline = eval_probe_with_augmult(enc, X_proc, X_chem, y_success, n_aug_per_row=0, n_splits=5, seed=11)
one_x    = eval_probe_with_augmult(enc, X_proc, X_chem, y_success, n_aug_per_row=1, n_splits=5, seed=11)
five_x   = eval_probe_with_augmult(enc, X_proc, X_chem, y_success, n_aug_per_row=5, n_splits=5, seed=11)
 
print("Linear probe with augmentation multiplicity")
for name, res in [("0x (no aug)", baseline), ("1x", one_x), ("5x", five_x)]:
    (acc_m, acc_s), (f1_m, f1_s), auc_m = res
    print(f"{name:10s} | acc {acc_m:.3f} ± {acc_s:.3f} | f1 {f1_m:.3f} ± {f1_s:.3f} | auc {auc_m:.3f}")
```
 
---
 
## Practical notes and equations at a glance
 
- **Temperature** $\tau$: controls the softness of the softmax in NT-Xent. Larger $\tau$ reduces repulsion, smaller $\tau$ increases it. Try $\tau \in \{0.1, 0.2, 0.5\}$.
- **Batch size** $B$: more negatives generally help NT-Xent since each anchor sees $2B-2$ negatives. With small CPU batches keep expectations modest.
- **Mask-and-impute**: in standardized space, setting a coordinate to zero corresponds to imputing the training mean.
- **Frozen probe**: we evaluate $f_\theta$ by training a small linear classifier on $H = f_\theta(X)$ only. If probe performance beats the baseline on raw features, the representation is more linearly separable for the task.
- **Why projection head** $g_\phi$: empirically helps the encoder avoid collapsing to trivial directions used by the loss. At evaluation we drop $g_\phi$ and use $h$.
 
---
 
**Summary**
- Augmentations create two friendly views per experiment and small supervised variants.  
- Contrastive training organizes the space without labels using NT-Xent.  
- A tiny probe on the frozen encoder can separate success vs failure with limited data.  
- To scale up, remove the 100-row sampling and tune epochs and batch size.
