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

# Lecture 9 - Graph Neural Networks (Chemistry)

```{contents}
:local:
:depth: 1
```

## Learning goals

- Build basic MLP-style Neural Networks from scratch using **PyTorch**.  
- Explain molecules as **graphs**: atoms = nodes, bonds = edges, features at each.  
- Write the **message passing** equations and understand neighborhood aggregation.  
- Build a tiny GNN in PyTorch on toy molecules.  
- Prepare molecular **graphs from SMILES** and run a mini GNN.  
- Experience with a MPNN example **Chemprop**.  

---

## 1) Setup

We reuse most of the stack from earlier lectures. 

```{code-cell} ipython3
# 1. Setup
import warnings, math, os, sys, json, time, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn bits for splitting and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

# Torch for MLP and GNN
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    print("Installing torch, this may take a minute...")
    %pip -q install torch
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

# RDKit is optional
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except Exception:
    Chem = None  # we will use toy graphs if RDKit is unavailable

# A small global seed helper
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_seed(0)
```

---

## 2) From MLP (Lecture 8) to PyTorch MLP (recap)

In Lecture 8 we built MLPs with scikit‑learn. Today we start by doing the same with **PyTorch** so that the later GNN will feel familiar.

We will predict **melting point** from four descriptors: `MolWt`, `LogP`, `TPSA`, `NumRings`, same as before.

### 2.1 Load the same CSV and compute descriptors quickly (RDKit optional)

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)
df_raw.head(3)
```

If RDKit is available we recompute 4 descriptors. If not, we will use the columns already present in the CSV.

```{code-cell} ipython3
def quick_desc(smiles):
    if Chem is None:
        # If RDKit not present, return NaNs to fall back to existing columns
        return pd.Series({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan})
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return pd.Series({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan})
    # Minimal set computed from RDKit's atom/bond info without heavy deps
    mw = rdchem.CalcExactMolWt(m) if hasattr(rdchem, "CalcExactMolWt") else np.nan
    # Fallbacks if helpers are missing
    logp = float(np.nan)
    tpsa = float(np.nan)
    rings = Chem.GetSSSR(m) if hasattr(Chem, "GetSSSR") else np.nan
    return pd.Series({"MolWt": mw, "LogP": logp, "TPSA": tpsa, "NumRings": rings})

# Try to add descriptors; if they already exist with numbers, we will use them
maybe_desc = df_raw["SMILES"].head(3).apply(quick_desc)  # quick probe
maybe_desc.head()
```

To keep the class portable, we will **prefer the 4 columns already in the dataset** if they are numeric.

```{code-cell} ipython3
cols = ["MolWt", "LogP", "TPSA", "NumRings", "Melting Point"]
df_reg = df_raw[cols].dropna()
X = df_reg[["MolWt", "LogP", "TPSA", "NumRings"]].values.astype(np.float32)
y = df_reg["Melting Point"].values.astype(np.float32).reshape(-1, 1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr).astype(np.float32), scaler.transform(Xte).astype(np.float32)

Xtr_s.shape, ytr.shape
```

### 2.2 Define a tiny MLP in PyTorch

We keep one hidden layer to keep the shape clear.

```{code-cell} ipython3
class TinyMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=32, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out

mlp = TinyMLP(in_dim=Xtr_s.shape[1], hidden=32, out_dim=1)
mlp
```

Check the parameter shapes so you see what is being learned.

```{code-cell} ipython3
for name, p in mlp.named_parameters():
    print(name, tuple(p.shape))
```

### 2.3 One training epoch by hand

We will implement a plain loop: forward → loss → backward → update.

```{code-cell} ipython3
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)

xb = torch.from_numpy(Xtr_s[:64])
yb = torch.from_numpy(ytr[:64])
pred = mlp(xb)
loss = loss_fn(pred, yb)
opt.zero_grad(); loss.backward(); opt.step()
float(loss.item())
```

### 2.4 Full training with a DataLoader

```{code-cell} ipython3
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(NumpyDataset(Xtr_s, ytr), batch_size=64, shuffle=True)

mlp = TinyMLP(in_dim=Xtr_s.shape[1], hidden=32, out_dim=1)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)

train_losses = []
for epoch in range(150):
    batch_losses = []
    for xb, yb in train_loader:
        pred = mlp(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

plt.plot(train_losses); plt.xlabel("epoch"); plt.ylabel("train MSE"); plt.title("MLP training curve"); plt.grid(alpha=0.3)
plt.show()
```

Evaluate on test.

```{code-cell} ipython3
mlp.eval()
with torch.no_grad():
    yhat = mlp(torch.from_numpy(Xte_s)).numpy()

print(f"MSE: {mean_squared_error(yte, yhat):.2f}")
print(f"MAE: {mean_absolute_error(yte, yhat):.2f}")
print(f"R2 : {r2_score(yte, yhat):.3f}")

plt.scatter(yte, yhat, alpha=0.6)
lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
plt.plot(lims, lims, "k--"); plt.xlabel("True MP"); plt.ylabel("Pred MP"); plt.title("MLP parity")
plt.show()
```

```{admonition} ⏰ Exercises 2.x
1) Change the hidden size to 64 and rerun. Does R² improve.  
2) Switch `ReLU` to `Tanh` and rerun. Compare training curve shape.  
3) Increase `weight_decay` to `1e-2`. What happens to train vs test scores.  
```

---

## 3) Molecules as graphs

A molecule can be seen as a graph:  
- **Nodes** are atoms with a feature vector per atom.  
- **Edges** are bonds that carry types such as single, double, aromatic.  
- A molecule level label (melting point, toxicity) requires **pooling** node representations into one vector.

### 3.1 Minimal node and edge features

Common node features:
- One‑hot element type, degree, formal charge, aromatic flag.  
Common edge features:
- Bond type one‑hot: single, double, triple, aromatic.

We will assemble a small structure that holds:
- `x`: node feature matrix, shape `[n_nodes, d_node]`  
- `edge_index`: list of edges as two rows `[2, n_edges]`  
- `edge_attr`: edge features `[n_edges, d_edge]` (optional)  
- `y`: target for the graph

### 3.2 Build tiny toy graphs by hand

We start with two toy molecules without RDKit: Methane and Ethane.

```{code-cell} ipython3
def toy_methane():
    # C with 4 H; simple graph centered on C
    # nodes: 0=C, 1..4=H
    x = np.array([
        [6, 4, 0, 0],  # very small features: [atomic_number, degree, is_aromatic, formal_charge]
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    ], dtype=np.float32)
    edges = []
    for h in [1,2,3,4]:
        edges += [(0, h), (h, 0)]
    edge_index = np.array(edges, dtype=np.int64).T  # shape [2, 8]
    return {"x": x, "edge_index": edge_index}

def toy_ethane():
    # C-C with 3 H on each carbon
    x = np.array([
        [6, 4, 0, 0], [6, 4, 0, 0],  # two carbons
        [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],  # Hs on C0
        [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],  # Hs on C1
    ], dtype=np.float32)
    edges = [(0,1),(1,0)]
    for h in [2,3,4]: edges += [(0,h),(h,0)]
    for h in [5,6,7]: edges += [(1,h),(h,1)]
    edge_index = np.array(edges, dtype=np.int64).T
    return {"x": x, "edge_index": edge_index}

g1, g2 = toy_methane(), toy_ethane()
[g1["x"].shape, g1["edge_index"].shape, g2["x"].shape, g2["edge_index"].shape]
```

Peek at arrays to make sure the shapes are what you expect.

```{code-cell} ipython3
print("Methane x:\n", g1["x"])
print("Methane edges (first 6 cols):\n", g1["edge_index"][:, :6])
```

```{admonition} Tip
Large SMILES → RDKit graphs come later. For now the goal is to **see** the shapes and write a message passing layer on these toy graphs.
```

```{admonition} ⏰ Exercises 3.x
Add a new toy graph for **propane** with indices `[0,1,2]` as the carbon chain and correct hydrogens. Return the same keys `x` and `edge_index`. Print its shapes.
```

---

## 4) Message passing: the core idea

At each layer `t`, every node gets messages from its neighbors and updates its hidden state.

A simple form:

$$
h_v^{(t+1)} = \sigma\left(W_\text{self}\, h_v^{(t)} + \sum_{u \in \mathcal{N}(v)} W_\text{nei}\, h_u^{(t)} + b\right)
$$

- `h_v^{(t)}` is the node vector at layer `t`.  
- `W_self` maps the node to itself.  
- `W_nei` maps neighbor messages.  
- Sum neighbor messages, then apply a nonlinearity `σ` such as ReLU.

After `T` layers, pool all node vectors to get a graph vector:

$$
h_\text{graph} = \text{POOL}\left(\{h_v^{(T)} : v \in \text{nodes}\}\right)
$$

POOL can be **sum**, **mean**, or **max**. For regression we feed `h_graph` to a linear head to predict a scalar.

---

## 5) Your first GNN layer in pure PyTorch

We implement a very small GNN layer that follows the equation above. No external graph library is needed.

### 5.1 A neighbor sum helper

```{code-cell} ipython3
def neighbor_sum(x, edge_index):
    """
    x: [N, d] node features
    edge_index: [2, E] with rows [src, dst]
    returns: [N, d] sum of neighbor features for each node (incoming edges)
    """
    N, d = x.shape
    out = torch.zeros_like(x)
    src, dst = edge_index
    out.index_add_(0, dst, x[src])  # sum x[src] into row dst
    return out
```

### 5.2 Define one message passing layer

```{code-cell} ipython3
class SimpleMP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)
        self.nei_lin  = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        # x: [N, in_dim], edge_index: tensor [2, E]
        nei_agg = neighbor_sum(x, edge_index)           # [N, in_dim]
        out = self.self_lin(x) + self.nei_lin(nei_agg)  # [N, out_dim]
        return self.act(out)
```

### 5.3 Stack layers and pool to a graph vector

```{code-cell} ipython3
class TinyGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, pool="mean"):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(SimpleMP(dim, hidden))
            dim = hidden
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, out_dim)
        self.pool = pool

    def readout(self, x):
        if self.pool == "sum":  return x.sum(dim=0, keepdim=True)
        if self.pool == "max":  return x.max(dim=0, keepdim=True).values
        return x.mean(dim=0, keepdim=True)  # default mean

    def forward(self, x, edge_index):
        for mp in self.layers:
            x = mp(x, edge_index)
        hg = self.readout(x)         # [1, hidden]
        y = self.head(hg)            # [1, out_dim]
        return y  # graph-level output
```

### 5.4 Sanity check with toy graphs

We will make up fake targets for the toys to see the forward pass shape.

```{code-cell} ipython3
def to_tensor_graph(g):
    x = torch.from_numpy(g["x"])
    edge_index = torch.from_numpy(g["edge_index"])
    return x, edge_index

x1, e1 = to_tensor_graph(g1)
x2, e2 = to_tensor_graph(g2)

gnn = TinyGNN(in_dim=x1.shape[1], hidden=16, out_dim=1, num_layers=2, pool="mean")
with torch.no_grad():
    y1 = gnn(x1, e1)
    y2 = gnn(x2, e2)
print("Toy outputs:", y1.shape, y2.shape)
```

### 5.5 Train on toy data (regression on a made‑up scalar)

Let us define a toy label such as “number of heavy atoms”. This is only for demonstration.

```{code-cell} ipython3
def heavy_atom_count(g):
    # heavy atoms = atomic_number > 1
    return float((g["x"][:,0] > 1).sum())

train_graphs = [toy_methane(), toy_ethane()]
y_train = torch.tensor([[heavy_atom_count(g)] for g in train_graphs], dtype=torch.float32)

gnn = TinyGNN(in_dim=train_graphs[0]["x"].shape[1], hidden=16, out_dim=1, num_layers=2, pool="sum")
opt = torch.optim.Adam(gnn.parameters(), lr=5e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

losses = []
for epoch in range(200):
    epoch_loss = 0.0
    for g, ytrue in zip(train_graphs, y_train):
        x, ei = to_tensor_graph(g)
        pred = gnn(x, ei)
        loss = loss_fn(pred, ytrue.view(1,1))
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += float(loss.item())
    losses.append(epoch_loss/len(train_graphs))

plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Toy GNN regression loss")
plt.grid(alpha=0.3); plt.show()

for g in train_graphs:
    with torch.no_grad():
        p = gnn(*to_tensor_graph(g)).item()
    print("pred", round(p,3), "target", heavy_atom_count(g))
```

```{admonition} What you saw
- Node vectors flowed through two message passing layers.  
- A readout pooled node states to a graph vector.  
- A tiny head predicted a scalar.  
```

```{admonition} ⏰ Exercises 5.x
1) Switch pooling from `"sum"` to `"mean"` and retrain. Compare convergence.  
2) Replace `ReLU` with `Tanh` inside `SimpleMP`. Does the curve change.  
3) Change the target to “number of hydrogens” and retrain.  
```

---

## 6) From SMILES to graphs with RDKit (mini pipeline)

Now we prepare graphs from real SMILES. If RDKit is not available in your environment, skip to Section 7 (Chemprop), which does not require RDKit.

### 6.1 Featurization helpers

```{code-cell} ipython3
def atom_features(atom):
    # one short vector per atom; keep it tiny for speed
    return np.array([
        atom.GetAtomicNum(),                 # Z
        atom.GetTotalDegree(),               # degree
        int(atom.GetIsAromatic()),           # aromatic
        atom.GetFormalCharge()               # charge
    ], dtype=np.float32)

def bond_pairs(mol):
    src, dst = [], []
    for b in mol.GetBonds():
        a = b.GetBeginAtomIdx()
        c = b.GetEndAtomIdx()
        src += [a, c]
        dst += [c, a]
    return np.array([src, dst], dtype=np.int64)

def smiles_to_graph(smi):
    if Chem is None:
        raise RuntimeError("RDKit not available. Please skip to Section 7 (Chemprop).")
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = np.vstack([atom_features(a) for a in mol.GetAtoms()])  # [N,d]
    ei = bond_pairs(mol)                                       # [2,E]
    return {"x": x, "edge_index": ei}
```

### 6.2 Make a small dataset from the CSV

We will try to regress **melting point** as a quick demo with the tiny GNN. This is only to show the flow; Chemprop will do a stronger job later.

```{code-cell} ipython3
if Chem is not None:
    small = df_raw[["SMILES","Melting Point"]].dropna().sample(200, random_state=0)
    graphs, targets = [], []
    for smi, mp in zip(small["SMILES"], small["Melting Point"]):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g); targets.append(mp)
    y_all = torch.tensor(targets, dtype=torch.float32).view(-1,1)
    len(graphs), y_all.shape
else:
    print("RDKit missing. Skip to Section 7.")
```

### 6.3 Tiny training loop for graphs

We split by index since graphs are already constructed.

```{code-cell} ipython3
if Chem is not None and len(graphs) > 20:
    idx = np.arange(len(graphs))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)
    gnn = TinyGNN(in_dim=graphs[0]["x"].shape[1], hidden=32, out_dim=1, num_layers=3, pool="mean")
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(60):
        batch = np.random.choice(tr_idx, size=min(64, len(tr_idx)), replace=False)
        epoch_loss = 0.0
        for i in batch:
            x = torch.from_numpy(graphs[i]["x"])
            ei = torch.from_numpy(graphs[i]["edge_index"])
            pred = gnn(x, ei)
            loss = loss_fn(pred, y_all[i:i+1])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss/len(batch))

    plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Tiny GNN on melting point")
    plt.grid(alpha=0.3); plt.show()

    # Evaluate
    y_true, y_pred = [], []
    gnn.eval()
    with torch.no_grad():
        for i in te_idx:
            p = gnn(torch.from_numpy(graphs[i]["x"]),
                    torch.from_numpy(graphs[i]["edge_index"])).item()
            y_true.append(float(y_all[i]))
            y_pred.append(p)

    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R2 : {r2_score(y_true, y_pred):.3f}")

    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(lims, lims, "k--"); plt.xlabel("True MP"); plt.ylabel("Pred MP"); plt.title("Tiny GNN parity")
    plt.show()
```

```{admonition} Takeaway
This simple GNN is **minimal**. Real molecular GNNs use richer atom and bond features, multiple heads, dropout, and careful training schedules.
```

```{admonition} ⏰ Exercises 6.x
1) Change `num_layers` from 3 to 2 and 4. Record test R² for each.  
2) Change pooling from mean to sum. Does performance change.  
3) Add `nn.Dropout(p=0.1)` after each `SimpleMP` in `TinyGNN` and test again.  
```

---

## 7) Chemprop v2: practical graph models for chemistry

Chemprop implements a directed message passing neural network with strong defaults. We will:

1) **Install Chemprop v2**  
2) Run a **melting point** regression  
3) Run a **reactivity** classification and predict on new SMILES

### 7.1 Install Chemprop

```{code-cell} ipython3
# You may need a restart after install in some environments
%pip -q install chemprop
```

### 7.2 Melting point regression

Prepare a minimal CSV: `SMILES,Melting Point`.

```{code-cell} ipython3
# Load data and write a small CSV
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df = pd.read_csv(url)
reg_cols = ["SMILES", "Melting Point"]
df_reg = df[reg_cols].dropna().copy()
df_reg.head(3)
```

Save to disk for Chemprop CLI.

```{code-cell} ipython3
df_reg.to_csv("mp_data.csv", index=False)
len(df_reg), df_reg.head(2)
```

Train a **small** model so it runs in class. We log common metrics.

```{code-cell} ipython3
# A short run. Increase epochs later if you have time/GPU.
!chemprop train \
  --data-path mp_data.csv \
  -t regression \
  -s SMILES \
  --target-columns "Melting Point" \
  -o mp_model \
  --num-replicates 1 \
  --epochs 15 \
  --save-smiles-splits \
  --metrics mae rmse r2 \
  --tracking-metric r2
```

Make quick predictions on a few molecules.

```{code-cell} ipython3
smiles_list = [
    "CCO",              # ethanol
    "c1ccccc1",         # benzene
    "CC(=O)O",          # acetic acid
    "CCN(CC)CC"         # triethylamine
]
pd.DataFrame({"SMILES": smiles_list}).to_csv("custom_smiles_reg.csv", index=False)

!chemprop predict \
  --test-path custom_smiles_reg.csv \
  --model-paths mp_model/replicate_0/model_0/best.pt \
  --preds-path mp_preds.csv

pd.read_csv("mp_preds.csv")
```

### 7.3 Reactivity classification (C–H oxidation dataset)

We use the `Reactivity` column and convert it to **binary** 0/1.

```{code-cell} ipython3
df = pd.read_csv(url)
df["Reactivity_bin"] = df["Reactivity"].replace({-1: 0}).astype(int)
df[["SMILES","Reactivity","Reactivity_bin"]].head(3)
```

Write a minimal file.

```{code-cell} ipython3
df[["SMILES", "Reactivity_bin"]].to_csv("reactivity_data_bin.csv", index=False)

# Optional: sanity check the class balance
print(df["Reactivity"].value_counts(dropna=False).to_dict())
print(df["Reactivity_bin"].value_counts(dropna=False).to_dict())
```

Train a short classification model.

```{code-cell} ipython3
!chemprop train \
  --data-path reactivity_data_bin.csv \
  -t classification \
  -s SMILES \
  --target-columns Reactivity_bin \
  -o reactivity_model \
  --num-replicates 1 \
  --epochs 15 \
  --class-balance \
  --metrics roc prc accuracy \
  --tracking-metric roc
```

Predict on new SMILES.

```{code-cell} ipython3
smiles_list = [
    "CCO",
    "c1ccccc1C(F)",
    "C1=C([C@@H]2C[C@H](C1)C2(C)C)",
    "C1=CC=CC=C1C=O",
    "CCN(CC)CC",
    "c1cccc(C=CC)c1"
]
pd.DataFrame({"SMILES": smiles_list}).to_csv("custom_smiles.csv", index=False)

!chemprop predict \
  --test-path custom_smiles.csv \
  --model-paths reactivity_model/replicate_0/model_0/best.pt \
  --preds-path custom_preds.csv

pd.read_csv("custom_preds.csv")
```

```{admonition} Tips
- Increase `--num-replicates` to 3 and `--epochs` to 50-100 for stronger baselines.  
- For class imbalance, keep `--class-balance`.  
- Use `--save-smiles-splits` to capture exact train/val/test molecules for reproducibility.  
```

```{admonition} ⏰ Exercises 7.x
1) Add `--ensemble-size 5` during prediction by passing multiple `--model-paths` if you trained replicates. Compare ROC.  
2) Change tracking metric to `prc` and rerun. Does validation selection change.  
3) For melting point, add `--ffn-hidden-size 800` to increase the head capacity and try 30 epochs.  
```

---

## 8) References

- Gilmer, J. et al., **Neural Message Passing for Quantum Chemistry**, ICML 2017.  
- Yang, K. et al., **Analyzing Learned Molecular Representations for Property Prediction**, J. Chem. Inf. Model. 2019.  
- Chemprop documentation: training flags, metrics, and examples.  
- RDKit: molecule objects, atom and bond APIs.  
- PyTorch: custom `nn.Module`, autograd, and optimizers.

*(If you need links, see your course page where these are listed with URLs.)*

---

## 9) Glossary

```{glossary}
graph neural network
  A neural model that updates node states by aggregating messages from neighbors.

message passing
  The update step where a node combines its own vector with aggregated neighbor vectors.

readout (pooling)
  Operation that compresses node states into a single graph vector, often by sum, mean, or max.

edge index
  A 2×E tensor listing source and destination of each edge.

Chemprop
  A practical library that trains message passing networks directly from SMILES.

replicate
  Independent training run with a different random seed. Often ensembled for stability.

tracking metric
  The metric used to pick the best checkpoint during training.

class balance
  Loss weighting that compensates for skewed class proportions.
```

---

## 10) In‑class activity and solutions

Each question is designed to be solved with what you coded above. Try first, then check the solution blocks.

### Q1. MLP warmup on melting point

- Use features `[MolWt, LogP, TPSA, NumRings]`  
- Standardize, then train the PyTorch MLP `(32,)` for 150 epochs  
- Report `MSE`, `MAE`, `R²` and draw a parity plot

```python
# TO DO: write your MLP training in PyTorch as in Section 2
```

---

### Q2. Build and inspect the toy GNN

- Train `TinyGNN` with pooling `"sum"` on methane and ethane to predict **heavy atom count**  
- Plot the loss  
- Change pooling to `"mean"` and compare loss and predictions

```python
# TO DO: reuse the code in Section 5
```

---

### Q3. RDKit graph mini‑set (optional if RDKit missing)

- Sample 200 SMILES with melting points from the CSV  
- Convert to graphs and train `TinyGNN` with `num_layers=3`, hidden 32  
- Report `R²` on a 80/20 split and draw a parity plot

```python
# TO DO: follow Section 6
```

---

### Q4. Chemprop regression on melting point

- Train Chemprop on `mp_data.csv` for `--epochs 20`  
- Predict on at least 5 new SMILES of your choice and list the predictions

```python
# TO DO: run the chemprop CLI as in Section 7.2
```

---

### Q5. Chemprop classification on toxicity (student challenge)

- Convert `Toxicity` to `1/0` using the mapping `{toxic:1, non_toxic:0}`  
- Save `["SMILES","Toxicity_bin"]` to `tox_data.csv`  
- Train with `--class-balance --epochs 20` and metrics `roc prc accuracy`  
- Predict on a small set of SMILES; show the class probability

```python
# TO DO: similar to Section 7.3, but target is Toxicity_bin
```

---

### Solutions

#### Solution Q1

```{code-cell} ipython3
# Q1 solution
set_seed(0)
df_reg = df_raw[["MolWt", "LogP", "TPSA", "NumRings", "Melting Point"]].dropna()
X = df_reg[["MolWt","LogP","TPSA","NumRings"]].values.astype(np.float32)
y = df_reg["Melting Point"].values.astype(np.float32).reshape(-1,1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr).astype(np.float32)
Xte_s = sc.transform(Xte).astype(np.float32)

mlp = TinyMLP(in_dim=4, hidden=32, out_dim=1)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
train_loader = DataLoader(NumpyDataset(Xtr_s, ytr), batch_size=64, shuffle=True)

losses = []
for epoch in range(150):
    bs = []
    for xb, yb in train_loader:
        pred = mlp(xb); loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        bs.append(loss.item())
    losses.append(np.mean(bs))

mlp.eval()
with torch.no_grad():
    yhat = mlp(torch.from_numpy(Xte_s)).numpy()

print(f"MSE={mean_squared_error(yte, yhat):.2f}  MAE={mean_absolute_error(yte, yhat):.2f}  R2={r2_score(yte, yhat):.3f}")
plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Q1 MLP loss"); plt.grid(alpha=0.3); plt.show()

plt.scatter(yte, yhat, alpha=0.6)
lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
plt.plot(lims, lims, "k--"); plt.xlabel("True MP"); plt.ylabel("Pred MP"); plt.title("Q1 parity")
plt.show()
```

#### Solution Q2

```{code-cell} ipython3
# Q2 solution
train_graphs = [toy_methane(), toy_ethane()]
y_train = torch.tensor([[heavy_atom_count(g)] for g in train_graphs], dtype=torch.float32)

def run(pool):
    gnn = TinyGNN(in_dim=train_graphs[0]["x"].shape[1], hidden=16, out_dim=1, num_layers=2, pool=pool)
    opt = torch.optim.Adam(gnn.parameters(), lr=5e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(200):
        s = 0.0
        for g, ytrue in zip(train_graphs, y_train):
            x = torch.from_numpy(g["x"]); ei = torch.from_numpy(g["edge_index"])
            pred = gnn(x, ei); loss = loss_fn(pred, ytrue.view(1,1))
            opt.zero_grad(); loss.backward(); opt.step(); s += loss.item()
        losses.append(s/len(train_graphs))
    return gnn, losses

g_sum, L_sum = run("sum")
g_mean, L_mean = run("mean")

plt.plot(L_sum, label="sum"); plt.plot(L_mean, label="mean")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("Q2 loss"); plt.legend(); plt.grid(alpha=0.3); plt.show()

for tag, gnn in [("sum", g_sum), ("mean", g_mean)]:
    for g in train_graphs:
        with torch.no_grad():
            p = gnn(torch.from_numpy(g["x"]), torch.from_numpy(g["edge_index"])).item()
        print(tag, "pred", round(p,3), "target", heavy_atom_count(g))
```

#### Solution Q3

```{code-cell} ipython3
# Q3 solution (skip if RDKit missing)
if Chem is not None:
    small = df_raw[["SMILES","Melting Point"]].dropna().sample(200, random_state=0)
    graphs, y = [], []
    for smi, mp in zip(small["SMILES"], small["Melting Point"]):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g); y.append(mp)
    y = torch.tensor(y, dtype=torch.float32).view(-1,1)

    idx = np.arange(len(graphs))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    gnn = TinyGNN(in_dim=graphs[0]["x"].shape[1], hidden=32, out_dim=1, num_layers=3, pool="mean")
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(60):
        batch = np.random.choice(tr, size=min(64, len(tr)), replace=False)
        for i in batch:
            x = torch.from_numpy(graphs[i]["x"])
            ei = torch.from_numpy(graphs[i]["edge_index"])
            pred = gnn(x, ei); loss = loss_fn(pred, y[i:i+1])
            opt.zero_grad(); opt.step()

    yt, yp = [], []
    gnn.eval()
    with torch.no_grad():
        for i in te:
            yp.append(gnn(torch.from_numpy(graphs[i]["x"]),
                          torch.from_numpy(graphs[i]["edge_index"])).item())
            yt.append(float(y[i]))

    from sklearn.metrics import mean_absolute_error, r2_score
    print(f"R2: {r2_score(yt, yp):.3f}  MAE: {mean_absolute_error(yt, yp):.2f}")
else:
    print("RDKit missing; skip Q3.")
```

#### Solution Q4

```{code-cell} ipython3
# Q4 solution
df_reg = df[["SMILES","Melting Point"]].dropna().copy()
df_reg.to_csv("mp_data.csv", index=False)

!chemprop train \
  --data-path mp_data.csv \
  -t regression \
  -s SMILES \
  --target-columns "Melting Point" \
  -o mp_model_q4 \
  --num-replicates 1 \
  --epochs 20 \
  --metrics mae rmse r2 \
  --tracking-metric r2

pd.DataFrame({"SMILES": ["CCO","c1ccccc1","CC(=O)O","CCN(CC)CC","O=C(O)C(O)C"]}).to_csv("q4_smiles.csv", index=False)

!chemprop predict \
  --test-path q4_smiles.csv \
  --model-paths mp_model_q4/replicate_0/model_0/best.pt \
  --preds-path q4_preds.csv

pd.read_csv("q4_preds.csv")
```

#### Solution Q5

```{code-cell} ipython3
# Q5 solution
df = pd.read_csv(url)
df = df[["SMILES","Toxicity"]].dropna().copy()
df["Toxicity_bin"] = df["Toxicity"].str.lower().map({"toxic":1, "non_toxic":0}).astype(int)
df[["SMILES","Toxicity_bin"]].to_csv("tox_data.csv", index=False)

!chemprop train \
  --data-path tox_data.csv \
  -t classification \
  -s SMILES \
  --target-columns Toxicity_bin \
  -o tox_model \
  --num-replicates 1 \
  --epochs 20 \
  --class-balance \
  --metrics roc prc accuracy \
  --tracking-metric roc

pd.DataFrame({"SMILES": ["CCO","c1ccccc1","O=[N+](=O)[O-]","ClCCl","CC(=O)Cl"]}).to_csv("q5_smiles.csv", index=False)

!chemprop predict \
  --test-path q5_smiles.csv \
  --model-paths tox_model/replicate_0/model_0/best.pt \
  --preds-path q5_preds.csv

pd.read_csv("q5_preds.csv")
```

---

### Quick cheat sheet

- Message passing layer shape checks help catch indexing mistakes early.  
- Pooling choice can change gradients a lot on small graphs.  
- Chemprop is fast to try. Start with 15–30 epochs and 1–3 replicates during class. Increase later.  
- For classification targets, always look at `roc` and `prc`, not only accuracy.  

---

```{admonition} What to remember
- An MLP ignores graph structure. A GNN uses edges to mix neighbor information.  
- The core update is neighbor aggregation followed by a learnable transformation.  
- Chemprop encodes many best practices so you can focus on data and targets.  
```

```{admonition} Where this connects
Back to Lecture 8: you built MLPs and read loss curves. The same training loop ideas apply to GNNs. The difference is the **forward** pass uses a graph structure to route information.
```

```{admonition} Try at home
- Compare `sum` vs `mean` pooling on size‑varying molecules.  
- Add bond type one‑hot to the neighbor message with a small MLP inside `SimpleMP`.  
- Use Chemprop for **logS** regression and compare to your MLP from Lecture 8.  
```

