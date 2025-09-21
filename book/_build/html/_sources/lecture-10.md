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






# Lecture 10 - Property & Reaction Prediction

```{contents}
:local:
:depth: 1
```

## Learning goals

- Set up **Chemprop v2** for regression and classification on our C-H oxidation dataset.
- Train four **single task** models for: Solubility, pKa, Melting Point, Toxicity.
- Train a **reactivity** classifier and an **atom-level selectivity** predictor.
- Interpret a trained model with **Shapley values (SHAP)** at the feature and node levels.

Important note: For this lecture 10, it is recommended to run everything in Colab. On this HTML page, some outputs are disabled due to execution limits, so only code and some example output is displayed.
---


```{code-cell} ipython3
:tags: [hide-input]
# 1. Setup

# Hide code cell source
import warnings, os, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# RDKit (must be installed in THIS venv)
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors, AllChem

# Lightning (either import works depending on install)
try:
    from lightning import pytorch as pl
except ImportError:
    import pytorch_lightning as pl


import warnings, math, os, sys, json, time, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path  
import torch



# Sklearn bits for splitting and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)

import logging
logging.getLogger("chemprop.data.splitting").setLevel(logging.ERROR)

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors, AllChem
except Exception:
    try:
        %pip install rdkit
        from rdkit import Chem
        from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors, AllChem
    except Exception as e:
        print("RDKit is not available in this environment. Drawing and descriptors will be skipped.")
        Chem = None

```



## 1. Directed message-passing neural network (D-MPNN)

We will train models for four molecular properties and reaction-related labels using **Chemprop**.

Briefly speaking, Chemprop builds neural models for molecules using a *directed* message passing neural network (D-MPNN).

As you recall from previous lecture, a message passing neural network (MPNN) updates hidden vectors on nodes and edges with local neighbor information, then an aggregation step creates a graph-level vector for prediction.

Chemprop’s directed variant changes the way messages flow: instead of passing information back and forth between atoms, it assigns a hidden state to each directed bond (atom `i` → atom `j`). This prevents immediate backtracking (“tottering”) where messages would simply bounce between two atoms without capturing new context. By using directed bonds, the model distinguishes subtle chemical environments. For example, the information carried from a carbon toward a nitrogen can be different than the reverse direction, which matters for reactivity and selectivity.

As a GNN, Chemprop also featurizes a molecule as a **graph**:

- **Nodes** are atoms with features like atomic number, degree, aromaticity.
- **Edges** are bonds with features like bond order and stereo.

Initial directed bond state $h_{i→j}^{(0)}$ is a learned function of the source atom features and the bond features. For `t = 1..T`, update
$
h_{i \to j}^{(t)} = \sigma \Big( W \cdot \big( h_{i \to j}^{(t-1)} + \sum_{k \in \mathcal{N}(i) \setminus \{j\}} h_{k \to i}^{(t-1)} \big) + b \Big)
$,

where `σ` is an activation such as `ReLU`, `W` is a learned weight, $x_{i→j}$ are featurized inputs, $⊕$ is concatenation, and $N(i)\j$ removes the target atom to avoid immediate backtracking. After T steps, Chemprop aggregates per directed bond states to atom states, then pools to a molecule vector $h_mol$ using sum or mean or attention pooling. $h_mol$ feeds a multitask feedforward head.

We have been working with the following quite many times:
- **Solubility_mol_per_L**: continuous. Regression with loss like MSE or MAE.
- **pKa**: continuous. Regression.
- **Melting Point**: continuous. Regression.
- **Toxicity**: categorical with values like `toxic` or `non_toxic`. Binary classification.

While these two we never try before:
- **Reactivity**: binary label `1` vs `-1`. Binary classification. In our C-H oxidation dataset, this means whether the substrate will undergo oxidation.
- **Site Selectivity**: a set of atom indices. Atom-level classification inside a molecule. In our C-H oxidation dataset, this means which atom(s) are most likely to oxidize under certain electrochemical reaction condition, expressed as atom indices in the SMILES.

As a reminder, below are some reference formulas:

- Regression losses  
  $$
  \text{MSE} = \frac{1}{n}\sum_i (y_i - \hat y_i)^2,\qquad
  \text{MAE} = \frac{1}{n}\sum_i |y_i - \hat y_i|
  $$
- Binary cross entropy  
  $$
  \mathcal{L} = -\frac{1}{n}\sum_i \big(y_i\log \hat p_i + (1-y_i)\log(1-\hat p_i)\big)
  $$




> You saw this idea in earlier lectures. The new part is that Chemprop builds the graph from SMILES and offers modules for molecule, reaction and atom/bond tasks.

---
## 2. Data preparation

We begin with load and inspect the C-H oxidation dataset.

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)
df_raw.head(5)
```

```{code-cell} ipython3
# Basic info
df_raw.shape, df_raw.columns.tolist()
```

```{code-cell} ipython3
# Clean a copy and normalize a few columns
df = df_raw.copy()

# Toxicity -> binary string 'toxic'/'non_toxic' to 1/0 if present
tox_map = {"toxic": 1, "non_toxic": 0}
if "Toxicity" in df:
    df["tox_bin"] = df["Toxicity"].str.lower().map(tox_map)

# Reactivity -> 1/-1 to 1/0
if "Reactivity" in df:
    df["react_bin"] = df["Reactivity"].map(lambda x: 1 if x==1 else 0)

# Oxidation Site -> list of ints
def parse_sites(x):
    if isinstance(x, str) and len(x.strip())>0:
        return [int(v) for v in x.split(",")]
    return []
df["site_list"] = df["Oxidation Site"].apply(parse_sites)

# Take log of solubility (keep same column name)
if "Solubility_mol_per_L" in df:
    df["logS"] = np.log10(df["Solubility_mol_per_L"] + 1e-6)

df[["SMILES","logS","pKa","Toxicity","Melting Point","react_bin","site_list"]].head(8)
```



```{admonition} ⏰ **Exercise 1**
Count number of postive and negative reaction outcomes in `react_bin`.. 
```


We will create **MoleculeDatapoint** objects from SMILES and targets, split the data, and build loaders.
Our first target will be `solubility`.

> Step 1. Build datapoints.

 Each row of the dataframe is now represented as a `MoleculeDatapoint`. It stores the SMILES, the numeric target (solubility here), plus metadata like optional weights.  
 This is the *atomic unit* Chemprop will pass to the featurizer.

```python
# Keep rows that have both SMILES and solubility
df_sol = df[["SMILES","logS"]].dropna()
smis = df_sol["SMILES"].tolist()
ys   = df_sol["logS"].to_numpy().reshape(-1,1)

sol_datapoints = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]
len(sol_datapoints), sol_datapoints[0].y.shape
```


Step 2. Split to train, val, test.

  We divided the list of datapoints into three folds.  
   - Training: used to fit model weights.  
   - Validation: used to monitor progress and stop early.  
   - Test: kept blind until the end.  
   Even though we used a random split here, Chemprop also supports scaffold-based splits which are often better for chemistry.


```python
mols = [dp.mol for dp in sol_datapoints]

train_lists, val_lists, test_lists = make_split_indices(
    mols=mols,
    split="random",  # or "scaffold", "stratified", etc.
    sizes=(0.8, 0.1, 0.1),
    seed=0,
    num_replicates=1
)

train_dpss, val_dpss, test_dpss = split_data_by_indices(
    sol_datapoints, train_lists, val_lists, test_lists
)

print(len(train_dpss[0]), len(val_dpss[0]), len(test_dpss[0]))
```

> Step 3. Build dataset objects and scale targets.
A `MoleculeDataset` wraps the datapoints and applies the chosen featurizer.  
   - Here we used `SimpleMoleculeMolGraphFeaturizer`, which turns atoms and bonds into numeric arrays.  
   - We also normalized the target values (subtract mean, divide by std) so the model trains smoothly. The stored `scaler` allows us to unscale predictions back.

```python
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
train_set = data.MoleculeDataset(train_dpss[0], featurizer=feat)
scaler = train_set.normalize_targets()  # store mean/var

val_set = data.MoleculeDataset(val_dpss[0], featurizer=feat)
val_set.normalize_targets(scaler)

test_set = data.MoleculeDataset(test_dpss[0], featurizer=feat)

# Peek at one item structure
item0 = train_set[0]
type(item0).__name__, item0.y, item0.mg.V.shape, item0.mg.E.shape
```

> Step 4. Dataloaders.
 Finally, we wrapped datasets in PyTorch-style `DataLoader`s.  
   - Training loader will shuffle each epoch.  
   - Validation and test loaders do not shuffle, to keep evaluation consistent.  
   Batching is automatic: molecules of different sizes are packed together and masks are used internally.
```python
train_loader = data.build_dataloader(train_set, num_workers=0)
val_loader   = data.build_dataloader(val_set, num_workers=0, shuffle=False)
test_loader  = data.build_dataloader(test_set, num_workers=0, shuffle=False)
```

```{admonition} Tip
Scaling targets helps training for many regressors. We keep the unscale transform for outputs inside the model so that you get predictions in the original units.
```
---

## 3. Property prediction (regression)

We will configure a small MPNN for regression.  

In particular, we will:  
1. Choose the neural blocks that define how messages are passed, pooled, and transformed into outputs.  
2. Assemble them into a complete model object.  
3. Set up a training loop with early stopping and checkpoints.  
4. Evaluate predictions on a held-out test set and visualize the quality using a parity plot.  


### 3.1 Pick blocks

```{code-cell} ipython3
mp  = nn.BondMessagePassing()        # node/edge update
agg = nn.MeanAggregation()           # pool node vectors
out = nn.RegressionFFN(              # simple FFN head
    output_transform=nn.UnscaleTransform.from_standard_scaler(scaler)
)
batch_norm = True
metrics = [nn.metrics.RMSE(), nn.metrics.MAE()]  # first metric used for early stopping
```

- **BondMessagePassing()** updates hidden states on each directed bond by passing information across neighbors.  
- **MeanAggregation()** pools hidden vectors to form atom or molecule-level representations. Other options like sum or attention pooling are possible.  
- **RegressionFFN()** is a feed-forward head. Here we attach an `UnscaleTransform` so predictions can be mapped back to the original solubility scale.  
- **Batch normalization** improves stability by normalizing hidden states during training.  
- **Metrics** let us monitor training. RMSE (root mean squared error) and MAE (mean absolute error) are both useful, but RMSE is often more sensitive to large errors and is used for early stopping.


### 3.2 Build model and trainer

Once the blocks are chosen, we wrap them into a full MPNN model. Chemprop uses PyTorch Lightning under the hood, so we also set up a Trainer:

- The ModelCheckpoint callback saves the best version of the model during training, based on validation loss.

- The trainer can run on CPU or GPU (`accelerator="auto"`).

We limit to 15 epochs here for demonstration, but in practice you might extend this depending on dataset size and convergence.

```{code-cell} ipython3
mpnn_sol = models.MPNN(mp, agg, out, batch_norm, metrics)

checkpoint_dir = Path("checkpoints_sol")
checkpoint_dir.mkdir(exist_ok=True)
ckpt = pl.callbacks.ModelCheckpoint(
    dirpath=str(checkpoint_dir), filename="best-{epoch}-{val_loss:.3f}",
    monitor="val_loss", mode="min", save_last=True
)

trainer = pl.Trainer(
    logger=False, enable_checkpointing=True, accelerator="auto",
    devices=1, max_epochs=15, callbacks=[ckpt]
)
mpnn_sol
```
At this stage, we have a complete pipeline: dataset loaders, model blocks, and a trainer that knows when to save progress.
### 3.3 Train

Since we implement everything earlier, now training is as simple as calling `fit()`. The trainer will:

1. Iterate over the training loader each epoch.

2. Evaluate on the validation loader.

3. Save checkpoints when the validation RMSE improves.

During training, you can monitor validation loss to see whether the model is underfitting, overfitting, or converging as expected.

```{code-cell} ipython3
trainer.fit(mpnn_sol, train_loader, val_loader)
```

### 3.4 Test and parity plot

After training, we hold back the test set for final evaluation. We then visualize predicted vs. true values with a parity plot:



```{code-cell} ipython3
results = trainer.test(mpnn_sol, test_loader)

# Gather predictions for parity
import torch
with torch.inference_mode():
    preds = trainer.predict(mpnn_sol, test_loader)
preds = np.concatenate(preds, axis=0).ravel()

y_true = test_set.Y.ravel()
print("Test size:", len(y_true))

plt.scatter(y_true, preds, alpha=0.6)

# Set both axes to the same range
lims = [-3, 1]
plt.plot(lims, lims, "k--")
plt.xlim(lims)
plt.ylim(lims)

plt.xlabel("True solubility (mol/L)")
plt.ylabel("Predicted")
plt.title("Parity plot: Solubility")
plt.grid(True)
plt.show()
```

```{admonition} ⏰ Exercise
Change the aggregation from `MeanAggregation()` to `SumAggregation()` and retrain for 10 epochs. Compare RMSE and the parity plot. What changed?
```

---

## 4. Property prediction (classification)


Here we predict `toxic` vs `non_toxic`.

### 4.1 Build classification dataset

```{code-cell} ipython3
df_tox = df[["SMILES","tox_bin"]].dropna()
smis = df_tox["SMILES"].tolist()
ys   = df_tox["tox_bin"].astype(int).to_numpy().reshape(-1,1)

tox_dps = [data.MoleculeDatapoint.from_smi(s,y) for s,y in zip(smis,ys)]
mols = [dp.mol for dp in tox_dps]
tr_idx, va_idx, te_idx = data.make_split_indices(mols, "random", (0.8,0.1,0.1))
tr, va, te = data.split_data_by_indices(tox_dps, tr_idx, va_idx, te_idx)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
tox_tr = data.MoleculeDataset(tr[0], featurizer=feat)
tox_va = data.MoleculeDataset(va[0], featurizer=feat)
tox_te = data.MoleculeDataset(te[0], featurizer=feat)
```

### 4.2 Model and training

```{code-cell} ipython3
mp  = nn.BondMessagePassing()
agg = nn.MeanAggregation()
ffn = nn.BinaryClassificationFFN(n_tasks=1)
mpnn_tox = models.MPNN(mp, agg, ffn, batch_norm=False)  

tr_loader = data.build_dataloader(tox_tr, num_workers=0)
va_loader = data.build_dataloader(tox_va, num_workers=0, shuffle=False)
te_loader = data.build_dataloader(tox_te, num_workers=0, shuffle=False)

trainer_tox = pl.Trainer(logger=False, enable_checkpointing=True, accelerator="auto",
                         devices=1, max_epochs=15)
trainer_tox.fit(mpnn_tox, tr_loader, va_loader)
trainer_tox.test(mpnn_tox, te_loader)
```

### 4.3 ROC curve

```{code-cell} ipython3

# Gather probabilities
with torch.inference_mode():
    pred_chunks = trainer_tox.predict(mpnn_tox, te_loader)
proba = np.concatenate(pred_chunks, axis=0).ravel()
y_true = tox_te.Y.ravel().astype(int)

auc = roc_auc_score(y_true, proba)
acc = accuracy_score(y_true, (proba>=0.5).astype(int))
print(f"Test AUC: {auc:.3f}  Accuracy: {acc:.3f}")

fpr, tpr, thr = roc_curve(y_true, proba)
plt.plot(fpr, tpr, lw=2)
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC: Toxicity")
plt.grid(True)
plt.show()
```

---

## 5. Reactivity and selectivity

Two targets in this dataset relate to reactions.

- **Reactivity**: binary at the molecule level.
- **Selectivity**: oxidation site indices at the atom level.


Before we train models, let’s **look at the labels**. Let's pick three representative molecules from the C-H oxidation dataset and see their reactivity and selectivity label.

```{code-cell} ipython3
:tags: [hide-input]
# Helper selection: one negative (no reaction), one positive with a single site, one positive with multiple sites
def pick_representatives(df):
    df_pos = df[(df["react_bin"] == 1) & (df["site_list"].map(len) >= 1)].copy()
    df_pos_multi = df_pos[df_pos["site_list"].map(len) >= 2].copy()
    df_neg = df[(df["react_bin"] == 0)].copy()

    reps = []
    if not df_neg.empty:
        reps.append(("Negative (react_bin=0)", df_neg.iloc[0]))
    if not df_pos.empty:
        reps.append(("Positive (react_bin=1; 1 site)", df_pos[df_pos["site_list"].map(len) == 1].iloc[0]
                     if (df_pos["site_list"].map(len) == 1).any() else df_pos.iloc[0]))
    if not df_pos_multi.empty:
        reps.append(("Positive (react_bin=1; multi-site)", df_pos_multi.iloc[0]))

    # If fewer than 3 examples exist, just return what we have
    return reps

reps = pick_representatives(df)
len(reps), [t for t,_ in reps]
# Show the chosen rows so readers see SMILES and labels
import pandas as pd

def row_view(r):
    return {
        "SMILES": r["SMILES"],
        "react_bin": r["react_bin"],
        "site_list (1-based)": r["site_list"]
    }

rep_table = pd.DataFrame([row_view(row) for _, row in reps])
rep_table
```

Let's draw them:

```{code-cell} ipython3
:tags: [hide-input]
from rdkit.Chem.Draw import rdMolDraw2D

def make_annotated_copy(mol, site_list_1based=None, tag_c123=True):
    m = Chem.Mol(mol)  # copy
    Chem.AssignAtomChiralTagsFromStructure(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    n = m.GetNumAtoms()
    # Highlight oxidation sites (convert to 0-based safely)
    hi_atoms = []
    if site_list_1based:
        for idx1 in site_list_1based:
            j = idx1 - 1
            if 0 <= j < n:
                hi_atoms.append(j)

    # Always annotate the atom index so readers see 1-based indexing used in labels
    for j in range(n):
        a = m.GetAtomWithIdx(j)
        idx1 = j + 1
        old = a.GetProp("atomNote") if a.HasProp("atomNote") else ""
        # If this atom is an oxidation site, add a star
        star = "*" if (j in hi_atoms) else ""
        a.SetProp("atomNote", f"{old} {idx1}{star}".strip())

    return m, hi_atoms

def draw_examples(reps, mol_size=(320, 280)):
    ms = []
    legends = []
    highlights = []

    for title, row in reps:
        smi = row["SMILES"]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        m_annot, hi = make_annotated_copy(m, site_list_1based=row["site_list"], tag_c123=True)
        ms.append(m_annot)
        lbl = f"Reactivity={row['react_bin']}, Sites={row['site_list']}"
        legends.append(lbl)
        highlights.append(hi)

    imgs = []
    for m, hi, lg in zip(ms, highlights, legends):
        img = Draw.MolToImage(m, size=mol_size, highlightAtoms=hi)
        img.info["legend"] = lg
        imgs.append(img)

    # Create a grid image manually by re-drawing with legends
    return Draw.MolsToGridImage(ms, molsPerRow=len(ms), subImgSize=mol_size,
                                legends=legends,
                                highlightAtomLists=highlights)

grid_img = draw_examples(reps)
grid_img

```


### 5.1 Reactivity classifier

This mirrors the toxicity workflow.

```{code-cell} ipython3
df_rxn = df[["SMILES","react_bin"]].dropna()
smis = df_rxn["SMILES"].tolist()
ys   = df_rxn["react_bin"].astype(int).to_numpy().reshape(-1,1)

rxn_dps = [data.MoleculeDatapoint.from_smi(s,y) for s,y in zip(smis, ys)]
mols = [dp.mol for dp in rxn_dps]
tr_idx, va_idx, te_idx = data.make_split_indices(mols, "random", (0.8,0.1,0.1))
tr, va, te = data.split_data_by_indices(rxn_dps, tr_idx, va_idx, te_idx)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
rxn_tr = data.MoleculeDataset(tr[0], featurizer=feat)
rxn_va = data.MoleculeDataset(va[0], featurizer=feat)
rxn_te = data.MoleculeDataset(te[0], featurizer=feat)

mp  = nn.BondMessagePassing()
agg = nn.MeanAggregation()
ffn = nn.BinaryClassificationFFN(n_tasks=1)
mpnn_rxn = models.MPNN(mp, agg, ffn, batch_norm=False)

tr_loader = data.build_dataloader(rxn_tr, num_workers=0)
va_loader = data.build_dataloader(rxn_va, num_workers=0, shuffle=False)
te_loader = data.build_dataloader(rxn_te, num_workers=0, shuffle=False)

trainer_rxn = pl.Trainer(logger=False, enable_checkpointing=True, accelerator="auto",
                         devices=1, max_epochs=15)
trainer_rxn.fit(mpnn_rxn, tr_loader, va_loader)
trainer_rxn.test(mpnn_rxn, te_loader)
```

### 5.2 Atom-level selectivity: build labels per atom

We want atom targets `1` for indices that appear in `site_list`, and `0` otherwise.

We will create **MolAtomBondDatapoint** objects. For a gentle first pass, we only supply `atom_y` and leave other advanced inputs out.

```{code-cell} ipython3
import ast

def atoms_labels_from_smiles(smi, positive_idxs):
    if Chem is None:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    y = np.zeros((n,1), dtype=float)
    for idx in positive_idxs:
        # dataset uses 1-based indexing in the text, RDKit uses 0-based
        j = idx-1
        if 0 <= j < n:
            y[j,0] = 1.0
    return y

# Build list of MolAtomBondDatapoint for selectivity
sel_rows = df[["SMILES","site_list"]].dropna()
sel_dps = []
for smi, sites in sel_rows.itertuples(index=False):
    atom_y = atoms_labels_from_smiles(smi, sites)
    if atom_y is None:
        continue
    # We provide atom_y, molecule-level y is optional here
    dp = data.MolAtomBondDatapoint.from_smi(
        smi, atom_y=atom_y, reorder_atoms=False
    )
    sel_dps.append(dp)

len(sel_dps), type(sel_dps[0]).__name__
```

Split and dataset construction.

```{code-cell} ipython3
mols = [Chem.MolFromSmiles(dp.name) if hasattr(dp, "name") else None for dp in sel_dps]
# For structure-based split we need RDKit Mol. Build directly from SMILES fallback:
mols = [Chem.MolFromSmiles(df.loc[df["SMILES"]==dp.name, "SMILES"].iloc[0]) if Chem else None for dp in sel_dps]

tr_idx, va_idx, te_idx = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
tr, va, te = data.split_data_by_indices(sel_dps, tr_idx, va_idx, te_idx)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
tr_set = data.MolAtomBondDataset(tr[0], featurizer=feat)
va_set = data.MolAtomBondDataset(va[0], featurizer=feat)
te_set = data.MolAtomBondDataset(te[0], featurizer=feat)

tr_loader = data.build_dataloader(tr_set, shuffle=True, batch_size=8)
va_loader = data.build_dataloader(va_set, shuffle=False, batch_size=8)
te_loader = data.build_dataloader(te_set, shuffle=False, batch_size=8)
```

Model for molecule and atom predictions. Here we focus on atom prediction.

```{code-cell} ipython3
mp = nn.MABBondMessagePassing(
    d_v=feat.atom_fdim, d_e=feat.bond_fdim, d_h=300, depth=3, dropout=0.1
)
agg = nn.MeanAggregation()

atom_predictor = nn.BinaryClassificationFFN(n_tasks=1)  # atom-level 0/1

model_sel = models.MolAtomBondMPNN(
    message_passing=mp,
    agg=agg,
    mol_predictor=None,
    atom_predictor=atom_predictor,
    bond_predictor=None,
    batch_norm=True,
    metrics=[nn.metrics.BinaryAUROC()],
)

trainer_sel = pl.Trainer(logger=False, enable_checkpointing=True, accelerator="auto",
                         devices=1, max_epochs=8)
trainer_sel.fit(model_sel, tr_loader, va_loader)
trainer_sel.test(model_sel, te_loader)
```

Inspect a molecule with predicted atom probabilities by Visualizing probabilities as atom annotations for the first molecule in the batch.

```{code-cell} ipython3
:tags: [hide-input]
# 1) Pick which test item to visualize
idx = 11  # change as you like

# 2) Grab the original datapoint (not the already-featurized datum)
dp = te_set.data[idx]   # IMPORTANT: .data gives you the raw MolAtomBondDatapoint

# 3) Make a one-item MolAtomBondDataset with the SAME featurizer you used before
single_ds = data.MolAtomBondDataset([dp], featurizer=feat)

# 4) Build a loader and get its batch
single_loader = data.build_dataloader(single_ds, shuffle=False, batch_size=1)
batch = next(iter(single_loader))  # this now matches collate expectations

with torch.inference_mode():
    out = model_sel(batch[0], None)   # batch[0] is the MolGraphBatch

# 6) Unpack outputs and get per-atom probabilities
_, atom_logits, _ = out
atom_probs = torch.sigmoid(atom_logits).cpu().numpy().ravel()

print("Atom count:", len(atom_probs))
print("First 10 probabilities:", atom_probs[:10])

# 7) Draw the SAME molecule with aligned probabilities
smi = dp.name  # MolAtomBondDatapoint stores SMILES in .name
m = Chem.MolFromSmiles(smi)
m2 = Chem.Mol(m)
for i, a in enumerate(m2.GetAtoms()):
    a.SetProp("atomNote", f"{atom_probs[i]:.2f}")
img = Draw.MolToImage(m2, size=(400, 400))
display(img)

```

```{admonition} ⏰ Exercise 7.1
Switch the atom predictor to a small regression head and train with labels 0.0 or 1.0. Then threshold the outputs at 0.5 to recover classes. Compare AUROC.
```



---

## 6. Interpretation with SHAP

We show a compact variant of SHAP for a trained **molecule-level** model. This helps answer: which features or nodes influenced a prediction.


### 6.1 Install and prepare

```{code-cell} ipython3
try:
    import shap
except Exception:
    # !pip install shap 
    import shap
```

### 6.2 Wrap the model for SHAP

We illustrate feature-ablation style with the default featurizers.

```{code-cell} ipython3
from copy import deepcopy

# Use the solubility model as example
test_smi = df_sol["SMILES"].iloc[0]

atom_featurizer = featurizers.atom.MultiHotAtomFeaturizer.v2()
bond_featurizer = featurizers.bond.MultiHotBondFeaturizer()

def predict_with_keep_masks(keep_atom_feats, keep_bond_feats, smi):
    # This uses masks of booleans for atom and bond feature groups
    fz = featurizers.molgraph.molecule.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer
    )
    dp = data.MoleculeDatapoint.from_smi(smi)
    ds = data.MoleculeDataset([dp], featurizer=fz)
    dl = data.build_dataloader(ds, shuffle=False, batch_size=1)

    with torch.inference_mode():
        pred = trainer.predict(mpnn_sol, dl)[0][0]
    return float(pred)
```

Define a SHAP-compatible callable and a simple masker.

```{code-cell} ipython3
# Example wrapper that expects a 12-dim mask [8 atom feat groups + 4 bond feat groups]
def model_wrapper(X):
    preds = []
    for row in X:
        a_mask = [bool(v) for v in row[:8]]
        b_mask = [bool(v) for v in row[8:12]]
        preds.append([predict_with_keep_masks(a_mask, b_mask, test_smi)])
    return np.array(preds)

def binary_masker(binary_mask, x):
    x2 = deepcopy(x)
    x2[binary_mask == 0] = 0
    return np.array([x2])
```

Run SHAP on a single mask example.

```{code-cell} ipython3
explainer = shap.PermutationExplainer(model_wrapper, masker=binary_masker)
X0 = np.ones((1,12))
explanation = explainer(X0, max_evals=100)

explanation.values, explanation.base_values
```

Bar plot of contributions.

```{code-cell} ipython3
shap.plots.bar(explanation[0])
```

```{admonition} Notes
- Values indicate how keeping a feature group pushes the predicted solubility up or down relative to a base.
- Use care when interpreting these numbers. For rigorous analysis, use trained models with stable validation performance and repeat SHAP sampling.
```

```{admonition} ⏰ Exercise 8.1
Change `test_smi` to another molecule from the dataset and rerun the SHAP explainer. Which feature group appears most helpful for that molecule?
```

---

## 9. Save and run inference

We collect model checkpoints and write a helper that predicts all four properties plus reactivity from a list of SMILES.

```{code-cell} ipython3
# Keep references to trained models. If you restarted the kernel, reload from checkpoints.
sol_model  = mpnn_sol
pka_model  = mpnn_pka
mp_model   = mpnn_mp
tox_model  = mpnn_tox
rxn_model  = mpnn_rxn

def predict_all(smis):
    outs = []
    # Build datasets once per model to reuse featurizers
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dps = [data.MoleculeDatapoint.from_smi(s) for s in smis]
    ds = data.MoleculeDataset(test_dps, featurizer=feat)
    dl = data.build_dataloader(ds, shuffle=False)

    with torch.inference_mode():
        sol = np.concatenate(trainer.predict(sol_model, dl), axis=0).ravel()
        pka = np.concatenate(trainer_pka.predict(pka_model, dl), axis=0).ravel()
        mpv = np.concatenate(trainer_mp.predict(mp_model, dl), axis=0).ravel()
        tox = np.concatenate(trainer_tox.predict(tox_model, dl), axis=0).ravel()
        rxn = np.concatenate(trainer_rxn.predict(rxn_model, dl), axis=0).ravel()

    return pd.DataFrame({
        "SMILES": smis,
        "Solubility_pred": sol,
        "pKa_pred": pka,
        "MeltingPoint_pred": mpv,
        "Toxicity_prob": tox,
        "Reactivity_prob": rxn
    })

demo = predict_all(df["SMILES"].head(8).tolist())
demo
```

```{admonition} ⏰ Exercise 9.1
Create a small list of three custom SMILES and call `predict_all`. Inspect which of the four properties differs the most across your three examples.
```

---

## 10. Glossary • References • In-class activity with solutions

### 10.1 Glossary

- **MPNN**: Message Passing Neural Network. Learns node and edge embeddings by exchanging information along bonds.
- **Aggregation**: Pools node embeddings to a molecule vector. Common choices are mean and sum.
- **FFN**: Feed-forward network head used for regression or classification.
- **AUROC**: Area under ROC. Measures ranking quality of a binary classifier.
- **Selectivity (atom-level)**: Probability that a given atom is the reaction site under a given transformation.
- **CGR**: Condensed Graph of Reaction. A featurization for reaction tasks.
- **SHAP**: Shapley values for local explanation of a prediction.

### 10.2 References

- Li, S.-C., Wu, H., Menon, A., Spiekermann, K. A., Li, Y.-P., Green, W. H. When Do Quantum Mechanical Descriptors Help Graph Neural Networks to Predict Chemical Properties? **JACS** 2024, 146, 23103–23120. https://doi.org/10.1021/jacs.4c04670  
  SHAP implementation for Chemprop v2 is based on the approach described by the authors.

- Chemprop v2 documentation and examples: training, classification, multicomponent, reaction and atom/bond prediction.

### 10.3 In-class activity

Answer all five questions. Most are small edits of code we wrote today.

**Q1. Solubility log-transform**  
Create `y_log = log10(Solubility_mol_per_L + 1e-6)` and train a regression MPNN. Compare RMSE to the non-log model.

**Q2. pKa vs Melting multi-compare**  
Train two separate models as done above and report test MAE for each. Which one is easier to predict? Add a parity plot for both.

**Q3. Toxicity thresholding**  
Change the classification threshold from 0.5 to the value that maximizes F1 on the validation set. Report the new test accuracy and AUROC.

**Q4. Reactivity calibration**  
For the reactivity model, plot a reliability curve (predicted prob vs empirical rate binned into deciles). Is the model over or under confident?

**Q5. Atom-level selectivity**  
Take five molecules from the test set with nonempty `site_list`. For each, show the top 2 atoms by predicted probability and compare with ground truth indices.

---

### 10.4 Solutions

```{admonition} Q1 solution
:class: dropdown
```

```{code-cell} ipython3
# Log-solubility
df_log = df[["SMILES","Solubility_mol_per_L"]].dropna().copy()
df_log["y_log"] = np.log10(df_log["Solubility_mol_per_L"] + 1e-6)

smis = df_log["SMILES"].tolist()
ys   = df_log["y_log"].to_numpy().reshape(-1,1)
dps = [data.MoleculeDatapoint.from_smi(s,y) for s,y in zip(smis, ys)]
mols = [dp.mol for dp in dps]
tr_idx, va_idx, te_idx = data.make_split_indices(mols, "random", (0.8,0.1,0.1))
tr, va, te = data.split_data_by_indices(dps, tr_idx, va_idx, te_idx)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
tr_set = data.MoleculeDataset(tr[0], featurizer=feat)
sc = tr_set.normalize_targets()
va_set = data.MoleculeDataset(va[0], featurizer=feat); va_set.normalize_targets(sc)
te_set = data.MoleculeDataset(te[0], featurizer=feat)

mp  = nn.BondMessagePassing()
agg = nn.MeanAggregation()
out = nn.RegressionFFN(output_transform=nn.UnscaleTransform.from_standard_scaler(sc))
model = models.MPNN(mp, agg, out, batch_norm=True, metrics=[nn.metrics.RMSE(), nn.metrics.MAE()])

tr_loader = data.build_dataloader(tr_set)
va_loader = data.build_dataloader(va_set, shuffle=False)
te_loader = data.build_dataloader(te_set, shuffle=False)

trainer_log = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="auto",
                         devices=1, max_epochs=12)
trainer_log.fit(model, tr_loader, va_loader)
trainer_log.test(model, te_loader)
```

```{admonition} Q2 solution
:class: dropdown
```

```{code-cell} ipython3
# pKa
pka_tr, pka_va, pka_te, pka_un = build_regression_loaders(df, "pKa")
mpnn_pka2, trainer_pka2, pka_te_loader2, pka_te_set2 = train_regression_mpn(
    pka_tr, pka_va, pka_te, pka_un, epochs=12, tag="pka_q2"
)

# Melting
mp_tr, mp_va, mp_te, mp_un = build_regression_loaders(df, "Melting Point")
mpnn_mp2, trainer_mp2, mp_te_loader2, mp_te_set2 = train_regression_mpn(
    mp_tr, mp_va, mp_te, mp_un, epochs=12, tag="mp_q2"
)
```

```{admonition} Q3 solution
:class: dropdown
```

```{code-cell} ipython3
from sklearn.metrics import f1_score

# Get validation probabilities and find best threshold
va_loader = data.build_dataloader(tox_va, shuffle=False)
with torch.inference_mode():
    va_probs = np.concatenate(trainer_tox.predict(mpnn_tox, va_loader), axis=0).ravel()
y_val = tox_va.Y.ravel().astype(int)

ths = np.linspace(0.1, 0.9, 41)
f1s = [f1_score(y_val, (va_probs>=t).astype(int)) for t in ths]
best_t = ths[int(np.argmax(f1s))]
print("Best threshold by F1:", best_t)

# Evaluate on test
te_loader = data.build_dataloader(tox_te, shuffle=False)
with torch.inference_mode():
    te_probs = np.concatenate(trainer_tox.predict(mpnn_tox, te_loader), axis=0).ravel()
y_test = tox_te.Y.ravel().astype(int)

acc = accuracy_score(y_test, (te_probs>=best_t).astype(int))
auc = roc_auc_score(y_test, te_probs)
print(f"New test Accuracy: {acc:.3f}  AUC: {auc:.3f}")
```

```{admonition} Q4 solution
:class: dropdown
```

```{code-cell} ipython3
# Reliability curve for reactivity
from sklearn.calibration import calibration_curve

te_loader = data.build_dataloader(rxn_te, shuffle=False)
with torch.inference_mode():
    re_probs = np.concatenate(trainer_rxn.predict(mpnn_rxn, te_loader), axis=0).ravel()
y = rxn_te.Y.ravel().astype(int)

frac_pos, mean_pred = calibration_curve(y, re_probs, n_bins=10, strategy="quantile")
plt.plot(mean_pred, frac_pos, "o-")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction positive")
plt.title("Reliability curve: Reactivity")
plt.grid(True)
plt.show()
```

```{admonition} Q5 solution
:class: dropdown
```

```{code-cell} ipython3
# Pick five with nonempty site_list
cand = df[df["site_list"].map(len)>0].head(5)

results = []
for smi, sites in cand[["SMILES","site_list"]].itertuples(index=False):
    dp = data.MolAtomBondDatapoint.from_smi(smi, reorder_atoms=False)
    ds = data.MolAtomBondDataset([dp], featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer())
    dl = data.build_dataloader(ds, shuffle=False, batch_size=1)
    with torch.inference_mode():
        batch = next(iter(dl))
        out = model_sel(*batch[:1]) # mg only
        probs = torch.sigmoid(out["atom"]).cpu().numpy().ravel()
    top2 = np.argsort(probs)[-2:] + 1  # convert to 1-based for comparison
    results.append({"SMILES": smi, "true_sites": sites, "top2_pred": top2.tolist()})

pd.DataFrame(results)
```
















---

## Chemprop v2: practical graph models for chemistry

Chemprop implements a directed message passing neural network with strong defaults. We will:

1) **Install Chemprop v2**  
2) Run a **melting point** regression  
3) Run a **reactivity** classification and predict on new SMILES

###  Install Chemprop

```{code-cell} ipython3
# You may need a restart after install in some environments
%pip -q install chemprop
```

### Melting point regression

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

### Reactivity classification (C–H oxidation dataset)

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
