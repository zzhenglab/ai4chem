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





## Learning goals

- Set up **Chemprop 1.x** for regression and classification on our C–H oxidation dataset.
- Train four **single task** models for: Solubility, pKa, Melting Point, Toxicity.
- Train a **reactivity** classifier.
- Build a simple **atom-level selectivity** baseline using atom-centered descriptors plus a small neural net.
- Interpret a trained model with **SHAP** using descriptor features as a proxy.

---

```{code-cell} ipython3
:tags: [hide-input]
import warnings, os, sys, json, time, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
except Exception:
    %pip -q install rdkit-pypi
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem

# Ensure chemprop 1.x
try:
    import chemprop, importlib
    from packaging import version
    if version.parse(chemprop.__version__).major >= 2:
        raise RuntimeError(f"Chemprop {chemprop.__version__} detected. Installing 1.x for this lecture.")
except Exception:
    pass

try:
    import chemprop
except Exception:
    %pip -q install "chemprop<2"
    import chemprop

import chemprop
from packaging import version
print("Chemprop version:", chemprop.__version__)
assert version.parse(chemprop.__version__).major == 1, "This notebook expects Chemprop 1.x"
```

---

## 1. Directed message passing neural network

Chemprop 1.x uses a directed message passing neural network over bonds. Each bond i→j maintains a hidden state, which helps reduce tottering. Atom and bond features are built from SMILES, message passing updates edge states for T steps, then the model pools to a molecule vector and applies a feedforward head. We will use the standard Chemprop 1.x training entry points, either via CLI or via `chemprop.args` and `chemprop.train`.

Targets in this dataset:

- **Solubility_mol_per_L**: regression. We will predict `logS = log10(Solubility + 1e-6)`.
- **pKa**: regression.
- **Melting Point**: regression.
- **Toxicity**: binary classification.
- **Reactivity**: binary classification for whether oxidation occurs under the given conditions.
- **Site Selectivity**: atom indices for likely oxidation sites. Chemprop 1.x does not provide an atom-level head, so we will build a descriptor-based baseline for per-atom labels.

---

## 2. Data preparation

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)
df_raw.head(3)
```

```{code-cell} ipython3
# Clean a copy and engineer columns used below
df = df_raw.copy()

# Toxicity
tox_map = {"toxic": 1, "non_toxic": 0}
if "Toxicity" in df:
    df["tox_bin"] = df["Toxicity"].str.lower().map(tox_map)

# Reactivity -> 1/-1 to 1/0
if "Reactivity" in df:
    df["react_bin"] = df["Reactivity"].map(lambda x: 1 if x==1 else 0)

# Site list
def parse_sites(x):
    if isinstance(x, str) and len(x.strip())>0:
        return [int(v) for v in x.split(",")]
    return []
df["site_list"] = df["Oxidation Site"].apply(parse_sites)

# Log solubility
if "Solubility_mol_per_L" in df:
    df["logS"] = np.log10(df["Solubility_mol_per_L"] + 1e-6)

df[["SMILES","logS","pKa","Melting Point","tox_bin","react_bin","site_list"]].head(8)
```

```{admonition} ⏰ Exercise 1
Count positives and negatives for `react_bin`.
```

Chemprop expects CSVs with a **smiles** column plus one column per task.

```{code-cell} ipython3
data_dir = Path("chemprop_data"); data_dir.mkdir(exist_ok=True)

# Write per task CSVs
df_sol = df[["SMILES","logS"]].dropna().rename(columns={"SMILES":"smiles"})
df_sol.to_csv(data_dir/"solubility.csv", index=False)

df_pka = df[["SMILES","pKa"]].dropna().rename(columns={"SMILES":"smiles"})
df_pka.to_csv(data_dir/"pka.csv", index=False)

df_mp  = df[["SMILES","Melting Point"]].dropna().rename(columns={"SMILES":"smiles"})
df_mp.to_csv(data_dir/"melting.csv", index=False)

df_tox = df[["SMILES","tox_bin"]].dropna().rename(columns={"SMILES":"smiles"})
df_tox.to_csv(data_dir/"toxicity.csv", index=False)

df_rxn = df[["SMILES","react_bin"]].dropna().rename(columns={"SMILES":"smiles"})
df_rxn.to_csv(data_dir/"reactivity.csv", index=False)

for p in ["solubility.csv","pka.csv","melting.csv","toxicity.csv","reactivity.csv"]:
    print(p, pd.read_csv(data_dir/p).shape)
```

---

## 3. Property prediction with Chemprop 1.x

Chemprop 1.x offers two standard entry points.

1) **CLI**  
- `chemprop_train` for training  
- `chemprop_predict` for inference

2) **Python API**  
- `chemprop.args.TrainArgs` then `chemprop.train.cross_validate` or `chemprop.train.run_training`  
- `chemprop.args.PredictArgs` then `chemprop.train.make_predictions`

We will show both patterns. Random split is the default. For chemistry, a **scaffold split** often provides a stricter test.

### 3.1 Solubility regression (logS)

#### Option A. CLI

```{code-cell} ipython3
!chemprop_train \
  --data_path chemprop_data/solubility.csv \
  --dataset_type regression \
  --save_dir ckpt_sol \
  --target_columns logS \
  --epochs 30 \
  --split_type scaffold_balanced \
  --metric rmse \
  --quiet
```

#### Option B. Python API

```{code-cell} ipython3
from chemprop.args import TrainArgs
from chemprop.train import cross_validate

args = TrainArgs().parse_args([
    '--data_path', str(data_dir/'solubility.csv'),
    '--dataset_type', 'regression',
    '--save_dir', 'ckpt_sol_api',
    '--target_columns', 'logS',
    '--split_type', 'scaffold_balanced',
    '--epochs', '30',
    '--metric', 'rmse',
    '--quiet'
])
results = cross_validate(args=args, train_func=None)  # uses default training
results
```

### 3.2 Test set parity plot

```{code-cell} ipython3
# Use chemprop_predict to produce predictions on the test set saved by Chemprop
# The split indices live in the save_dir. We feed the original CSV, chemprop handles splits.
!chemprop_predict \
  --test_path chemprop_data/solubility.csv \
  --checkpoint_dir ckpt_sol \
  --preds_path preds_sol.csv \
  --quiet

preds = pd.read_csv("preds_sol.csv")
truth = pd.read_csv(data_dir/"solubility.csv")["logS"]
plt.scatter(truth, preds.iloc[:,0], alpha=0.6)
lims = [min(truth.min(), preds.iloc[:,0].min())-0.2, max(truth.max(), preds.iloc[:,0].max())+0.2]
plt.plot(lims, lims, "k--")
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("True logS"); plt.ylabel("Predicted"); plt.title("Parity plot: Solubility")
plt.grid(True); plt.show()
```

```{admonition} Tip
Use `--split_type scaffold_balanced` for property prediction. For very small data consider `--split_sizes 0.8 0.1 0.1` with a fixed `--seed`.
```

---

## 4. Classification: Toxicity

```{code-cell} ipython3
!chemprop_train \
  --data_path chemprop_data/toxicity.csv \
  --dataset_type classification \
  --save_dir ckpt_tox \
  --target_columns tox_bin \
  --split_type scaffold_balanced \
  --class_balance \
  --epochs 25 \
  --metric auc \
  --quiet

!chemprop_predict \
  --test_path chemprop_data/toxicity.csv \
  --checkpoint_dir ckpt_tox \
  --preds_path preds_tox.csv \
  --quiet

proba = pd.read_csv("preds_tox.csv").iloc[:,0].values
y_true = pd.read_csv(data_dir/"toxicity.csv")["tox_bin"].astype(int).values

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
auc = roc_auc_score(y_true, proba)
acc = accuracy_score(y_true, (proba>=0.5).astype(int))
print(f"AUC: {auc:.3f}  Accuracy@0.5: {acc:.3f}")

fpr, tpr, thr = roc_curve(y_true, proba)
plt.plot(fpr, tpr, lw=2)
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC: Toxicity")
plt.grid(True); plt.show()
```

---

## 5. Reactivity classifier

Same pattern as toxicity.

```{code-cell} ipython3
!chemprop_train \
  --data_path chemprop_data/reactivity.csv \
  --dataset_type classification \
  --save_dir ckpt_rxn \
  --target_columns react_bin \
  --split_type scaffold_balanced \
  --class_balance \
  --epochs 25 \
  --metric auc \
  --quiet

!chemprop_predict \
  --test_path chemprop_data/reactivity.csv \
  --checkpoint_dir ckpt_rxn \
  --preds_path preds_rxn.csv \
  --quiet

re_proba = pd.read_csv("preds_rxn.csv").iloc[:,0].values
y_true = pd.read_csv(data_dir/"reactivity.csv")["react_bin"].astype(int).values

from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, re_proba)
print(f"Reactivity AUC: {auc:.3f}")
fpr, tpr, _ = roc_curve(y_true, re_proba)
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"k--"); plt.title("ROC: Reactivity")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(True); plt.show()
```

---

## 6. Atom-level selectivity baseline

Chemprop 1.x does not expose an atom prediction head. We will build a **baseline** outside Chemprop by training on atom-centered descriptors. Each atom becomes one sample with label 1 if it is in `site_list`, else 0. Features are Morgan fingerprints centered on that atom plus a few simple atom properties.

```{code-cell} ipython3
def atom_samples_from_smiles(smi, pos_idx_1based):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    n = m.GetNumAtoms()
    pos0 = set(i-1 for i in pos_idx_1based if 0 < i <= n)
    rows = []
    for i in range(n):
        # atom-centered ECFP
        info = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024, fromAtoms=[i])
        bits = np.array(list(fp.ToBitString()), dtype=int)
        a = m.GetAtomWithIdx(i)
        basic = [
            a.GetAtomicNum(), a.GetTotalDegree(), int(a.GetIsAromatic()),
            a.GetTotalNumHs(), a.GetFormalCharge()
        ]
        rows.append({
            "smi": smi,
            "atom_idx0": i,
            "y": 1 if i in pos0 else 0,
            "bits": bits,
            "basic": np.array(basic, dtype=float)
        })
    return rows

rows = []
for smi, sites in df[["SMILES","site_list"]].itertuples(index=False):
    rows.extend(atom_samples_from_smiles(smi, sites))
len(rows)
```

```{code-cell} ipython3
# Build design matrix
X = np.stack([np.concatenate([r["bits"], r["basic"]]) for r in rows])
y = np.array([r["y"] for r in rows])
meta = pd.DataFrame({"SMILES":[r["smi"] for r in rows], "atom_idx0":[r["atom_idx0"] for r in rows], "y": y})

# Split by molecules to avoid leakage
smis = meta["SMILES"].values
uniq = np.unique(smis)
rng = np.random.RandomState(0)
rng.shuffle(uniq)
n = len(uniq)
train_smis = set(uniq[:int(0.8*n)])
test_mask = ~np.array([s in train_smis for s in smis])
train_mask = ~test_mask

Xtr, ytr = X[train_mask], y[train_mask]
Xte, yte = X[test_mask], y[test_mask]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

clf = LogisticRegression(max_iter=2000, n_jobs=None)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:,1]
print("Atom AUC:", roc_auc_score(yte, proba), "AP:", average_precision_score(yte, proba))
```

### Visualize predicted site probabilities on a molecule

```{code-cell} ipython3
# Pick a test molecule with at least one positive site
test_smi = None
for smi in np.unique(meta[test_mask]["SMILES"]):
    sites = df.loc[df["SMILES"]==smi, "site_list"].iloc[0]
    if len(sites)>0:
        test_smi = smi
        break

m = Chem.MolFromSmiles(test_smi)
n = m.GetNumAtoms()
probs = []
for i in range(n):
    r = atom_samples_from_smiles(test_smi, [])[i]
    xb = np.concatenate([r["bits"], r["basic"]]).reshape(1,-1)
    probs.append(clf.predict_proba(xb)[0,1])
probs = np.array(probs)

# Annotate
m2 = Chem.Mol(m)
for i, a in enumerate(m2.GetAtoms()):
    a.SetProp("atomNote", f"{probs[i]:.2f}")
img = Draw.MolToImage(m2, size=(380, 340))
display(img)

print("True sites (1-based):", df.loc[df["SMILES"]==test_smi, "site_list"].iloc[0])
```

```{admonition} Note
This atom baseline is a proxy. It is valuable for teaching and quick screening. For production-grade site models with a D-MPNN head, use Chemprop 2.x or implement a custom PyTorch head.
```

---

## 7. Interpretation with SHAP

Chemprop 1.x does not expose internal feature groups in a way that supports node or edge ablation out of the box. We will demonstrate SHAP using the **descriptor baseline** from section 6 as the explainee. It still teaches key ideas: local attributions, feature importance, and what pushes a prediction up or down.

```{code-cell} ipython3
try:
    import shap
except Exception:
    %pip -q install shap
    import shap

# Use KernelExplainer on the atom baseline
background = Xtr[np.random.RandomState(0).choice(len(Xtr), size=min(200, len(Xtr)), replace=False)]
explainer = shap.KernelExplainer(clf.predict_proba, background)

# Explain a few atoms of the selected molecule
test_rows = atom_samples_from_smiles(test_smi, [])
Xt = np.stack([np.concatenate([r["bits"], r["basic"]]) for r in test_rows])
shap_values = explainer.shap_values(Xt[:1], nsamples=200)  # class 1 explanations
vals = shap_values[1].ravel()
print("Top positive features (indices):", np.argsort(vals)[-10:][::-1])
```

```{admonition} Notes
- Because we explain descriptor inputs, not raw graph messages, treat this as a proxy explanation.
- For full graph explanations on a Chemprop model, upgrade to Chemprop 2.x or use gradient-based saliency with a custom wrapper.
```

---

## 8. Train the remaining properties

Repeat the solubility pattern for **pKa** and **Melting Point**.

```{code-cell} ipython3
# pKa
!chemprop_train \
  --data_path chemprop_data/pka.csv \
  --dataset_type regression \
  --save_dir ckpt_pka \
  --target_columns pKa \
  --split_type scaffold_balanced \
  --epochs 30 \
  --metric mae \
  --quiet

# Melting Point
!chemprop_train \
  --data_path chemprop_data/melting.csv \
  --dataset_type regression \
  --save_dir ckpt_mp \
  --target_columns "Melting Point" \
  --split_type scaffold_balanced \
  --epochs 30 \
  --metric mae \
  --quiet
```

---

## 9. Save and run multi-property inference

Chemprop 1.x predicts one CSV at a time. Here is a helper that loads SMILES and runs the right checkpoints in sequence, returning a single table.

```{code-cell} ipython3
from pathlib import Path
import subprocess, tempfile

def chemprop_predict_table(smis, ckpt_dir, colname):
    tmp = Path(tempfile.mkstemp(suffix=".csv")[1])
    pd.DataFrame({"smiles": smis}).to_csv(tmp, index=False)
    out = tmp.with_suffix(".preds.csv")
    cmd = [
        "chemprop_predict",
        "--test_path", str(tmp),
        "--checkpoint_dir", str(ckpt_dir),
        "--preds_path", str(out),
        "--quiet"
    ]
    _ = subprocess.run(cmd, check=True, capture_output=True)
    vals = pd.read_csv(out).iloc[:,0].values
    tmp.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return vals

def predict_all(smis):
    return pd.DataFrame({
        "SMILES": smis,
        "Solubility_logS_pred": chemprop_predict_table(smis, "ckpt_sol", "logS"),
        "pKa_pred":              chemprop_predict_table(smis, "ckpt_pka", "pKa"),
        "MeltingPoint_pred":     chemprop_predict_table(smis, "ckpt_mp", "Melting Point"),
        "Toxicity_prob":         chemprop_predict_table(smis, "ckpt_tox", "tox_bin"),
        "Reactivity_prob":       chemprop_predict_table(smis, "ckpt_rxn", "react_bin"),
    })

demo = predict_all(df["SMILES"].head(6).tolist())
demo
```

```{admonition} ⏰ Exercise 9.1
Pick three custom SMILES and compare which property differs most across them.
```

---

## 10. Glossary • References • In-class activity with solutions

### 10.1 Glossary

- **MPNN**: Message passing neural network over the molecular graph.
- **Scaffold split**: Split by Bemis–Murcko scaffolds to test scaffold generalization.
- **AUC**: Area under ROC for binary classification.
- **Selectivity (atom level)**: Probability that a given atom is the reaction site.
- **SHAP**: Shapley values for local explanations.

### 10.2 References

- Chemprop 1.x documentation and tutorials: training, inference, splits, ensembling.
- Yang, K. et al. Analyzing Learned Molecular Representations for Property Prediction. J. Chem. Inf. Model. 2019.  
- Jiang, D. et al. Could graph neural networks learn better molecular representation for drug discovery? A comparison study. 2021.

### 10.3 In-class activity

Answer these with the 1.x toolchain.

**Q1. Solubility transform**  
Train both raw solubility and log-transformed targets. Compare RMSE.

**Q2. pKa vs Melting**  
Train separately. Report MAE. Include parity plots.

**Q3. Toxicity threshold**  
Pick a threshold that maximizes F1 on validation, then report test accuracy and AUC.

**Q4. Reactivity calibration**  
Make a reliability curve with 10 bins. Comment on calibration.

**Q5. Selectivity**  
Take five test molecules with nonempty `site_list`. For each, list the top 2 atoms by baseline probability and compare with ground truth.

---

### 10.4 Solutions

```{admonition} Q1 solution
:class: dropdown
```

```{code-cell} ipython3
# Raw solubility
df_rawsol = df[["SMILES","Solubility_mol_per_L"]].dropna().rename(columns={"SMILES":"smiles"})
df_rawsol.to_csv(data_dir/"solubility_raw.csv", index=False)

!chemprop_train \
  --data_path chemprop_data/solubility_raw.csv \
  --dataset_type regression \
  --save_dir ckpt_sol_raw \
  --target_columns Solubility_mol_per_L \
  --split_type scaffold_balanced \
  --epochs 30 \
  --metric rmse \
  --quiet

# Compare to logS run from Section 3.1
!chemprop_predict --test_path chemprop_data/solubility_raw.csv --checkpoint_dir ckpt_sol_raw --preds_path preds_raw.csv --quiet
!chemprop_predict --test_path chemprop_data/solubility.csv     --checkpoint_dir ckpt_sol     --preds_path preds_log.csv --quiet

from sklearn.metrics import mean_squared_error
raw_y  = pd.read_csv(data_dir/"solubility_raw.csv")["Solubility_mol_per_L"].values
raw_p  = pd.read_csv("preds_raw.csv").iloc[:,0].values
log_y  = pd.read_csv(data_dir/"solubility.csv")["logS"].values
log_p  = pd.read_csv("preds_log.csv").iloc[:,0].values

print("RMSE raw:", mean_squared_error(raw_y, raw_p, squared=False))
print("RMSE log:", mean_squared_error(log_y, log_p, squared=False), "(in log units)")
```

```{admonition} Q2 solution
:class: dropdown
```

```{code-cell} ipython3
# pKa parity
!chemprop_predict --test_path chemprop_data/pka.csv --checkpoint_dir ckpt_pka --preds_path preds_pka.csv --quiet
pka_t = pd.read_csv(data_dir/"pka.csv")["pKa"].values
pka_p = pd.read_csv("preds_pka.csv").iloc[:,0].values
from sklearn.metrics import mean_absolute_error
print("pKa MAE:", mean_absolute_error(pka_t, pka_p))
plt.scatter(pka_t, pka_p, alpha=0.6); plt.plot([pka_t.min(),pka_t.max()],[pka_t.min(),pka_t.max()],"k--")
plt.xlabel("True pKa"); plt.ylabel("Pred"); plt.title("Parity pKa"); plt.grid(True); plt.show()

# Melting parity
!chemprop_predict --test_path chemprop_data/melting.csv --checkpoint_dir ckpt_mp --preds_path preds_mp.csv --quiet
mp_t = pd.read_csv(data_dir/"melting.csv")["Melting Point"].values
mp_p = pd.read_csv("preds_mp.csv").iloc[:,0].values
print("Melting MAE:", mean_absolute_error(mp_t, mp_p))
plt.scatter(mp_t, mp_p, alpha=0.6); plt.plot([mp_t.min(),mp_t.max()],[mp_t.min(),mp_t.max()],"k--")
plt.xlabel("True MP"); plt.ylabel("Pred"); plt.title("Parity MP"); plt.grid(True); plt.show()
```

```{admonition} Q3 solution
:class: dropdown
```

```{code-cell} ipython3
# Use a simple validation holdout for threshold tuning
proba = pd.read_csv("preds_tox.csv").iloc[:,0].values
y_true = pd.read_csv(data_dir/"toxicity.csv")["tox_bin"].astype(int).values

from sklearn.model_selection import train_test_split
p_tr, p_va, y_tr, y_va = train_test_split(proba, y_true, test_size=0.2, random_state=0, stratify=y_true)

ths = np.linspace(0.1, 0.9, 41)
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
f1s = [f1_score(y_va, (p_va>=t).astype(int)) for t in ths]
best_t = ths[int(np.argmax(f1s))]
print("Best threshold:", best_t)

acc = accuracy_score(y_true, (proba>=best_t).astype(int))
auc = roc_auc_score(y_true, proba)
print(f"All-set Accuracy@best_t: {acc:.3f}  AUC: {auc:.3f}")
```

```{admonition} Q4 solution
:class: dropdown
```

```{code-cell} ipython3
# Reliability curve for reactivity
from sklearn.calibration import calibration_curve
frac_pos, mean_pred = calibration_curve(y_true=pd.read_csv(data_dir/"reactivity.csv")["react_bin"].astype(int),
                                        y_prob=pd.read_csv("preds_rxn.csv").iloc[:,0].values,
                                        n_bins=10, strategy="quantile")
plt.plot(mean_pred, frac_pos, "o-")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction positive")
plt.title("Reliability curve: Reactivity")
plt.grid(True); plt.show()
```

```{admonition} Q5 solution
:class: dropdown
```

```{code-cell} ipython3
# Take five molecules with nonempty site_list and compare with atom baseline
cand = df[df["site_list"].map(len)>0].head(5)
rows = []
for smi, sites in cand[["SMILES","site_list"]].itertuples(index=False):
    m = Chem.MolFromSmiles(smi)
    n = m.GetNumAtoms()
    probs = []
    for i in range(n):
        r = atom_samples_from_smiles(smi, [])[i]
        xb = np.concatenate([r["bits"], r["basic"]]).reshape(1,-1)
        probs.append(clf.predict_proba(xb)[0,1])
    top2 = (np.argsort(probs)[-2:]+1).tolist()  # 1-based
    rows.append({"SMILES": smi, "true_sites": sites, "top2_pred": top2})
pd.DataFrame(rows)
```
