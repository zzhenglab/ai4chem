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

# Lecture 7 - Decision Trees and Random Forests

```{contents}
:local:
:depth: 2
```

## 1. Setup and data

```{code-cell} ipython3
:tags: [hide-input]
# If you are on Colab, you may need:
# %pip install scikit-learn pandas matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

# Optional RDKit for descriptors (used in Lectures 5 and 6)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw
except Exception:
    print("RDKit not available. Descriptor drawing will be skipped.")
    Chem = None

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.tree import cost_complexity_pruning_path
```

We will reuse the C–H oxidation dataset and the same four lightweight descriptors: `MolWt`, `LogP`, `TPSA`, `NumRings`.

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)

def calc_desc(smiles):
    if Chem is None:
        return pd.Series({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan})
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return pd.Series({"MolWt": np.nan, "LogP": np.nan, "TPSA": np.nan, "NumRings": np.nan})
    return pd.Series({
        "MolWt": Descriptors.MolWt(m),
        "LogP": Crippen.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "NumRings": rdMolDescriptors.CalcNumRings(m)
    })

desc_df = df_raw["SMILES"].apply(calc_desc)
df = pd.concat([df_raw, desc_df], axis=1)

feat = ["MolWt", "LogP", "TPSA", "NumRings"]
X_all = df[feat]
print("Rows:", len(df))
X_all.describe().round(2)
```

```{admonition} What we will predict
- **Regression** target: `Melting Point`  
- **Classification** target: `Toxicity` mapped to 1 for toxic and 0 for non_toxic
```

---

## 2. Decision trees - intuition and API

```{admonition} Idea
A decision tree learns a sequence of questions like `MolWt <= 200.5`. Each split aims to make child nodes purer.
```

- **Regression tree** chooses splits that reduce **MSE** the most. A leaf predicts the **mean** of training `y` within that leaf.
- **Classification tree** chooses splits that reduce **Gini** or **entropy**. A leaf predicts the **majority class** and class probabilities.

Key hyperparameters you will tune frequently:
- `max_depth` - maximum levels of splits
- `min_samples_split` - minimum samples required to attempt a split
- `min_samples_leaf` - minimum samples allowed in a leaf
- `max_features` - number of features to consider when finding the best split

Trees handle different feature scales naturally, and they do not require standardization. They can struggle with high noise and very small datasets if left unconstrained.

```{admonition} Vocabulary
- **Node** is a point where a question is asked.  
- **Leaf** holds a simple prediction.  
- **Impurity** is a measure of how mixed a node is. Lower is better.
```

---

## 3. Tree regression on Melting Point

### 3.1 Prepare `X` and `y`

```{code-cell} ipython3
X = df[feat]
y = df["Melting Point"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

X_tr.shape, X_te.shape, y_tr.shape, y_te.shape
```

### 3.2 Fit a tiny stump to see one split

```{code-cell} ipython3
tree_stump = DecisionTreeRegressor(max_depth=1, random_state=0)
tree_stump.fit(X_tr, y_tr)

print("Train R2:", r2_score(y_tr, tree_stump.predict(X_tr)).round(3))
print("Test  R2:", r2_score(y_te, tree_stump.predict(X_te)).round(3))
print("Importances:", dict(zip(feat, np.round(tree_stump.feature_importances_, 3))))
```

```{code-cell} ipython3
plt.figure(figsize=(8,4))
plot_tree(tree_stump, feature_names=feat, filled=True, rounded=True)
plt.title("DecisionTreeRegressor depth=1")
plt.show()

print(export_text(tree_stump, feature_names=feat))
```

### 3.3 Increase depth and watch train vs test

```{code-cell} ipython3
depths = list(range(1, 11))
r2_tr, r2_te = [], []

for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=0)
    m.fit(X_tr, y_tr)
    r2_tr.append(r2_score(y_tr, m.predict(X_tr)))
    r2_te.append(r2_score(y_te, m.predict(X_te)))

plt.figure(figsize=(6,4))
plt.plot(depths, r2_tr, "o-", label="Train R2")
plt.plot(depths, r2_te, "o-", label="Test R2")
plt.xlabel("max_depth"); plt.ylabel("R2"); plt.title("Depth sweep - regression")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 3.4 Choose a reasonable depth and inspect fit

```{code-cell} ipython3
best_depth = 4
tree_reg = DecisionTreeRegressor(max_depth=best_depth, random_state=0).fit(X_tr, y_tr)

y_hat = tree_reg.predict(X_te)
plt.figure(figsize=(5,4))
plt.scatter(y_te, y_hat, alpha=0.6)
lims = [min(y_te.min(), y_hat.min()), max(y_te.max(), y_hat.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True MP"); plt.ylabel("Predicted MP")
plt.title("Parity plot - tree regression")
plt.show()

pd.Series(tree_reg.feature_importances_, index=feat).round(3)
```

---

## 4. Tree classification on Toxicity

### 4.1 Encode label and split

```{code-cell} ipython3
lab_map = {"toxic": 1, "non_toxic": 0}
y_cls = df["Toxicity"].str.lower().map(lab_map)

mask = y_cls.notna() & X.notna().all(axis=1)
Xc = X.loc[mask]; yc = y_cls.loc[mask].astype(int)

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.2, random_state=42, stratify=yc
)
Xc_tr.shape, yc_tr.value_counts(normalize=True).round(3)
```

### 4.2 Depth 1 then sweep `max_depth`

```{code-cell} ipython3
clf_stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xc_tr, yc_tr)
proba = clf_stump.predict_proba(Xc_te)[:,1]
pred  = (proba >= 0.5).astype(int)

print("Accuracy:", accuracy_score(yc_te, pred).round(3))
print("Precision:", precision_score(yc_te, pred).round(3))
print("Recall:", recall_score(yc_te, pred).round(3))
print("AUC:", roc_auc_score(yc_te, proba).round(3))
```

```{code-cell} ipython3
depths = list(range(1, 11))
accs, aucs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xc_tr, yc_tr)
    pr = m.predict_proba(Xc_te)[:,1]
    pd_ = (pr >= 0.5).astype(int)
    accs.append(accuracy_score(yc_te, pd_))
    aucs.append(roc_auc_score(yc_te, pr))

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(depths, accs, "o-"); ax[0].set_xlabel("max_depth"); ax[0].set_ylabel("Accuracy"); ax[0].grid(True, alpha=0.3)
ax[1].plot(depths, aucs, "o-"); ax[1].set_xlabel("max_depth"); ax[1].set_ylabel("AUC"); ax[1].grid(True, alpha=0.3)
plt.suptitle("Depth sweep - classification")
plt.show()
```

```{code-cell} ipython3
d_pick = 4
clf_d = DecisionTreeClassifier(max_depth=d_pick, random_state=0).fit(Xc_tr, yc_tr)
cm = confusion_matrix(yc_te, clf_d.predict(Xc_te))
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap="Blues"); plt.title(f"Confusion matrix - depth={d_pick}")
plt.xlabel("Predicted"); plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
```

---

## 5. Overfitting and regularization

Trees can overfit easily when depth is large or when leaves are tiny. You will see:
- **Very high train R2 or train accuracy**
- **Noticeably lower test R2 or test accuracy**

You can reduce variance by:
- Limiting depth with `max_depth`
- Requiring more samples in leaves using `min_samples_leaf`
- Requiring more samples to split using `min_samples_split`
- Cost complexity pruning using `ccp_alpha`

### 5.1 Bias vs variance picture using depth curves

We already plotted train vs test curves with a depth sweep for regression and classification. The gap between train and test curves grows when the model overfits. Pick a region where the test curve plateaus and the gap is small.

### 5.2 Minimum leaf size

```{code-cell} ipython3
leaf_sizes = [1, 2, 5, 10, 20, 40, 80]
r2_leaf = []
for leaf in leaf_sizes:
    m = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=0).fit(X_tr, y_tr)
    r2_leaf.append(r2_score(y_te, m.predict(X_te)))

pd.DataFrame({"min_samples_leaf": leaf_sizes, "test_R2": np.round(r2_leaf, 3)})
```

### 5.3 Cost complexity pruning path

Scikit-learn can compute a sequence of pruned trees controlled by `ccp_alpha`. Larger `ccp_alpha` means stronger pruning.

```{code-cell} ipython3
# Start from a relatively deep tree
deep_tree = DecisionTreeRegressor(random_state=0).fit(X_tr, y_tr)
path = cost_complexity_pruning_path(deep_tree, X_tr, y_tr)
ccp_alphas = path.ccp_alphas

# Train along the path
r2_te_alpha, r2_tr_alpha = [], []
for a in ccp_alphas:
    m = DecisionTreeRegressor(random_state=0, ccp_alpha=a).fit(X_tr, y_tr)
    r2_tr_alpha.append(r2_score(y_tr, m.predict(X_tr)))
    r2_te_alpha.append(r2_score(y_te, m.predict(X_te)))

plt.figure(figsize=(6,4))
plt.plot(ccp_alphas, r2_tr_alpha, marker="o", label="Train R2")
plt.plot(ccp_alphas, r2_te_alpha, marker="o", label="Test R2")
plt.xlabel("ccp_alpha"); plt.ylabel("R2"); plt.title("Pruning curve")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

```{admonition} Takeaway
Trees do not have to be deep to work well. A small amount of pruning or a modest leaf size can improve test performance and stability.
```

---

## 6. Random Forests

A Random Forest builds many trees on bootstrap samples and averages their predictions. Each split considers a random subset of features, which decorrelates trees.

### 6.1 Regression with OOB estimate

```{code-cell} ipython3
rf_reg = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    oob_score=True,
    n_jobs=-1,
)
rf_reg.fit(X_tr, y_tr)

print("OOB R2:", getattr(rf_reg, "oob_score_", None))
print("Test R2:", r2_score(y_te, rf_reg.predict(X_te)).round(3))
```

```{code-cell} ipython3
# n_estimators curve
ests = [20, 50, 100, 200, 300, 500]
r2s = []
for n in ests:
    m = RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    r2s.append(r2_score(y_te, m.predict(X_te)))

plt.figure(figsize=(6,4))
plt.plot(ests, r2s, "o-")
plt.xlabel("n_estimators"); plt.ylabel("Test R2"); plt.title("Forest size vs R2")
plt.grid(True, alpha=0.3); plt.show()
```

```{code-cell} ipython3
# Importance and permutation importance
imp_rf = pd.Series(rf_reg.feature_importances_, index=feat).sort_values(ascending=True)
imp_rf.plot(kind="barh", figsize=(5,3)); plt.title("RF regression - feature importance"); plt.show()

perm = permutation_importance(rf_reg, X_te, y_te, scoring="r2", n_repeats=20, random_state=0)
pd.Series(perm.importances_mean, index=feat).sort_values().plot(kind="barh", figsize=(5,3))
plt.title("Permutation importance - drop in R2"); plt.xlabel("Mean decrease in R2"); plt.show()
```

```{code-cell} ipython3
# Partial dependence for one feature
fig = plt.figure(figsize=(5,4))
PartialDependenceDisplay.from_estimator(rf_reg, X, ["MolWt"], ax=plt.gca())
plt.title("Partial dependence - MolWt (RF regression)")
plt.show()
```

### 6.2 Classification with ROC

```{code-cell} ipython3
rf_clf = RandomForestClassifier(
    n_estimators=400,
    random_state=0,
    oob_score=True,
    n_jobs=-1,
)
rf_clf.fit(Xc_tr, yc_tr)

proba = rf_clf.predict_proba(Xc_te)[:,1]
pred  = (proba >= 0.5).astype(int)
print("OOB accuracy:", getattr(rf_clf, "oob_score_", None))
print("Test accuracy:", accuracy_score(yc_te, pred).round(3))
print("Test AUC:", roc_auc_score(yc_te, proba).round(3))
```

```{code-cell} ipython3
# ROC
fpr, tpr, thr = roc_curve(yc_te, proba)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(yc_te, proba):.3f}")
plt.plot([0,1],[0,1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - RF classifier"); plt.legend(); plt.show()
```

```{code-cell} ipython3
# Classification importances
pd.Series(rf_clf.feature_importances_, index=feat).sort_values().plot(kind="barh", figsize=(5,3))
plt.title("RF classification - feature importance"); plt.show()
```

```{admonition} Why forests help
A single deep tree can fit noise. Averaging many diverse trees reduces variance and often boosts test performance.
```

---

## 7. Tuning and validation

We will use CV to pick hyperparameters for both tree and forest models. Keep grids compact so the run is quick in class.

### 7.1 Decision tree regression grid

```{code-cell} ipython3
param_grid_dt = {
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_leaf": [1, 2, 5, 10],
    "min_samples_split": [2, 5, 10]
}

dt = DecisionTreeRegressor(random_state=0)
cv = KFold(n_splits=4, shuffle=True, random_state=1)
grid_dt = GridSearchCV(dt, param_grid_dt, cv=cv, scoring="r2", n_jobs=-1)
grid_dt.fit(X_tr, y_tr)

best_dt = grid_dt.best_estimator_
print("Best params:", grid_dt.best_params_)
print("CV mean R2:", grid_dt.best_score_.round(3))
print("Test R2:", r2_score(y_te, best_dt.predict(X_te)).round(3))
```

### 7.2 Random forest regression grid

```{code-cell} ipython3
param_grid_rf = {
    "n_estimators": [200, 300],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["auto", "sqrt"]
}

rf = RandomForestRegressor(random_state=0, n_jobs=-1)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=cv, scoring="r2", n_jobs=-1)
grid_rf.fit(X_tr, y_tr)

best_rf = grid_rf.best_estimator_
print("Best params:", grid_rf.best_params_)
print("CV mean R2:", grid_rf.best_score_.round(3))
print("Test R2:", r2_score(y_te, best_rf.predict(X_te)).round(3))
```

### 7.3 Decision tree classification grid

```{code-cell} ipython3
param_grid_dtc = {
    "max_depth": [2, 3, 4, 6, None],
    "min_samples_leaf": [1, 2, 5, 10]
}

dtc = DecisionTreeClassifier(random_state=0)
cv_c = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
grid_dtc = GridSearchCV(dtc, param_grid_dtc, cv=cv_c, scoring="roc_auc", n_jobs=-1)
grid_dtc.fit(Xc_tr, yc_tr)

best_dtc = grid_dtc.best_estimator_
print("Best params:", grid_dtc.best_params_)
print("CV mean AUC:", grid_dtc.best_score_.round(3))
print("Test AUC:", roc_auc_score(yc_te, best_dtc.predict_proba(Xc_te)[:,1]).round(3))
```

### 7.4 Random forest classification grid

```{code-cell} ipython3
param_grid_rfc = {
    "n_estimators": [200, 400],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["auto", "sqrt"]
}

rfc = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_rfc = GridSearchCV(rfc, param_grid_rfc, cv=cv_c, scoring="roc_auc", n_jobs=-1)
grid_rfc.fit(Xc_tr, yc_tr)

best_rfc = grid_rfc.best_estimator_
proba_best = best_rfc.predict_proba(Xc_te)[:,1]
print("Best params:", grid_rfc.best_params_)
print("CV mean AUC:", grid_rfc.best_score_.round(3))
print("Test AUC:", roc_auc_score(yc_te, proba_best).round(3))
```

---

## 8. Interpretability and diagnostics

This section groups useful tools for understanding models and diagnosing issues.

### 8.1 Compare a single tree to a forest

```{code-cell} ipython3
tree_r = DecisionTreeRegressor(max_depth=6, random_state=0).fit(X_tr, y_tr)
rf_r   = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)

print("Tree test R2:", r2_score(y_te, tree_r.predict(X_te)).round(3))
print("Forest test R2:", r2_score(y_te, rf_r.predict(X_te)).round(3))
```

### 8.2 Inspect one tree from a forest

```{code-cell} ipython3
one_tree = best_rf.estimators_[0] if 'best_rf' in globals() else rf_r.estimators_[0]
plt.figure(figsize=(10,5))
plot_tree(one_tree, feature_names=feat, filled=True, rounded=True, max_depth=3)
plt.title("One tree from the Random Forest - top 3 levels")
plt.show()

print(export_text(one_tree, feature_names=feat, max_depth=3))
```

### 8.3 Importance choices

- `feature_importances_` is based on impurity reduction and can prefer variables with many possible splits.
- `permutation_importance` measures performance drop when a feature is shuffled. Use it on a held out split.

### 8.4 Partial dependence

Use `PartialDependenceDisplay.from_estimator` on a fitted forest to show the average effect of a feature while marginalizing others. Good for monotonic trends and rough response shapes.

---

## 9. In-class activities

Each task is short and uses the chapter material. Fill in the `...` lines where shown.

### Q1. Tree regression - sweep `min_samples_leaf`

Use `DecisionTreeRegressor` on Melting Point with `min_samples_leaf` in `[1, 2, 5, 10, 20, 40]` and `max_depth=None`. Plot test R2 vs `min_samples_leaf` on a log-x scale.

```python
# Starter
# values = [1, 2, 5, 10, 20, 40]
# r2s = []
# for v in values:
#     m = DecisionTreeRegressor(min_samples_leaf=v, random_state=0).fit(X_tr, y_tr)
#     r2s.append(r2_score(y_te, m.predict(X_te)))
# plt.plot(values, r2s, "o-"); plt.xscale("log")
# plt.xlabel("min_samples_leaf"); plt.ylabel("Test R2"); plt.title("Leaf size sweep")
# plt.show()
```

### Q2. Tree classification - threshold tuning

Train a depth 4 tree on toxicity. Scan thresholds from `0.2` to `0.8` in steps of `0.05`. Find the smallest threshold with **recall ≥ 0.80** and report the corresponding **precision** and **F1**.

```python
# Starter
# clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xc_tr, yc_tr)
# proba = clf.predict_proba(Xc_te)[:,1]
# ths = np.arange(0.20, 0.81, 0.05)
# rec_list, prec_list, f1_list = [], [], []
# best_t = None
# for t in ths:
#     pred_t = (proba >= t).astype(int)
#     r = recall_score(yc_te, pred_t)
#     p = precision_score(yc_te, pred_t, zero_division=0)
#     f = f1_score(yc_te, pred_t, zero_division=0)
#     rec_list.append(r); prec_list.append(p); f1_list.append(f)
#     if best_t is None and r >= 0.80:
#         best_t = t
# print("First threshold with recall >= 0.80:", best_t)
```

### Q3. Forest regression - n_estimators curve

Train `RandomForestRegressor` with `n_estimators` in `[50, 100, 200, 300, 500]` and record test R2. Plot R2 vs `n_estimators`.

```python
# Starter
# ns = [50, 100, 200, 300, 500]
# r2s = []
# for n in ns:
#     m = RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
#     r2s.append(r2_score(y_te, m.predict(X_te)))
# plt.plot(ns, r2s, "o-"); plt.xlabel("n_estimators"); plt.ylabel("Test R2"); plt.title("R2 vs forest size")
# plt.show()
```

### Q4. Forest classification - permutation importance

Train `RandomForestClassifier` with `n_estimators=300` on toxicity. Compute permutation importance on the test split with the `roc_auc` scorer and plot.

```python
# Starter
# rfc = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xc_tr, yc_tr)
# perm = permutation_importance(rfc, Xc_te, yc_te, scoring="roc_auc", n_repeats=20, random_state=0)
# pd.Series(perm.importances_mean, index=feat).sort_values().plot(kind="barh")
# plt.title("Permutation importance (AUC drop)")
# plt.show()
```

### Q5. End to end - small forest grid

Use GridSearchCV to tune a small forest for Melting Point with:
- `n_estimators`: `[200, 300]`
- `max_depth`: `[None, 6, 10]`
- `min_samples_leaf`: `[1, 2, 5]`
- `max_features`: `["auto", "sqrt"]`

Report best params, CV mean R2, and test R2. Predict for three SMILES of your choice after computing descriptors.

```python
# Starter
# param_grid = {...}
# rf = RandomForestRegressor(random_state=0, n_jobs=-1)
# grid = GridSearchCV(rf, param_grid, cv=KFold(n_splits=4, shuffle=True, random_state=1), scoring="r2", n_jobs=-1)
# grid.fit(X_tr, y_tr)
# best_rf = grid.best_estimator_
# print(grid.best_params_, grid.best_score_)
# print("Test R2:", r2_score(y_te, best_rf.predict(X_te)))
```

---

## 10. Solutions to in-class activities

### Solution Q1

```{code-cell} ipython3
values = [1, 2, 5, 10, 20, 40]
r2s = []
for v in values:
    m = DecisionTreeRegressor(min_samples_leaf=v, random_state=0).fit(X_tr, y_tr)
    r2s.append(r2_score(y_te, m.predict(X_te)))
plt.figure(figsize=(6,4))
plt.plot(values, r2s, "o-"); plt.xscale("log")
plt.xlabel("min_samples_leaf"); plt.ylabel("Test R2"); plt.title("Leaf size sweep")
plt.grid(True, alpha=0.3); plt.show()
pd.DataFrame({"min_samples_leaf": values, "test_R2": np.round(r2s,3)})
```

### Solution Q2

```{code-cell} ipython3
clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(Xc_tr, yc_tr)
proba = clf.predict_proba(Xc_te)[:,1]
ths = np.arange(0.20, 0.81, 0.05)
rec_list, prec_list, f1_list = [], [], []
best_t = None
for t in ths:
    pred_t = (proba >= t).astype(int)
    r = recall_score(yc_te, pred_t)
    p = precision_score(yc_te, pred_t, zero_division=0)
    f = f1_score(yc_te, pred_t, zero_division=0)
    rec_list.append(r); prec_list.append(p); f1_list.append(f)
    if best_t is None and r >= 0.80:
        best_t = t

print("First threshold with recall >= 0.80:", best_t)
plt.figure(figsize=(7,5))
plt.plot(ths, rec_list, marker="o", label="Recall")
plt.plot(ths, prec_list, marker="o", label="Precision")
plt.plot(ths, f1_list, marker="o", label="F1")
plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold tuning on toxicity (tree)")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

### Solution Q3

```{code-cell} ipython3
ns = [50, 100, 200, 300, 500]
r2s = []
for n in ns:
    m = RandomForestRegressor(n_estimators=n, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    r2s.append(r2_score(y_te, m.predict(X_te)))
plt.figure(figsize=(6,4))
plt.plot(ns, r2s, "o-")
plt.xlabel("n_estimators"); plt.ylabel("Test R2"); plt.title("R2 vs forest size")
plt.grid(True, alpha=0.3); plt.show()
pd.DataFrame({"n_estimators": ns, "test_R2": np.round(r2s,3)})
```

### Solution Q4

```{code-cell} ipython3
rfc_sol = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xc_tr, yc_tr)
perm = permutation_importance(rfc_sol, Xc_te, yc_te, scoring="roc_auc", n_repeats=20, random_state=0)
pd.Series(perm.importances_mean, index=feat).sort_values().plot(kind="barh", figsize=(5,3))
plt.title("Permutation importance (AUC drop)"); plt.xlabel("Mean decrease in AUC")
plt.show()
```

### Solution Q5

```{code-cell} ipython3
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["auto", "sqrt"]
}
rf = RandomForestRegressor(random_state=0, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=KFold(n_splits=4, shuffle=True, random_state=1), scoring="r2", n_jobs=-1)
grid.fit(X_tr, y_tr)

best_rf_final = grid.best_estimator_
print("Best params:", grid.best_params_)
print("CV mean R2:", grid.best_score_.round(3))
print("Test R2:", r2_score(y_te, best_rf_final.predict(X_te)).round(3))

# Predict three sample SMILES if RDKit is available
smiles_three = ["C(F)(F)(F)CC=CCO", "C1CCCC(COC)C1", "CC(CBr)CCl"]
if Chem is not None:
    desc = pd.DataFrame([calc_desc(s) for s in smiles_three])[feat]
    preds = best_rf_final.predict(desc)
    print(pd.DataFrame({"SMILES": smiles_three, "Predicted MP": preds.round(1)}))
else:
    print("RDKit not available. Skipping SMILES prediction.")
```
