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








## 8. Solutions


### 8.1 Tree vs Forest on log-solubility

Goal: predict log-solubility and compare a small tree to a forest.

```{code-cell} ipython3
# Create target: log10(solubility + 1e-6) to avoid log(0)
# If your dataframe already has a numeric solubility column named 'Solubility_mol_per_L', reuse it.
df_sol = df.copy()
df_sol["y_log"] = np.log10(df_sol["Solubility_mol_per_L"] + 1e-6)

Xs = df_sol[["MolWt", "LogP", "TPSA", "NumRings"]].dropna()
ys = df_sol.loc[Xs.index, "y_log"]

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.2, random_state=42
)

# Models
tree_sol = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0).fit(Xs_train, ys_train)
rf_sol   = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1).fit(Xs_train, ys_train)

# Scores
yhat_tree = tree_sol.predict(Xs_test)
yhat_rf   = rf_sol.predict(Xs_test)

print(f"Tree R2:   {r2_score(ys_test, yhat_tree):.3f}")
print(f"Forest R2: {r2_score(ys_test, yhat_rf):.3f}")
```

Parity plots for both models.

```{code-cell} ipython3
# Parity for tree
plt.scatter(ys_test, yhat_tree, alpha=0.6)
lims = [min(ys_test.min(), yhat_tree.min()), max(ys_test.max(), yhat_tree.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Parity plot: Tree on log-solubility")
plt.show()

# Parity for forest
plt.scatter(ys_test, yhat_rf, alpha=0.6)
lims = [min(ys_test.min(), yhat_rf.min()), max(ys_test.max(), yhat_rf.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Parity plot: Forest on log-solubility")
plt.show()
```

---

### 8.2 Pruning with `min_samples_leaf`

Fix `max_depth=None` for a classifier on toxicity and sweep leaf size.

```{code-cell} ipython3
leaf_grid = [1, 2, 3, 5, 8, 12, 20]
accs = []

for leaf in leaf_grid:
    clf = DecisionTreeClassifier(max_depth=None, min_samples_leaf=leaf, random_state=0).fit(X_train, y_train)
    accs.append(accuracy_score(y_test, clf.predict(X_test)))

pd.DataFrame({"min_samples_leaf": leaf_grid, "Accuracy": np.round(accs, 3)})
```

```{code-cell} ipython3
plt.plot(leaf_grid, accs, marker="o")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy (test)")
plt.title("Pruning with min_samples_leaf")
plt.grid(True)
plt.show()
```

Hint for interpretation: very small leaves may overfit while very large leaves may underfit.

---

### 8.3 Toxicity prediction



```{code-cell} ipython3
seeds = [0, 7, 21, 42]
rows_oob = []

for s in seeds:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt", min_samples_leaf=3,
        oob_score=True, random_state=s, n_jobs=-1
    ).fit(X_tr, y_tr)
    acc_test = accuracy_score(y_te, rf.predict(X_te))
    rows_oob.append({"seed": s, "OOB": rf.oob_score_, "TestAcc": acc_test})

pd.DataFrame(rows_oob).round(3)
```

```{code-cell} ipython3
df_oob = pd.DataFrame(rows_oob)
plt.plot(df_oob["seed"], df_oob["OOB"], "o-", label="OOB")
plt.plot(df_oob["seed"], df_oob["TestAcc"], "o-", label="Test")
plt.xlabel("random_state")
plt.ylabel("Accuracy")
plt.title("OOB vs Test accuracy")
plt.grid(True)
plt.legend()
plt.show()
```


```{code-cell} ipython3

small_tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_train, y_train)

plt.figure(figsize=(7,5))
plot_tree(small_tree, feature_names=feat_names, class_names=["non_toxic","toxic"], filled=True)
plt.title("Small Decision Tree (max_depth=2)")
plt.show()

# Extract split rules programmatically for the top two levels
feat_idx = small_tree.tree_.feature
thresh = small_tree.tree_.threshold
left = small_tree.tree_.children_left
right = small_tree.tree_.children_right

def node_rule(node_id):
    f = feat_idx[node_id]
    t = thresh[node_id]
    return f"{feat_names[f]} <= {t:.3f} ?"

print("Root rule:", node_rule(0))
print("Left child rule:", node_rule(left[0]) if left[0] != -1 else "Left child is a leaf")
print("Right child rule:", node_rule(right[0]) if right[0] != -1 else "Right child is a leaf")

```

Expect OOB to track test accuracy closely. Small differences are normal.

---

### 8.4 Feature importance agreement on melting point

Compare built-in importance to permutation importance for a random forest regressor.

```{code-cell} ipython3
rf_imp = RandomForestRegressor(
    n_estimators=400, min_samples_leaf=3, max_features="sqrt",
    random_state=0, n_jobs=-1
).fit(Xr_train, yr_train)

# Built-in importance
imp_series = pd.Series(rf_imp.feature_importances_, index=Xr_train.columns).sort_values()

# Permutation importance on test
perm_r = permutation_importance(
    rf_imp, Xr_test, yr_test, scoring="r2", n_repeats=20, random_state=0
)
perm_series = pd.Series(perm_r.importances_mean, index=Xr_train.columns).sort_values()

# Plots
imp_series.plot(kind="barh")
plt.title("Random Forest feature_importances_ (regression)")
plt.show()

perm_series.plot(kind="barh")
plt.title("Permutation importance on test (regression)")
plt.show()

pd.DataFrame({"Built_in": imp_series, "Permutation": perm_series})
```

Look for agreement on the top features. Disagreements can signal correlation or overfitting in the training trees.

---

### 8.5 CV on RF



```{code-cell} ipython3

# Data
X = df_clf[["MolWt", "LogP", "TPSA", "NumRings"]]
y = df_clf["Toxicity"].str.lower().map({"toxic":1, "non_toxic":0}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, stratify=y
)

# CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", 0.8],
}

rf_base = RandomForestClassifier(
    oob_score=False, random_state=0, n_jobs=-1
)

grid = GridSearchCV(
    rf_base,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=False,
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print(f"Best CV AUC: {grid.best_score_:.3f}")

# Refit on full training data already done by refit=True
rf_best = grid.best_estimator_

# Test metrics
y_hat = rf_best.predict(X_test)
y_proba = rf_best.predict_proba(X_test)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test, y_hat):.3f}")
print(f"Test AUC: {roc_auc_score(y_test, y_proba):.3f}")

# ROC plot
fpr, tpr, thr = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, lw=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve - RF with CV-tuned hyperparameters")
plt.show()

```

Read the rules as binary questions. Samples that satisfy a rule go left. Others go right.




---

## 11. Solutions

### Solution Q1

```{code-cell} ipython3
df_reg = df[["MolWt","LogP","TPSA","NumRings","Melting Point"]].dropna()
X = df_reg[["MolWt","LogP","TPSA","NumRings"]].values
y = df_reg["Melting Point"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(32,), activation="relu",
                         alpha=1e-3, learning_rate_init=0.01,
                         max_iter=1500, random_state=0))
]).fit(Xtr, ytr)

yhat = pipe.predict(Xte)
print(f"MSE={mean_squared_error(yte,yhat):.2f}  MAE={mean_absolute_error(yte,yhat):.2f}  R2={r2_score(yte,yhat):.3f}")

plt.figure(figsize=(4.5,4))
plt.scatter(yte, yhat, alpha=0.65)
lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True MP"); plt.ylabel("Pred MP"); plt.title("Q1 parity")
plt.show()
```

### Solution Q2

```{code-cell} ipython3
sizes = [(16,), (32,), (64,32)]
df_sol = df[["MolWt","LogP","TPSA","NumRings","Solubility_mol_per_L"]].dropna().copy()
df_sol["logS"] = np.log10(df_sol["Solubility_mol_per_L"]+1e-6)
X = df_sol[["MolWt","LogP","TPSA","NumRings"]].values
y = df_sol["logS"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=15)

r2s, curves = [], []
for sz in sizes:
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=sz, activation="relu",
                             alpha=1e-3, learning_rate_init=0.01,
                             early_stopping=True, validation_fraction=0.15,
                             max_iter=3000, random_state=0))
    ]).fit(Xtr, ytr)
    yhat = reg.predict(Xte)
    r2s.append(r2_score(yte, yhat))
    curves.append(reg.named_steps["mlp"].loss_curve_)

print(pd.DataFrame({"hidden_sizes":[str(s) for s in sizes],"R2":np.round(r2s,3)}))

plt.figure(figsize=(5.5,3.5))
for sz, c in zip(sizes, curves):
    plt.plot(c, label=str(sz))
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Q2 loss curves")
plt.legend(); plt.show()
```

### Solution Q3

```{code-cell} ipython3
df_clf = df[["MolWt","LogP","TPSA","NumRings","Toxicity"]].dropna()
y = df_clf["Toxicity"].str.lower().map({"toxic":1,"non_toxic":0}).astype(int).values
X = df_clf[["MolWt","LogP","TPSA","NumRings"]].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                          alpha=1e-3, learning_rate_init=0.01,
                          early_stopping=True, validation_fraction=0.15,
                          max_iter=3000, random_state=0))
]).fit(Xtr, ytr)

proba = clf.predict_proba(Xte)[:,1]
for t in [0.3, 0.5, 0.7]:
    pred = (proba >= t).astype(int)
    print(f"t={t:.1f}  acc={accuracy_score(yte,pred):.3f}  prec={precision_score(yte,pred):.3f}  rec={recall_score(yte,pred):.3f}  f1={f1_score(yte,pred):.3f}")
```

### Solution Q4

```{code-cell} ipython3
df_sol = df[["MolWt","LogP","TPSA","NumRings","Solubility_mol_per_L"]].dropna().copy()
df_sol["logS"] = np.log10(df_sol["Solubility_mol_per_L"]+1e-6)
X = df_sol[["MolWt","LogP","TPSA","NumRings"]].values
y = df_sol["logS"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=15)

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

lr = LinearRegression().fit(Xtr_s, ytr)
yhat_lr = lr.predict(Xte_s)

mlp = MLPRegressor(hidden_layer_sizes=(32,), activation="relu",
                   alpha=1e-3, learning_rate_init=0.01,
                   max_iter=3000, random_state=0).fit(Xtr_s, ytr)
yhat_mlp = mlp.predict(Xte_s)

print(f"Linear R2: {r2_score(yte, yhat_lr):.3f}")
print(f"MLP    R2: {r2_score(yte, yhat_mlp):.3f}")

plt.figure(figsize=(5.5,4))
plt.scatter(yte, yhat_lr, alpha=0.6, label="Linear")
plt.scatter(yte, yhat_mlp, alpha=0.6, label="MLP")
lims = [min(yte.min(), yhat_lr.min(), yhat_mlp.min()), max(yte.max(), yhat_lr.max(), yhat_mlp.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True logS"); plt.ylabel("Predicted")
plt.legend(); plt.title("Q4 parity: Linear vs MLP")
plt.show()
```

### Solution Q5

```{code-cell} ipython3
# Solution Q5 (full run + metrics + plots)

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Data
df_mp = df[["MolWt","LogP","TPSA","NumRings","Melting Point"]].dropna().copy()

X = df_mp[["MolWt","LogP","TPSA","NumRings"]].values.astype(np.float32)
y = df_mp["Melting Point"].values.astype(np.float32).reshape(-1,1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=15)
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr).astype(np.float32)
Xte_s = scaler.transform(Xte).astype(np.float32)

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(NumpyDataset(Xtr_s, ytr), batch_size=64, shuffle=True)

in_dim = Xtr_s.shape[1]
model = nn.Sequential(
    nn.Linear(in_dim, 32), nn.ReLU(),
    nn.Linear(32, 16),     nn.ReLU(),
    nn.Linear(16, 1)
)

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

train_losses = []
model.train()
for epoch in range(200):
    batch_losses = []
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

# Evaluate
model.eval()
with torch.no_grad():
    yhat = model(torch.from_numpy(Xte_s)).numpy()

print(f"MSE: {mean_squared_error(yte, yhat):.3f}")
print(f"MAE: {mean_absolute_error(yte, yhat):.3f}")
print(f"R2:  {r2_score(yte, yhat):.3f}")

# Learning curve
plt.figure(figsize=(5,3))
plt.plot(train_losses)
plt.xlabel("epoch"); plt.ylabel("train MSE"); plt.title("Training loss (melting point)")
plt.grid(alpha=0.3)
plt.show()

# Parity plot
plt.figure(figsize=(4.6,4))
plt.scatter(yte, yhat, alpha=0.65)
lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True MP"); plt.ylabel("Pred MP"); plt.title("Parity plot (PyTorch MP)")
plt.show()

```








