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








