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

# Lecture 8 - Neural Networks

```{contents}
:local:
:depth: 1
```

## Learning goals

- Build intuition for neurons, layers, activation functions, loss, and optimization.
- Create a first neural network for a toy regression, then for chemistry data.
- Use PyTorch tensors, `Dataset`, `DataLoader`, and a simple training loop.
- Track shapes at each step and visualize learning curves.
- Compare to models from Lectures 5-7 and connect ideas like splits and metrics.

---

## 0. Setup

We start light, then switch to the C-H dataset you used in earlier lectures.  
If RDKit is missing, code will skip molecule drawings but still compute with the CSV.

```{code-cell} ipython3
:tags: [hide-input]
# 0. Setup
%pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
%pip -q install pandas numpy matplotlib scikit-learn

# RDKit is optional for descriptors already stored in the CSV
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
except Exception:
    Chem = None

import numpy as np, pandas as pd, matplotlib.pyplot as plt, math, time, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
torch.__version__
```

---

## 1. What is a neural network

```{admonition} Picture in words
A **neuron** computes `z = w·x + b`, then applies a **nonlinearity** `a = σ(z)`.
A **layer** stacks many neurons.
A **network** stacks layers.
Training finds weights `w, b` that minimize a **loss** on your data.
```

**Key pieces**

- **Activation**: ReLU, Sigmoid, Tanh.  
- **Loss**: MSE for regression, Cross-Entropy for classification.  
- **Optimizer**: Gradient descent variants (SGD, Adam).  
- **Epoch**: one pass over the training set.

We will not jump into a large script. We will build the pipeline in tiny steps, check shapes, and talk through each part.

---

## 2. Tensors in PyTorch: a 2-minute tour

```{code-cell} ipython3
# Scalars, vectors, matrices
t_scalar = torch.tensor(3.14)
t_vec    = torch.tensor([1.0, 2.0, 3.0])
t_mat    = torch.tensor([[1., 2.], [3., 4.]])

t_scalar.shape, t_vec.shape, t_mat.shape
```

```{admonition} Tip
Use `.shape` often. In deep learning, many bugs are shape bugs.
```

```{code-cell} ipython3
# Basic ops
x = torch.tensor([[2., -1.],[0.5, 4.]])
y = torch.tensor([[1.,  3.],[2.0, 1.]])
x + y, x @ y
```

```{admonition} Exercise 2.1
Create a 3x4 tensor of ones, multiply by 2, then compute its mean.
Confirm the shape at each step.
```

---

## 3. A first neural net on a toy regression

Before touching chemistry, we learn the loop on a simple function: `y = sin(x)` with noise. One input, one output. This shows how a network learns a curve.

### 3.1 Make a tiny dataset

```{code-cell} ipython3
rng = np.random.default_rng(0)
x_np = np.linspace(-3*np.pi, 3*np.pi, 400).astype(np.float32)
y_np = np.sin(x_np) + rng.normal(0, 0.1, size=x_np.shape).astype(np.float32)

plt.figure(figsize=(5,3))
plt.scatter(x_np, y_np, s=10, alpha=0.6)
plt.xlabel("x"); plt.ylabel("y = sin(x) + noise"); plt.title("Toy regression")
plt.show()
```

### 3.2 Convert to tensors and check shapes

```{code-cell} ipython3
X = torch.from_numpy(x_np).reshape(-1, 1)  # (N, 1)
y = torch.from_numpy(y_np).reshape(-1, 1)  # (N, 1)
X.shape, y.shape
```

### 3.3 Define a small network

We choose a compact multilayer perceptron (MLP): `1 -> 64 -> 64 -> 1` with ReLU.

```{code-cell} ipython3
class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = TinyRegressor()
model
```

### 3.4 Loss and optimizer

```{code-cell} ipython3
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
```

### 3.5 One manual training step (by hand)

We do a single gradient step to see each operation. This is not a full loop yet.

```{code-cell} ipython3
model.train()
opt.zero_grad()

y_pred = model(X)           # forward
loss = loss_fn(y_pred, y)   # scalar tensor
loss_item_before = loss.item()

loss.backward()             # compute gradients
opt.step()                  # update weights

loss_item_before, loss_fn(model(X), y).item()
```

```{admonition} Checkpoint
You saw forward, loss, backward, step. This is the basic learning step.
Repeat many times over mini-batches to train.
```

### 3.6 Mini training loop

We train for a few epochs and track training loss.

```{code-cell} ipython3
model = TinyRegressor()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

losses = []
for epoch in range(400):
    opt.zero_grad()
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

plt.figure(figsize=(5,3))
plt.plot(losses)
plt.xlabel("epoch"); plt.ylabel("train MSE"); plt.title("Training curve")
plt.show()
```

### 3.7 Visualize predictions

```{code-cell} ipython3
model.eval()
with torch.no_grad():
    y_fit = model(X).numpy()

plt.figure(figsize=(5,3))
plt.scatter(X.numpy(), y.numpy(), s=10, alpha=0.3, label="data")
plt.plot(X.numpy(), y_fit, lw=2, label="model")
plt.legend(); plt.title("Fit on toy regression"); plt.show()
```

```{admonition} Exercise 3.1
Change hidden width from 64 to 16. Retrain and compare the training loss and the curve.
Which model underfits or overfits more on this toy?
```

```{admonition} Exercise 3.2
Try `Tanh` instead of `ReLU`. Keep architecture the same.
Is convergence slower or faster with the default learning rate?
```

---

## 4. Neural nets for chemistry: regression on Melting Point

We now switch to a familiar table. As in Lectures 5-7, we will use four simple descriptors as features.

### 4.1 Load data and build feature matrix

```{admonition} Data
`MolWt`, `LogP`, `TPSA`, `NumRings` will be our `X`.
Target `y` will be `Melting Point`.
```

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)

# If RDKit available, we can recompute. Otherwise reuse stored numeric columns where present.
def descriptors_from_smiles(smiles):
    if Chem is None:
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

# Compute once if needed
desc = df_raw["SMILES"].apply(descriptors_from_smiles)
df = pd.concat([df_raw, desc], axis=1)

use_cols = ["MolWt","LogP","TPSA","NumRings","Melting Point"]
df_reg = df[use_cols].dropna().reset_index(drop=True)
df_reg.head()
```

### 4.2 Train, validation, test split

In Lecture 6, you learned why validation is helpful for picking settings. We will do a 60-20-20 split.

```{code-cell} ipython3
X_all = df_reg[["MolWt","LogP","TPSA","NumRings"]].values.astype(np.float32)
y_all = df_reg["Melting Point"].values.astype(np.float32).reshape(-1,1)

X_trainval, X_test, y_trainval, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
X_train, X_val,  y_train,  y_val  = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2

X_train.shape, X_val.shape, X_test.shape
```

### 4.3 Standardize features

Most neural nets train more smoothly when inputs are standardized.

```{code-cell} ipython3
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train).astype(np.float32)
Xva = scaler.transform(X_val).astype(np.float32)
Xte = scaler.transform(X_test).astype(np.float32)

ytr = y_train.astype(np.float32)
yva = y_val.astype(np.float32)
yte = y_test.astype(np.float32)
```

### 4.4 Wrap tensors and peek at shapes

```{code-cell} ipython3
Xtr_t = torch.from_numpy(Xtr)
Xva_t = torch.from_numpy(Xva)
Xte_t = torch.from_numpy(Xte)
ytr_t = torch.from_numpy(ytr)
yva_t = torch.from_numpy(yva)
yte_t = torch.from_numpy(yte)

Xtr_t.shape, ytr_t.shape
```

### 4.5 A neat `Dataset` and `DataLoader`

We will create small batches for stable gradients.

```{code-cell} ipython3
class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ArrayDataset(Xtr, ytr)
val_ds   = ArrayDataset(Xva, yva)
test_ds  = ArrayDataset(Xte, yte)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
```

```{admonition} Exercise 4.1
Change `batch_size` to 16 and 256.
Observe the training curve. Which setting is noisier per epoch? Which converges faster in wall time?
```

### 4.6 Define a small regression MLP and inspect parameters

```{code-cell} ipython3
class MPRegressor(nn.Module):
    def __init__(self, d_in=4, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x):
        return self.net(x)

mp_model = MPRegressor(d_in=4, d_hidden=64)
sum(p.numel() for p in mp_model.parameters()), mp_model
```

### 4.7 One forward pass to check shapes

```{code-cell} ipython3
xb, yb = next(iter(train_loader))
with torch.no_grad():
    pred = mp_model(xb)
pred.shape, yb.shape
```

### 4.8 Loss, optimizer, and a clear training loop

We will compute validation loss each epoch to watch for overfitting.

```{code-cell} ipython3
def train_regression(model, train_loader, val_loader, max_epochs=200, lr=1e-3):
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {"train": [], "val": []}
    for epoch in range(1, max_epochs+1):
        # train
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        hist["train"].append(np.mean(batch_losses))

        # validate
        model.eval()
        with torch.no_grad():
            v_losses = []
            for xv, yv in val_loader:
                pv = model(xv)
                v_losses.append(loss_fn(pv, yv).item())
        hist["val"].append(np.mean(v_losses))

        if epoch % 20 == 0:
            print(f"epoch {epoch:3d}  train MSE={hist['train'][-1]:.2f}  val MSE={hist['val'][-1]:.2f}")
    return hist

mp_model = MPRegressor()
hist = train_regression(mp_model, train_loader, val_loader, max_epochs=200, lr=1e-3)
```

### 4.9 Plot learning curves

```{code-cell} ipython3
plt.figure(figsize=(5,3))
plt.plot(hist["train"], label="train")
plt.plot(hist["val"], label="val")
plt.xlabel("epoch"); plt.ylabel("MSE")
plt.title("Learning curves: Melting Point")
plt.legend(); plt.show()
```

### 4.10 Evaluate on test and draw parity plot

```{code-cell} ipython3
mp_model.eval()
with torch.no_grad():
    y_pred_te = mp_model(Xte_t).numpy().ravel()

mse = mean_squared_error(yte, y_pred_te)
mae = mean_absolute_error(yte, y_pred_te)
r2  = r2_score(yte, y_pred_te)

print(f"Test MSE={mse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")

plt.figure(figsize=(4.2,4))
plt.scatter(yte, y_pred_te, alpha=0.6)
lims = [min(yte.min(), y_pred_te.min()), max(yte.max(), y_pred_te.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True MP"); plt.ylabel("Predicted MP"); plt.title("Parity: NN")
plt.show()
```

```{admonition} Exercise 4.2
Add a **Dropout(p=0.1)** layer after the first ReLU. Retrain with the same settings.
Compare validation MSE and test R2. Does Dropout help a little here?
```

```{admonition} Exercise 4.3
Reduce `d_hidden` from 64 to 16. Retrain and compare.
Which setting gives better generalization on this target?
```

---

## 5. Neural nets for chemistry: classification on Toxicity

We reuse the same four descriptors. Now the target is binary: toxic vs non_toxic.

### 5.1 Prepare labels and splits

```{code-cell} ipython3
df_clf = df[["MolWt","LogP","TPSA","NumRings","Toxicity"]].dropna().reset_index(drop=True)
y_txt = df_clf["Toxicity"].str.lower()
y_bin = (y_txt == "toxic").astype(np.int64).values  # 1 toxic, 0 non_toxic
X_all = df_clf[["MolWt","LogP","TPSA","NumRings"]].values.astype(np.float32)

X_trainval, X_test, y_trainval, y_test = train_test_split(X_all, y_bin, test_size=0.2, random_state=7, stratify=y_bin)
X_train, X_val,  y_train,  y_val  = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=7, stratify=y_trainval)

scaler_c = StandardScaler().fit(X_train)
Xtr = scaler_c.transform(X_train).astype(np.float32)
Xva = scaler_c.transform(X_val).astype(np.float32)
Xte = scaler_c.transform(X_test).astype(np.float32)

ytr = y_train.astype(np.int64)
yva = y_val.astype(np.int64)
yte = y_test.astype(np.int64)

train_loader_c = DataLoader(ArrayDataset(Xtr, ytr), batch_size=128, shuffle=True)
val_loader_c   = DataLoader(ArrayDataset(Xva, yva), batch_size=256, shuffle=False)
```

### 5.2 Define a small classifier

Outputs logits of shape `(batch, 2)`. We will use `nn.CrossEntropyLoss`.

```{code-cell} ipython3
class ToxicityMLP(nn.Module):
    def __init__(self, d_in=4, d_hidden=32, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x)  # logits

clf = ToxicityMLP()
sum(p.numel() for p in clf.parameters()), clf
```

### 5.3 Single forward pass and shapes

```{code-cell} ipython3
xb, yb = next(iter(train_loader_c))
logits = clf(xb)   # (B, 2)
proba  = torch.softmax(logits, dim=1)  # probabilities
logits.shape, proba.shape, yb.shape
```

### 5.4 Train with Cross-Entropy

We also compute validation accuracy per epoch.

```{code-cell} ipython3
def train_classifier(model, train_loader, val_loader, max_epochs=100, lr=5e-3):
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_hist, val_hist = [], []

    for epoch in range(1, max_epochs+1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_hist.append(np.mean(losses))

        # simple val accuracy
        model.eval()
        n_correct, n_total, v_losses = 0, 0, []
        with torch.no_grad():
            for xv, yv in val_loader:
                lv = model(xv)
                v_losses.append(loss_fn(lv, yv).item())
                pred = lv.argmax(dim=1)
                n_correct += (pred == yv).sum().item()
                n_total   += yv.shape[0]
        val_hist.append(np.mean(v_losses))
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train CE={train_hist[-1]:.3f}  val CE={val_hist[-1]:.3f}  val acc={n_correct/n_total:.3f}")
    return {"train": train_hist, "val": val_hist}

clf = ToxicityMLP()
hist_c = train_classifier(clf, train_loader_c, val_loader_c, max_epochs=100, lr=5e-3)
```

### 5.5 Learning curves

```{code-cell} ipython3
plt.figure(figsize=(5,3))
plt.plot(hist_c["train"], label="train CE")
plt.plot(hist_c["val"], label="val CE")
plt.xlabel("epoch"); plt.ylabel("Cross-Entropy"); plt.legend(); plt.title("Classification learning curves")
plt.show()
```

### 5.6 Test metrics: Accuracy and AUC

```{code-cell} ipython3
clf.eval()
with torch.no_grad():
    logits_te = clf(torch.from_numpy(Xte))
    proba_te  = torch.softmax(logits_te, dim=1)[:,1].numpy()
    pred_te   = logits_te.argmax(dim=1).numpy()

acc = accuracy_score(yte, pred_te)
auc = roc_auc_score(yte, proba_te)
print(f"Test Accuracy={acc:.3f}  AUC={auc:.3f}")
```

```{admonition} Exercise 5.1
Add a second hidden layer with `ReLU`. Keep the same total parameter count by reducing widths.
Does AUC change on the test split?
```

```{admonition} Exercise 5.2
Set learning rate to `1e-2` and then `1e-4`. Train for the same epochs.
Report validation CE after epoch 100. Which rate is better here?
```

---

## 6. Practical notes

- **Normalization**: Standardize inputs. We reused `StandardScaler` from Lecture 6.
- **Initialization**: PyTorch initializes layers reasonably, but very deep nets may need care.
- **Learning rate**: If loss does not decrease, lower it. If training is slow, try higher but watch for divergence.
- **Batch size**: Small batches add noise that may help generalization.
- **Regularization**: Dropout is easy. Weight decay via `Adam(..., weight_decay=1e-4)` can help.

```{admonition} Link to earlier lectures
Train/val/test split, parity plots, and metrics come from Lectures 5-7. The same diagnostics help with neural nets.
```

---

## 7. Saving and loading models

```{code-cell} ipython3
# Save
torch.save(mp_model.state_dict(), "mp_regressor.pt")

# Load into a fresh instance
mp2 = MPRegressor()
mp2.load_state_dict(torch.load("mp_regressor.pt", map_location="cpu"))
mp2.eval()
```

---

## 8. Quick reference

```{admonition} PyTorch recipe
1) Prepare `X, y` as float tensors (long for class labels).
2) `Dataset` and `DataLoader` with a batch size.
3) Define `nn.Module` with layers and activations.
4) Choose loss and optimizer.
5) Loop: `zero_grad -> forward -> loss -> backward -> step`.
6) Track validation loss and stop if it rises for too long.
```

```{admonition} Common layers
- `nn.Linear(d_in, d_out)`
- `nn.ReLU()`, `nn.Tanh()`, `nn.Sigmoid()`
- `nn.Dropout(p)`
```

---

## 9. Optional: compare to scikit-learn MLP

This gives you a quick baseline. It hides some details but is useful for sanity checks.

```{code-cell} ipython3
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu", random_state=0, max_iter=1000)
mlp.fit(Xtr, ytr.ravel())
pred = mlp.predict(Xte)
print(f"sklearn MLP  R2={r2_score(yte, pred):.3f}  MAE={mean_absolute_error(yte, pred):.2f}")
```

---

## 10. In-class activity (5 questions)

Each task can be done with the code above and small edits.

### Q1. Width vs depth on Melting Point

Train two regression MLPs:

- Model A: `d_hidden=16`, 2 hidden layers
- Model B: `d_hidden=64`, 1 hidden layer

Use the same split as Section 4. Report **validation MSE** at the end of training and **test R2**.

```python
# TO DO: build two MPRegressor variants and compare hist["val"][-1] and R2 on test
```

### Q2. Weight decay on Melting Point

Repeat Section 4 with `weight_decay=1e-4` in Adam. Keep everything else the same.
Report test **MAE** and compare to the no-decay setting.

```python
# TO DO: train_regression but create optimizer with weight decay
```

### Q3. Calibration for Toxicity

For the classifier, collect `proba_te`.
Compute accuracy at thresholds `0.3, 0.5, 0.7`.
Plot the three points on a simple threshold vs accuracy line.

```python
# TO DO: use proba_te and vary threshold to get predictions, then accuracy_score
```

### Q4. Swap activation

Replace ReLU with Tanh in the regression model and repeat training.
Report final validation MSE and test R2.
Is convergence slower?

```python
# TO DO: define MPRegressor with Tanh and train
```

### Q5. Predict new molecules (Melting Point)

Given two descriptor rows:

- `[135.0, 2.0, 9.2, 2]`
- `[301.0, 0.5, 17.7, 2]`

Use the trained regression model and the same `scaler` to predict MP.

```python
# TO DO: transform with scaler, run model.eval() and predict
```

---

## 11. Solutions

### 11.1 Solution Q1

```{code-cell} ipython3
class MPRegressorA(nn.Module):
    def __init__(self, d_in=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x): return self.net(x)

class MPRegressorB(nn.Module):
    def __init__(self, d_in=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)

mA, mB = MPRegressorA(), MPRegressorB()
histA = train_regression(mA, train_loader, val_loader, max_epochs=200, lr=1e-3)
histB = train_regression(mB, train_loader, val_loader, max_epochs=200, lr=1e-3)

def eval_r2(model):
    model.eval()
    with torch.no_grad():
        yp = model(Xte_t).numpy().ravel()
    return r2_score(yte, yp)

print(f"A: val MSE={histA['val'][-1]:.2f}  test R2={eval_r2(mA):.3f}")
print(f"B: val MSE={histB['val'][-1]:.2f}  test R2={eval_r2(mB):.3f}")
```

### 11.2 Solution Q2

```{code-cell} ipython3
def train_regression_wd(model, train_loader, val_loader, max_epochs=200, lr=1e-3, wd=1e-4):
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    hist = {"train": [], "val": []}
    for epoch in range(max_epochs):
        model.train()
        tloss = []
        for xb, yb in train_loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tloss.append(loss.item())
        hist["train"].append(np.mean(tloss))
        model.eval(); vloss=[]
        with torch.no_grad():
            for xv, yv in val_loader:
                vloss.append(loss_fn(model(xv), yv).item())
        hist["val"].append(np.mean(vloss))
    return hist

m_wd = MPRegressor()
hist_wd = train_regression_wd(m_wd, train_loader, val_loader, max_epochs=200, lr=1e-3, wd=1e-4)
m_wd.eval()
with torch.no_grad():
    y_pred_te = m_wd(Xte_t).numpy().ravel()

print(f"With weight decay  MAE={mean_absolute_error(yte, y_pred_te):.2f}")
```

### 11.3 Solution Q3

```{code-cell} ipython3
def acc_at_thresh(proba, y_true, thr):
    y_hat = (proba >= thr).astype(int)
    return accuracy_score(y_true, y_hat)

# Reuse proba_te from Section 5.6 after clf is trained
thr_list = [0.3, 0.5, 0.7]
accs = [acc_at_thresh(proba_te, yte, t) for t in thr_list]
for t,a in zip(thr_list, accs):
    print(f"threshold={t:.2f}  acc={a:.3f}")

plt.figure(figsize=(4.2,3))
plt.plot(thr_list, accs, marker="o")
plt.xlabel("threshold"); plt.ylabel("accuracy"); plt.title("Threshold vs accuracy")
plt.show()
```

### 11.4 Solution Q4

```{code-cell} ipython3
class MPRegressorTanh(nn.Module):
    def __init__(self, d_in=4, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x): return self.net(x)

m_tanh = MPRegressorTanh()
hist_tanh = train_regression(m_tanh, train_loader, val_loader, max_epochs=200, lr=1e-3)

m_tanh.eval()
with torch.no_grad():
    ypt = m_tanh(Xte_t).numpy().ravel()

print(f"Tanh val MSE={hist_tanh['val'][-1]:.2f}  test R2={r2_score(yte, ypt):.3f}")
```

### 11.5 Solution Q5

```{code-cell} ipython3
new_desc = np.array([[135.0, 2.0, 9.2, 2],
                     [301.0, 0.5, 17.7, 2]], dtype=np.float32)
new_scaled = scaler.transform(new_desc).astype(np.float32)
with torch.no_grad():
    preds = mp_model(torch.from_numpy(new_scaled)).numpy().ravel()
pd.DataFrame({
    "MolWt":[135.0,301.0],
    "LogP":[2.0,0.5],
    "TPSA":[9.2,17.7],
    "NumRings":[2,2],
    "Pred_MP":preds
})
```

---

## 12. Glossary

```{glossary}
neuron
  Computes a weighted sum plus bias and passes it through a nonlinearity.

activation
  Function applied to a neuron output. ReLU, Tanh, Sigmoid.

loss
  A number that measures mismatch between predictions and targets.

optimizer
  An algorithm that updates weights to reduce the loss. Adam, SGD.

epoch
  One full pass through the training data.

batch size
  Number of samples per gradient update.

Dropout
  Randomly zeros hidden units during training to reduce overfitting.

weight decay
  L2 penalty on weights controlled by the optimizer.

Cross-Entropy
  Loss used for classification with logits.

MSE
  Mean squared error, common for regression.
```

---

## 13. Wrap-up

You built two neural networks:

- A regressor for Melting Point with 4 simple descriptors.
- A classifier for Toxicity with the same inputs.

Along the way you tracked shapes, watched loss curves, and reused plots and metrics from earlier lectures. The same habits carry into deeper models and molecular representations later in the course.
