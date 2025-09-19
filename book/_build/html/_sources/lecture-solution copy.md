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




---



## 10. Solutions

### Solution Q1

```{code-cell} ipython3

# Q1 solution
set_seed(0)
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_oxidation_raw = pd.read_csv(url)
def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series({
            "MolWt": None,
            "LogP": None,
            "TPSA": None,
            "NumRings": None
        })
    return pd.Series({
        "MolWt": Descriptors.MolWt(mol),                    # molecular weight
        "LogP": Crippen.MolLogP(mol),                       # octanol-water logP
        "TPSA": rdMolDescriptors.CalcTPSA(mol),             # topological polar surface area
        "NumRings": rdMolDescriptors.CalcNumRings(mol)      # number of rings
    })

# Apply the function to the SMILES column
desc_df = df_oxidation_raw["SMILES"].apply(calc_descriptors)

# Concatenate new descriptor columns to original DataFrame
df_clf = pd.concat([df_oxidation_raw, desc_df], axis=1)
df_reg = df_clf.copy() 

# Logistic regression baseline
from sklearn.linear_model import LogisticRegression
X = df_clf[["MolWt","LogP","TPSA","NumRings"]].values
y = df_clf["Toxicity"].str.lower().map({"toxic":1,"non_toxic":0}).values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
baseline = LogisticRegression(max_iter=1000).fit(Xtr,ytr)
print("Baseline acc:", accuracy_score(yte, baseline.predict(Xte)))

# MLP from scratch
class MLP_Q1(nn.Module):
    def __init__(self,in_dim,hidden=32,n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,hidden)
        self.fc2 = nn.Linear(hidden,n_classes)
    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

model = MLP_Q1(4)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    xb = torch.tensor(Xtr, dtype=torch.float32)
    yb = torch.tensor(ytr, dtype=torch.long)
    opt.zero_grad()
    out = model(xb)
    loss = loss_fn(out,yb)
    loss.backward()
    opt.step()

model.eval()
ypred = model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
print("MLP acc:", accuracy_score(yte, ypred))
#For this question and the following questions, it is fine if the model performance is worse than the baseline. 
```

### Solution Q2

```{code-cell} ipython3
X = df_reg[["MolWt","LogP","TPSA","NumRings"]].values.astype(np.float32)
y = df_reg["Melting Point"].values.astype(np.float32).reshape(-1,1)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=0)

# Baseline
rf = RandomForestRegressor(n_estimators=200,random_state=0).fit(Xtr,ytr.ravel())
print("Baseline RF R2:", r2_score(yte,rf.predict(Xte)))

# PyTorch MLP
class MLPreg_Q2(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): return self.net(x)

model = MLPreg_Q2(4)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    xb = torch.tensor(Xtr, dtype=torch.float32)
    yb = torch.tensor(ytr, dtype=torch.float32)
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward(); opt.step()

yhat = model(torch.tensor(Xte, dtype=torch.float32)).detach().numpy()
print("MLP R2:", r2_score(yte,yhat))


```

### Solution Q3

```{code-cell} ipython3
# Build toxicity graphs
label_map = {"toxic":1, "non_toxic":0}
df_tox = df[df["Toxicity"].str.lower().isin(label_map.keys())].copy()
y_bin = df_tox["Toxicity"].str.lower().map(label_map).astype(int).values

graphs_tox = []
for smi, yv in zip(df_tox["SMILES"], y_bin):
    mol, x_np, ei_np, ea_np = smiles_to_graph(smi)
    g = {
        "x": torch.tensor(x_np, dtype=torch.float32),
        "edge_index": torch.tensor(ei_np, dtype=torch.long),
        "edge_attr": torch.tensor(ea_np, dtype=torch.long),
        "y": torch.tensor(yv, dtype=torch.long)
    }
    graphs_tox.append(g)

train_idx, test_idx = train_test_split(
    np.arange(len(graphs_tox)),
    test_size=0.2, random_state=42, stratify=y_bin
)
train_graphs = [graphs_tox[i] for i in train_idx]
test_graphs  = [graphs_tox[i] for i in test_idx]

# Baseline on descriptors: simple MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X_desc = df_tox[["MolWt","LogP","TPSA","NumRings"]].values
Xtr_d, Xte_d = X_desc[train_idx], X_desc[test_idx]
ytr_d, yte_d = y_bin[train_idx], y_bin[test_idx]

mlp_base = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(16,),
                          activation="relu",
                          learning_rate_init=0.01,
                          max_iter=2000,
                          random_state=0))
]).fit(Xtr_d, ytr_d)
base_acc = accuracy_score(yte_d, mlp_base.predict(Xte_d))

# GNN variant: 3 layers, sum pooling, dropout
class MPNNClassifierV3(nn.Module):
    def __init__(self, in_dim=4, hidden=48, n_layers=3, n_classes=2, dropout=0.2, pool="sum"):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden]*(n_layers-1) + [hidden]
        for a, b in zip(dims[:-1], dims[1:]):
            self.layers.append(MPNNLayer(a, b))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)
        self.pool = pool

    def forward(self, g):
        x, edge_index, edge_attr = g["x"], g["edge_index"], g["edge_attr"]
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            h = self.dropout(h)
        if self.pool == "sum":
            h_graph = h.sum(dim=0)
        else:
            h_graph = h.mean(dim=0)
        return self.fc(h_graph)

def train_epoch(model, graphs, opt, loss_fn):
    model.train(); total=0
    for g in graphs:
        opt.zero_grad()
        out = model(g).unsqueeze(0)
        loss = loss_fn(out, g["y"].unsqueeze(0))
        loss.backward(); opt.step()
        total += float(loss.item())
    return total/len(graphs)

@torch.no_grad()
def eval_acc(model, graphs):
    model.eval(); correct=0
    for g in graphs:
        pred = model(g).argmax().item()
        correct += int(pred == g["y"].item())
    return correct/len(graphs)

gnn = MPNNClassifierV3(in_dim=4, hidden=48, n_layers=3, dropout=0.2, pool="sum")
opt = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5): #you should replace with 25. Our class website can't run this long training so I put 5 here.
    _ = train_epoch(gnn, train_graphs, opt, loss_fn)

gnn_acc = eval_acc(gnn, test_graphs)

print(f"Descriptor MLP baseline acc: {base_acc:.3f}")
print(f"GNN (3 layers, sum pool, dropout) acc: {gnn_acc:.3f}")

```

### Solution Q4

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Prepare once
y_all = np.array([g["y"].item() for g in graphs_tox])

def fit_eval_fold(train_ids, test_ids):
    tr = [graphs_tox[i] for i in train_ids]
    te = [graphs_tox[i] for i in test_ids]

    # Different architecture from Q3 and class: 2 layers, hidden 96, max pooling
    class MPNNClassifierCV(nn.Module):
        def __init__(self, in_dim=4, hidden=96, n_classes=2):
            super().__init__()
            self.l1 = MPNNLayer(in_dim, hidden)
            self.l2 = MPNNLayer(hidden, hidden)
            self.fc = nn.Linear(hidden, n_classes)
        def forward(self, g):
            x, ei, ea = g["x"], g["edge_index"], g["edge_attr"]
            h = self.l1(x, ei, ea)
            h = self.l2(h, ei, ea)
            h_graph, _ = torch.max(h, dim=0)  # max pooling
            return self.fc(h_graph)

    model = MPNNClassifierCV()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # quick train
    for epoch in range(5): #again, you should replace with 20
        model.train()
        for g in tr:
            opt.zero_grad()
            out = model(g).unsqueeze(0)
            loss = loss_fn(out, g["y"].unsqueeze(0))
            loss.backward(); opt.step()

    # eval acc and AUC
    model.eval()
    correct=0; probs=[]; ys=[]
    with torch.no_grad():
        for g in te:
            logits = model(g)
            p = torch.softmax(logits, dim=0)[1].item()
            pred = int(logits.argmax().item())
            probs.append(p); ys.append(int(g["y"].item()))
            correct += int(pred == ys[-1])
    acc = correct/len(te)
    auc = roc_auc_score(ys, probs)
    return acc, auc

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs, aucs = [], []
for tr_ids, te_ids in skf.split(np.arange(len(graphs_tox)), y_all):
    acc, auc = fit_eval_fold(tr_ids, te_ids)
    accs.append(acc); aucs.append(auc)

print(f"5-fold mean acc: {np.mean(accs):.3f}  ± {np.std(accs):.3f}")
print(f"5-fold mean AUC: {np.mean(aucs):.3f}  ± {np.std(aucs):.3f}")

```

### Solution Q5

```python
# Reactivity classification with an MLP from scratch on descriptors
# Reactivity is −1 or 1. Map to {0,1}.
df_rxn = df[["SMILES","Reactivity","MolWt","LogP","TPSA","NumRings"]].dropna().copy()
df_rxn = df_rxn[df_rxn["Reactivity"].isin([-1, 1])]
y_rxn = (df_rxn["Reactivity"].map({-1:0, 1:1})).astype(int).values
X_rxn = df_rxn[["MolWt","LogP","TPSA","NumRings"]].values.astype(np.float32)

Xtr, Xte, ytr, yte = train_test_split(X_rxn, y_rxn, test_size=0.2, random_state=0, stratify=y_rxn)

# Baselines
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

logit = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=0))]).fit(Xtr, ytr)
rf    = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(Xtr, ytr)

base_acc_log = accuracy_score(yte, logit.predict(Xte))
base_auc_log = roc_auc_score(yte, logit.predict_proba(Xte)[:,1])
base_acc_rf  = accuracy_score(yte, rf.predict(Xte))
base_auc_rf  = roc_auc_score(yte, rf.predict_proba(Xte)[:,1])

# PyTorch MLP from scratch
class MLPReact(nn.Module):
    def __init__(self, in_dim=4, hidden=(32,16), n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_classes)
        )
    def forward(self, x): return self.net(x)

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr).astype(np.float32)
Xte_s = scaler.transform(Xte).astype(np.float32)

torch.manual_seed(0)
mlp = MLPReact(in_dim=4, hidden=(32,16))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

xb = torch.tensor(Xtr_s); yb = torch.tensor(ytr, dtype=torch.long)
for epoch in range(5): #replace with 60
    mlp.train()
    opt.zero_grad()
    loss = loss_fn(mlp(xb), yb)
    loss.backward(); opt.step()

mlp.eval()
with torch.no_grad():
    logits = mlp(torch.tensor(Xte_s))
    pred   = logits.argmax(1).numpy()
    proba  = torch.softmax(logits, dim=1)[:,1].numpy()

mlp_acc = accuracy_score(yte, pred)
mlp_auc = roc_auc_score(yte, proba)

print(f"LogReg baseline  acc={base_acc_log:.3f}  AUC={base_auc_log:.3f}")
print(f"RF baseline      acc={base_acc_rf:.3f}   AUC={base_auc_rf:.3f}")
print(f"MLP (scratch)    acc={mlp_acc:.3f}   AUC={mlp_auc:.3f}")


```

