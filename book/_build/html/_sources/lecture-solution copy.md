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






## 10. In-class activity


### 10.1 Linear Regression with two features

Use only `MolWt` and `TPSA` to predict **Melting Point** with Linear Regression. Use a 90/10 split and report **MSE**, **MAE**, and **R²**.

```python

# Q1 starter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = df_reg_mp[["MolWt", "TPSA"]]
y = df_reg_mp["Melting Point"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=0
)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R2:  {r2_score(y_test, y_pred):.3f}")
```

### 10.2 Ridge across splits

Train a Ridge model (`alpha=1.0`) for **Melting Point** using `MolWt, LogP, TPSA, NumRings`. Compare test **R²** for train sizes 60, 70, 80, 90 percent with `random_state=42`. Plot **R²** vs train percent.

```python
X = df_reg_mp[["MolWt", "LogP", "TPSA", "NumRings"]].values
y = df_reg_mp["Melting Point"].values

splits = [0.4, 0.3, 0.2, 0.1]  # corresponds to 60/40, 70/30, 80/20, 90/10
r2_scores = []

for t in splits:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t, random_state=42
    )
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))

# Plot results
plt.figure(figsize=(6,4))
plt.plot([60,70,80,90], r2_scores, "o-", lw=2)
plt.xlabel("Train %")
plt.ylabel("R² (test)")
plt.title("Effect of train/test split on Ridge Regression accuracy")
plt.show()
```

### 10.3 pKa regression two ways

Build Ridge regression for **pKa** and for **exp(pKa)** using the same four descriptors. Report **R²** and **MSE** for each.

```python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Keep rows with a valid pKa
df_pka = df[["MolWt", "LogP", "TPSA", "NumRings", "pKa"]].dropna()

X = df_pka[["MolWt", "LogP", "TPSA", "NumRings"]].values
y = df_pka["pKa"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Test R2:  {r2_score(y_test, y_pred):.3f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.3f}")

# Parity plot
plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True pKa")
plt.ylabel("Predicted pKa")
plt.title("Parity plot for pKa regression (Ridge)")
plt.show()
```

### 10.4 pKa to classification

Turn **pKa** into a binary label and train Logistic Regression with the same descriptors. Report Accuracy, Precision, Recall, F1, and AUC, and draw the ROC. You may pick either rule.

- Option A: acidic if pKa ≤ 7  
- Option B: median split on pKa

```python

# Clean pKa subset
df_pka = df[["MolWt", "LogP", "TPSA", "NumRings", "pKa"]].dropna()
X = df_pka[["MolWt", "LogP", "TPSA", "NumRings"]].values
pka_vals = df_pka["pKa"].values

# ---- Helper to run classification and plot ----
def run_classification(y_cls, rule_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    print(f"--- {rule_name} ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"AUC:       {auc:.3f}")
    print()

    # ROC plot
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for pKa classification ({rule_name})")
    plt.legend()
    plt.show()

# ---- Rule A: acidic if pKa ≤ 7 ----
y_cls_A = (pka_vals <= 7.0).astype(int)
run_classification(y_cls_A, "Rule A (pKa ≤ 7 = acidic)")

# ---- Rule B: median split ----
median_val = np.median(pka_vals)
y_cls_B = (pka_vals <= median_val).astype(int)
run_classification(y_cls_B, f"Rule B (≤ median pKa = acidic, median={median_val:.2f})")
```

### 10.5 Threshold tuning on toxicity

Using the toxicity classifier from Section 5, scan thresholds `0.2` to `0.8` in steps of `0.05`. Find the smallest threshold with **recall ≥ 0.80** and report the corresponding **precision** and **F1**. Plot the metric curves vs threshold.

```python
# Starter
ths = np.arange(0.20, 0.81, 0.05)
rec_list, prec_list, f1_list = [], [], []
best_t = None

for t in ths:
    pred_t = (y_proba >= t).astype(int)
    r = recall_score(y_test, pred_t)
    p = precision_score(y_test, pred_t, zero_division=0)
    f = f1_score(y_test, pred_t, zero_division=0)
    rec_list.append(r); prec_list.append(p); f1_list.append(f)
    if best_t is None and r >= 0.80:
        best_t = t

print("First threshold with recall >= 0.80:", best_t)

plt.figure(figsize=(7,5))
plt.plot(ths, rec_list, marker="o", label="Recall")
plt.plot(ths, prec_list, marker="o", label="Precision")
plt.plot(ths, f1_list, marker="o", label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold tuning on toxicity")
plt.legend()
plt.grid(True)
plt.show()
```










## 11. Solutions

### 11.1 Solution Q1

```{code-cell} ipython3
# Log target
df_reg_mp = df_reg_mp.copy()
y_log = np.log10(df_reg_mp["Solubility_mol_per_L"] + 1e-6)

# Features and split
X = df_reg_mp[["MolWt","LogP","TPSA","NumRings"]].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.2, random_state=15)

# Plots
plt.figure(figsize=(5,3))
plt.hist(y_log, bins=30, alpha=0.85)
plt.xlabel("log10(Solubility + 1e-6)"); plt.ylabel("Count"); plt.title("Log-solubility")
plt.show()

pd.plotting.scatter_matrix(df_reg_mp[["MolWt","LogP","TPSA","NumRings"]], figsize=(5.5,5.5))
plt.suptitle("Descriptor scatter matrix", y=1.02); plt.show()
```

### 11.2 Solution Q2

```{code-cell} ipython3
cv = KFold(n_splits=5, shuffle=True, random_state=1)
alphas = np.logspace(-2, 3, 12)
means = [cross_val_score(Ridge(alpha=a), X_tr, y_tr, cv=cv, scoring="r2").mean() for a in alphas]
best_a = float(alphas[int(np.argmax(means))])
ridge_best = Ridge(alpha=best_a).fit(X_tr, y_tr)
print(f"best alpha={best_a:.4f}  CV mean R2={max(means):.3f}")
```

### 11.3 Solution Q3

```{code-cell} ipython3
y_hat = ridge_best.predict(X_te)
print(f"Test MSE={mean_squared_error(y_te,y_hat):.4f}  "
      f"MAE={mean_absolute_error(y_te,y_hat):.4f}  "
      f"R2={r2_score(y_te,y_hat):.3f}")

plt.figure(figsize=(4.2,4))
plt.scatter(y_te, y_hat, alpha=0.7)
lims = [min(y_te.min(), y_hat.min()), max(y_te.max(), y_hat.max())]
plt.plot(lims, lims, "k--", lw=1)
plt.xlabel("True log-solubility"); plt.ylabel("Predicted"); plt.title("Parity — Ridge"); plt.show()

resid = y_te - y_hat
plt.figure(figsize=(4.2,4))
plt.scatter(y_hat, resid, alpha=0.7); plt.axhline(0, color="k", ls=":")
plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title("Residuals — Ridge"); plt.show()
```

### 11.4 Solution Q4

```{code-cell} ipython3
X_new = np.array([
    [135.0,  2.0,  9.2, 2],   # Molecule A
    [301.0,  0.5, 17.7, 2]    # Molecule B
])  # descriptors: [MolWt, LogP, TPSA, NumRings]

y_new = ridge_best.predict(X_new)
print(pd.DataFrame({
    "MolWt": X_new[:,0], "LogP": X_new[:,1], "TPSA": X_new[:,2], "NumRings": X_new[:,3],
    "Predicted log10(solubility)": y_new
}))
```

### 11.5 Solution Q5

```{code-cell} ipython3
feat = ["MolWt","LogP","TPSA","NumRings"]

coef_ser = pd.Series(ridge_best.coef_, index=feat).sort_values(key=np.abs, ascending=False)
print("Ridge coefficients:\n", coef_ser)
coef_ser.plot(kind="barh"); plt.gca().invert_yaxis()
plt.xlabel("Coefficient"); plt.title("Ridge coefficients"); plt.show()

perm = permutation_importance(ridge_best, X_te, y_te, scoring="r2", n_repeats=30, random_state=1)
perm_ser = pd.Series(perm.importances_mean, index=feat).sort_values()
perm_ser.plot(kind="barh"); plt.xlabel("Mean decrease in R²"); plt.title("Permutation importance on test"); plt.show()
```

---
