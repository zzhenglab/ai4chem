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
