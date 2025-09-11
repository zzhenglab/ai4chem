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

### 8.3 OOB sanity check

Compare OOB score to test accuracy across seeds.

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

### 8.5 Small tree visualization and split rules

Fit a very small classifier tree and print its first two rules in plain language.

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

Read the rules as binary questions. Samples that satisfy a rule go left. Others go right.
