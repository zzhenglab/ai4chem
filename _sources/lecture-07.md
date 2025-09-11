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
:depth: 1
```

## Learning goals

- Explain the intuition of **decision trees** for regression and classification.
- Read **Gini** and **entropy** for splits and **MSE** for regression splits.
- Grow a tree step by step and inspect internal structures: nodes, depth, leaf counts.
- Control tree growth with `max_depth`, `min_samples_leaf`, `min_samples_split`.
- Visualize a fitted tree and feature importance.
- Train a **Random Forest** and compare to a single tree.
- Use **out of bag (OOB)** score for quick validation.
- Put it all together in a short end-to-end workflow.

  [![Colab](https://img.shields.io/badge/Open-Colab-orange)](https://colab.research.google.com/drive/1Mkzv1qh9t9tL9w6m4C2U1Qw3W2c2W8bR?usp=sharing)

---

## 0. Setup

```{code-cell} ipython3
:tags: [hide-input]
# 0. Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
except Exception:
    Chem = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")
```

## 1. Decision tree

### 1.1 Load data and build descriptors

We will reuse the same dataset to keep the context consistent. If RDKit is available, we compute four descriptors; otherwise we fallback to numeric columns that are already present.

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/C_H_oxidation_dataset.csv"
df_raw = pd.read_csv(url)
df_raw.head()
```

```{code-cell} ipython3
def calc_descriptors(smiles):
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

desc = df_raw["SMILES"].apply(calc_descriptors)
df = pd.concat([df_raw, desc], axis=1)
df.head(3)
```

```{admonition} Features
We will use `MolWt`, `LogP`, `TPSA`, `NumRings` as base features. They worked well in earlier lectures and are fast to compute.
```

---
### 1.2 What is a decision tree


A tree splits the feature space into rectangles by asking simple questions like `LogP <= 1.3`. Each split tries to make the target inside each branch more pure.

- For **classification**, purity is measured by **Gini** or **entropy**.
- For **regression**, it is common to use **MSE** reduction.

```{admonition} Purity
- Gini for a node with class probs \(p_k\): \(1 - \sum_k p_k^2\)
- Entropy: \(-\sum_k p_k \log_2 p_k\)
- Regression impurity at a node: mean squared error to the node mean
```




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
 > For regression, impurity at a node is measured by mean squared error to the node mean.
 > For classification, it is measured by **Gini** for a node with class probs $p_k$: $(1 - \sum_k p_k^2$ or **Entropy**: $-\sum_k p_k \log_2 p_k\$

```
---

### 1.3 Tiny classification example: one split

We start with toxicity as a binary label to see a single split and the data shape at each step.

We will split the dataset into train and test parts. Stratification (`stratify = y`) keeps the class ratio similar in both parts, which is important when classes are imbalanced.

```{code-cell} ipython3
df_clf = df[["MolWt", "LogP", "TPSA", "NumRings", "Toxicity"]].dropna()
label_map = {"toxic": 1, "non_toxic": 0}
y = df_clf["Toxicity"].str.lower().map(label_map).astype(int)
X = df_clf[["MolWt", "LogP", "TPSA", "NumRings"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape
```
As you remember, we have total 575 data points and 80% goes to training samples.

You can glance at the first few rows to get a feel for the feature scales. Trees do not require scaling, but it is still useful context.



```{code-cell} ipython3
X_train.head()
```


We will grow a stump: a tree with `max_depth=1`. This forces one split. It helps you see how a split is chosen and how samples are divided.

```{admonition} Stump

The tree considers possible thresholds on each feature.
For each candidate threshold it computes an impurity score on the left and right child nodes. We use Gini impurity, which gets smaller when a node contains mostly one class.
It picks the feature and threshold that bring the largest impurity decrease.
```

```{code-cell} ipython3
stump = DecisionTreeClassifier(max_depth=1, criterion="gini", random_state=0)
stump.fit(X_train, y_train)

print("Feature used at root:", stump.tree_.feature[0])
print("Threshold at root:", stump.tree_.threshold[0])
print("n_nodes:", stump.tree_.node_count)
print("children_left:", stump.tree_.children_left[:3])
print("children_right:", stump.tree_.children_right[:3])
```

The model stores feature indices internally. Mapping that index back to the column name makes the split human readable.

```{code-cell} ipython3
# Map index to name for readability
feat_names = X_train.columns.tolist()
root_feat = feat_names[stump.tree_.feature[0]]
thr = stump.tree_.threshold[0]
print(f"Root rule: {root_feat} <= {thr:.3f}?")
```

Read the rule as: if the condition is true, the sample goes to the left leaf, otherwise to the right leaf.


```{admonition} How good is one split
A single split is intentionally simple. It may already capture a strong signal if one feature provides a clean separation. We will check test performance with standard metrics.
> Precision of class k: among items predicted as k, how many were truly k
> Recall of class k: among true items of k, how many did we catch
> F1 is the harmonic mean of precision and recall
> Support is the number of true samples for each class
It picks the feature and threshold that bring the largest impurity decrease.
```

```{code-cell} ipython3
# Evaluate stump
from sklearn.metrics import classification_report
y_pred = stump.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred)
cm
```


Use the confusion matrix to see the error pattern.

In this case:
`FP` are non toxic predicted as toxic
`FN` are toxic predicted as non toxic

Which side is larger tells you which type of mistake the one split is making more often.

Now, let's visualize the rule.
The tree plot below shows the root node with its split, the Gini impurity at each node, the sample counts, and the class distribution. Filled colors hint at the majority class in each leaf.

```{code-cell} ipython3
# Visualize stump
plt.figure(figsize=(5,5))
plot_tree(stump, feature_names=feat_names, class_names=["non_toxic","toxic"], filled=True, impurity=True)
plt.show()
```

We can also visualize `max_depth=2` to see the difference:
```{code-cell} ipython3
stump2 = DecisionTreeClassifier(max_depth=2, criterion="gini", random_state=0)
stump2.fit(X_train, y_train)

plt.figure(figsize=(5,5))
plot_tree(stump2, feature_names=feat_names, class_names=["non_toxic","toxic"], filled=True, impurity=True)
plt.show()
```

---

### 1.4 Grow deeper and control overfitting

Trees can fit noise if we let them grow without limits. We control growth using a few simple knobs.

- `max_depth`: limit the number of levels
- `min_samples_split`: a node needs at least this many samples to split
- `min_samples_leaf`: each leaf must have at least this many samples

```{code-cell} ipython3
def fit_eval_tree(max_depth=None, min_leaf=1):
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, acc

depths = [1, 2, 3, 4, 5, None]  # None means grow until pure or exhausted
scores = []
for d in depths:
    _, acc = fit_eval_tree(max_depth=d, min_leaf=3)
    scores.append(acc)

pd.DataFrame({"max_depth": [str(d) for d in depths], "Accuracy": np.round(scores,3)})
```
Here, we hold `min_samples_leaf = 3` and vary `max_depth`. This shows the classic underfit to overfit trend.
```{code-cell} ipython3
plt.plot([str(d) for d in depths], scores, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Accuracy (test)")
plt.title("Tree depth vs test accuracy")
plt.grid(True)
plt.show()
```

```{admonition} Takeaway
Shallow trees underfit. Very deep trees often overfit. Start small, add depth only if validation improves.
```

Now, let's try sweep leaf size at several depths.

We will try a small grid. This lets you see **both** knobs together.

```{code-cell} ipython3
def fit_acc_leaves(max_depth=None, min_leaf=1):
    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0
    )
    clf.fit(X_train, y_train)
    acc_test = accuracy_score(y_test, clf.predict(X_test))
    n_leaves = clf.get_n_leaves()  # simple and reliable
    return acc_test, n_leaves

leaf_sizes = [1, 3, 5, 10, 20]
depths = [1, 2, 3, 4, 5, None]

rows = []
for leaf in leaf_sizes:
    for d in depths:
        acc, leaves = fit_acc_leaves(max_depth=d, min_leaf=leaf)
        rows.append({
            "min_samples_leaf": leaf,
            "max_depth": str(d),
            "acc_test": acc,
            "n_leaves": leaves
        })

df_grid = pd.DataFrame(rows).sort_values(
    ["min_samples_leaf","max_depth"]
).reset_index(drop=True)
df_grid
```
Now, we plot test accuracy vs depth for each leaf size.

Higher min_samples_leaf tends to smooth the curves and reduce the train minus test gap.

```{admonition} Hyperparamter
Higher `min_samples_leaf` tends to smooth the curves and reduce the train minus test gap.
```


```{code-cell} ipython3
plt.figure()
for leaf in leaf_sizes:
    sub = df_grid[df_grid["min_samples_leaf"]==leaf]
    plt.plot(sub["max_depth"], sub["acc_test"], marker="o", label=f"leaf={leaf}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy (test)")
plt.title("Depth vs test accuracy at different min_samples_leaf")
plt.grid(True)
plt.legend()
plt.show()

```
Finally, let's look at what happen if we fix depth and vary leaf size:

Pick a moderate depth (`4`), then look at how leaf size alone affects accuracy and model size.

```{code-cell} ipython3

fixed_depth = 4
rows = []
for leaf in leaf_sizes:
    clf = DecisionTreeClassifier(
        max_depth=fixed_depth, min_samples_leaf=leaf, random_state=0
    ).fit(X_train, y_train)

    rows.append({
        "min_samples_leaf": leaf,
        "acc_train": accuracy_score(y_train, clf.predict(X_train)),
        "acc_test":  accuracy_score(y_test,  clf.predict(X_test)),
        "n_nodes":   clf.tree_.node_count,
        "n_leaves":  clf.get_n_leaves(),
    })

df_leaf = pd.DataFrame(rows).sort_values("min_samples_leaf").reset_index(drop=True)
df_leaf[["min_samples_leaf","acc_train","acc_test","n_nodes","n_leaves"]]

```

```{code-cell} ipython3
plt.figure()
plt.plot(df_leaf["min_samples_leaf"], df_leaf["acc_train"], marker="o", label="Train")
plt.plot(df_leaf["min_samples_leaf"], df_leaf["acc_test"],  marker="o", label="Test")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title(f"Effect of min_samples_leaf at max_depth={fixed_depth}")
plt.grid(True)
plt.legend()
plt.show()

```

```{code-cell} ipython3
plt.figure()
plt.plot(df_leaf["min_samples_leaf"], df_leaf["n_leaves"], marker="o")
plt.xlabel("min_samples_leaf")
plt.ylabel("Number of leaves")
plt.title(f"Model size vs min_samples_leaf at max_depth={fixed_depth}")
plt.grid(True)
plt.show()
```

```{admonition} Underfitting vs. Overfitting
Higher `min_samples_leaf` tends to smooth the curves and reduce the train minus test gap.


- **Underfitting**  
  Happens when the model is too simple to capture meaningful patterns.  
  Signs:
  - Both training and test accuracy are low.
  - The decision boundary looks crude.
  - Increasing model capacity (like depth) improves results.

- **Overfitting**  
  Happens when the model is too complex and memorizes noise in the training set.  
  Signs:
  - Training accuracy is very high (often near 100%).
  - Test accuracy lags behind.
  - Model has many small leaves with very few samples.
  - Predictions fluctuate wildly for minor changes in input.

The goal is **good generalization**: high performance on unseen data, not just on the training set.

```

---

### 1.5 Model-based importance: Gini importance  

As we learned in lecture 6, ML models not only make predictions but also provide insight into which features were most useful. 

There are two main ways we can measure this: **Gini importance** (model-based) and **permutation importance** (data-based). Both give us different perspectives on feature relevance.


When fitting a tree, each split reduces impurity (measured by Gini or entropy). A feature’s importance is computed as:

- The total decrease in impurity that results from splits on that feature
- Normalized so that all features sum to `1`

This is fast and built-in, but it reflects **how the model used the features** during training. It may overstate the importance of high-cardinality or correlated features.

```{code-cell} ipython3
tree_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=0).fit(X_train, y_train)

imp = pd.Series(tree_clf.feature_importances_, index=feat_names).sort_values(ascending=False)
imp
```

```{code-cell} ipython3
imp.plot(kind="barh")
plt.title("Gini importance (tree)")
plt.gca().invert_yaxis()
plt.show()

```
This bar chart above ranks features by how much they reduced impurity in the training process. The top feature is the one the tree found most useful for splitting. However, keep in mind this is internal to the model and may not reflect true predictive power on unseen data.



---

### 1.6 Data-based importance: Permutation importance
Permutation importance takes a different approach. Instead of looking inside the model, it asks: What happens if I scramble a feature’s values on the test set? If accuracy drops, that feature was important. If accuracy stays the same, the model did not really depend on it.

Steps:

> 1. Shuffle one feature column at a time in the test set.
> 2. Measure the drop in accuracy.
> 3. Repeat many times and average to reduce randomness.


```{code-cell} ipython3

perm = permutation_importance(
    tree_clf, X_test, y_test, scoring="accuracy", n_repeats=20, random_state=0
)
perm_ser = pd.Series(perm.importances_mean, index=feat_names).sort_values()
perm_ser.plot(kind="barh")
plt.title("Permutation importance (test)")
plt.show()

```

Here, the bars show how much accuracy falls when each feature is disrupted. This provides a more honest reflection of **predictive value on unseen data**. Features that looked strong in Gini importance may shrink here if they were just splitting on quirks of the training set.


```{admonition} Comparing the two methods
Gini importance (tree-based):
> Easy to compute.
> Biased toward features with many possible split points.
> Reflects training behavior, not necessarily generalization.



Permutation importance (test-based):
> More computationally expensive (requires multiple shuffles).
> Directly tied to model performance on new data.
> Can reveal when the model “thought” a feature was important but it doesn’t hold up in practice.
```


---

## 2. Regression trees on Melting Point

So far we used trees for **classification**. Now we switch to a **regression target**: the **melting point** of molecules. The mechanics are similar, but instead of predicting a discrete class, the tree predicts a continuous value.

```{code-cell} ipython3
df_reg = df[["MolWt", "LogP", "TPSA", "NumRings", "Melting Point"]].dropna()
Xr = df_reg[["MolWt", "LogP", "TPSA", "NumRings"]]
yr = df_reg["Melting Point"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42
)

Xr_train.head(2), yr_train.head(3)
```

Just like before, we can grow a stump (·max_depth=1·) to see a single split.
Instead of class impurity, the split criterion is reduction in variance of the target.


```{code-cell} ipython3
reg_stump = DecisionTreeRegressor(max_depth=1, random_state=0)
reg_stump.fit(Xr_train, yr_train)
print("Root feature:", Xr_train.columns[reg_stump.tree_.feature[0]])
print("Root threshold:", reg_stump.tree_.threshold[0])
```

```{admonition} What can we learn from Root Feature output?
This tells us the first cut is on `MolW`. Samples with weight below ~246 g/mol are grouped separately from heavier ones.

```

Let’s vary max_depth and track R² on the test set.
Remember: 
> R² = 1 means perfect prediction,
> R² = 0 means the model is no better than the mean.


Below, we pick `depth = 3`, `leaf size = 5`. This is a good trade-off.

```{code-cell} ipython3
# Evaluate shallow vs deeper
depths = [1, 2, 3, 4, 6, 8, None]
r2s = []
for d in depths:
    reg = DecisionTreeRegressor(max_depth=d, min_samples_leaf=5, random_state=0).fit(Xr_train, yr_train)
    yhat = reg.predict(Xr_test)
    r2s.append(r2_score(yr_test, yhat))

pd.DataFrame({"max_depth": [str(d) for d in depths], "R2_test": np.round(r2s,3)})
```

```{code-cell} ipython3
plt.plot([str(d) for d in depths], r2s, marker="o")
plt.xlabel("max_depth")
plt.ylabel("R2 on test")
plt.title("Regression tree depth vs R2")
plt.grid(True)
plt.show()
```
This mirrors the classification case: shallow trees underfit, very deep trees overfit.




Points close to the dashed line = good predictions. Scatter away from the line = errors. Here, predictions track well but show some spread at high melting points.
```{code-cell} ipython3
# Diagnostics for a chosen depth
reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0).fit(Xr_train, yr_train)
yhat = reg.predict(Xr_test)

print(f"MSE={mean_squared_error(yr_test, yhat):.3f}")
print(f"MAE={mean_absolute_error(yr_test, yhat):.3f}")
print(f"R2={r2_score(yr_test, yhat):.3f}")

# Parity
plt.scatter(yr_test, yhat, alpha=0.6)
lims = [min(yr_test.min(), yhat.min()), max(yr_test.max(), yhat.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Parity plot: tree regressor")
plt.show()

# Residuals
res = yr_test - yhat
plt.scatter(yhat, res, alpha=0.6)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual plot: tree regressor")
plt.show()
```
Similar to the example we see on the linear regression, residuals (true – predicted) should scatter around zero. If you see patterns (e.g., always underpredicting high values), the model may be biased. Here, residuals are fairly centered but not perfectly homoscedastic.


```{code-cell} ipython3
# Visualize a small regression tree
plt.figure(figsize=(15,10))
plot_tree(reg, feature_names=Xr_train.columns, filled=True, impurity=True, rounded=True, proportion=False)
plt.show()
```

```{admonition} Regression tree vs classifier tree
A regression tree is structured the same as a classifier tree, but each leaf stores an *average target value* instead of a class distribution.

```
---

## 3. Random Forest: bagging many trees



Decision trees are intuitive but unstable: a small change in data can produce a very different tree. To make trees more reliable and accurate, we use **ensembles** — groups of models working together. The most widely used ensemble of trees is the **Random Forest**.

Eseentially, a random forest grows **many decision trees**, each trained on a slightly different view of the data, and then combines their predictions:

- **Bootstrap sampling (bagging)**:  
  Each tree sees a different random subset of the training rows, sampled *with replacement*. About one-third of rows are left out for that tree (these are the **out-of-bag samples**).
  
- **Feature subsampling**:  
  At each split, the tree does not see all features — only a random subset, controlled by `max_features`. This prevents trees from always picking the same strong predictor and encourages diversity.

Each tree may be a weak learner, but when you **average many diverse trees**, the variance cancels out. This makes forests much more stable and accurate than single trees, especially on noisy data.

---

Here are some helpful aspect of forests:

- **Single deep tree** → low bias, high variance (overfit easily).  
- **Forest of many deep trees** → still low bias, but variance shrinks when you average.  
- **Built-in validation**: out-of-bag samples allow you to estimate performance without needing a separate validation set.  

In practice, random forests are a strong default: robust, interpretable at the feature level, and requiring little parameter tuning.

### 3.1 Classification forest on toxicity
We now build a forest to classify molecules as toxic vs non-toxic.  
We set:
- `n_estimators=300`: number of trees.  
- `max_features="sqrt"`: common heuristic for classification.  
- `min_samples_leaf=3`: prevent leaves with only 1 or 2 samples.  

```{code-cell} ipython3
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=3,
    max_features="sqrt",
    oob_score=True,          # enable OOB estimation, if you dont specify, the default RF will not give you oob
    random_state=0,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

print("OOB score:", rf_clf.oob_score_)
acc = accuracy_score(y_test, rf_clf.predict(X_test))
print("Test accuracy:", acc)
```
Here, **Out-of-Bag (OOB) Score** gives an internal validation accuracy, while **test accuracy** confirms performance on held-out data.


```{admonition} How the OOB score is calculated
1. For each training point, collect predictions only from the trees that did **not** see that point during training (its out-of-bag trees).
2. Aggregate those predictions (majority vote for classification, mean for regression).
3. Compare aggregated predictions against the true labels.
4. The accuracy (or R² for regression) across all training samples is the **OOB score**.
```

Forests average over many trees, so feature importance is more reliable than from a single tree. We can view both Gini importance and permutation importance.
```{code-cell} ipython3
imp_rf = pd.Series(rf_clf.feature_importances_, index=feat_names).sort_values()
imp_rf.plot(kind="barh")
plt.title("Random Forest Gini importance")
plt.show()

perm_rf = permutation_importance(rf_clf, X_test, y_test, scoring="accuracy", n_repeats=20, random_state=0)
pd.Series(perm_rf.importances_mean, index=feat_names).sort_values().plot(kind="barh")
plt.title("Random Forest permutation importance (test)")
plt.show()
```

### 3.2 Regression forest on Melting Point
Next we apply the same idea for regression. The forest predicts a continuous value (melting point) by averaging predictions from many regression trees.
```{code-cell} ipython3
rf_reg = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=3,
    max_features="sqrt",
    oob_score=True,
    random_state=0,
    n_jobs=-1
)
rf_reg.fit(Xr_train, yr_train)

print("OOB R2:", rf_reg.oob_score_)
yhat_rf = rf_reg.predict(Xr_test)
print(f"Test R2: {r2_score(yr_test, yhat_rf):.3f}")
print(f"Test MAE: {mean_absolute_error(yr_test, yhat_rf):.3f}")
```

Parity and feature importance plots help check performance.

```{code-cell} ipython3
# Parity plot
plt.scatter(yr_test, yhat_rf, alpha=0.6)
lims = [min(yr_test.min(), yhat_rf.min()), max(yr_test.max(), yhat_rf.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Parity plot: Random Forest regressor")
plt.show()

# Feature importance
pd.Series(rf_reg.feature_importances_, index=Xr_train.columns).sort_values().plot(kind="barh")
plt.title("Random Forest importance (regression)")
plt.show()
```

---
## 4. Ensembles

Ensembles combine multiple models to improve stability and accuracy. Here we expand on trees with forests, simple averaging, voting, and boosting, and add **visual comparisons** to show their differences.

---

### 4.1 Experiment: single tree vs random forest

We repeat training several times with different random seeds for the train/test split. For each split:
- Fit a single unpruned tree.
- Fit a random forest with 300 trees.
- Record the test R² for both.

```{code-cell} ipython3
splits = [1, 7, 21, 42, 77]
rows = []
for seed in splits:
    X_tr, X_te, y_tr, y_te = train_test_split(Xr, yr, test_size=0.2, random_state=seed)
    t = DecisionTreeRegressor(max_depth=None, min_samples_leaf=3, random_state=seed).fit(X_tr, y_tr)
    f = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
    r2_t = r2_score(y_te, t.predict(X_te))
    r2_f = r2_score(y_te, f.predict(X_te))
    rows.append({"seed": seed, "Tree_R2": r2_t, "Forest_R2": r2_f})

df_cmp = pd.DataFrame(rows).round(3)
df_cmp
```

```{code-cell} ipython3
plt.plot(df_cmp["seed"], df_cmp["Tree_R2"], "o-", label="Tree")
plt.plot(df_cmp["seed"], df_cmp["Forest_R2"], "o-", label="Forest")
plt.xlabel("random_state")
plt.ylabel("R2 on test")
plt.title("Stability across splits")
plt.legend()
plt.grid(True)
plt.show()
```

```{admonition} Why forests are often a safe and strong default model
- **Single tree**: R² jumps up and down depending on the seed. Sometimes the tree performs decently, sometimes it collapses.  
- **Random forest**: R² is consistently higher and more stable. By averaging across 300 trees trained on different bootstrap samples and feature subsets, the forest cancels out randomness.  
Forests trade a bit of interpretability for much more reliability.
```

---

### 4.2 Simple Ensembles by Model Averaging

Random forests are powerful, but ensembling can be simpler. Even just **averaging two different models** can improve performance.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

tree = DecisionTreeRegressor(max_depth=4).fit(Xr_train, yr_train)
lin  = LinearRegression().fit(Xr_train, yr_train)

pred_tree = tree.predict(Xr_test)
pred_lin  = lin.predict(Xr_test)
pred_avg = (pred_tree + pred_lin) / 2.0

df_avg = pd.DataFrame({
    "True": yr_test,
    "Tree": pred_tree,
    "Linear": pred_lin,
    "Average": pred_avg
}).head(10)
df_avg
```

```{code-cell} ipython3
print("Tree R2:", r2_score(yr_test, pred_tree))
print("Linear R2:", r2_score(yr_test, pred_lin))
print("Averaged R2:", r2_score(yr_test, pred_avg))
```

```{code-cell} ipython3
plt.scatter(yr_test, pred_tree, alpha=0.5, label="Tree")
plt.scatter(yr_test, pred_lin, alpha=0.5, label="Linear")
plt.scatter(yr_test, pred_avg, alpha=0.5, label="Average")
lims = [min(yr_test.min(), pred_tree.min(), pred_lin.min()), max(yr_test.max(), pred_tree.max(), pred_lin.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Parity plot: Tree vs Linear vs Average")
plt.legend()
plt.show()
```

```{admonition} Difference
- **Tree**: captures nonlinear shapes but may overfit.  
- **Linear**: very stable but may underfit.  
- **Average**: balances the two, smoother than tree, more flexible than linear regression.  
```

---

### 4.3 Simple Ensembles by Voting

Voting is most common for classification. Each model votes for a class.  
- **Hard voting**: majority wins.  
- **Soft voting**: average predicted probabilities.

```{code-cell} ipython3
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

vote_clf_soft = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=500, random_state=0)),
        ("svc", SVC(probability=True, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
    ],
    voting="soft"
).fit(X_train, y_train)

print("Soft Voting classifier accuracy:", accuracy_score(y_test, vote_clf_soft.predict(X_test)))


vote_clf_hard = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=500, random_state=0)),
        ("svc", SVC(probability=True, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1))
    ],
    voting="hard"
).fit(X_train, y_train)

print("Hard Voting classifier accuracy:", accuracy_score(y_test, vote_clf_hard.predict(X_test)))

```

Compare voting against individual models:

```{code-cell} ipython3
acc_lr = accuracy_score(y_test, LogisticRegression(max_iter=500, random_state=0).fit(X_train,y_train).predict(X_test))
acc_svc = accuracy_score(y_test, SVC(probability=True, random_state=0).fit(X_train,y_train).predict(X_test))
acc_rf  = accuracy_score(y_test, RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_train,y_train).predict(X_test))

pd.DataFrame({
    "Model": ["LogReg", "SVC", "RandomForest", "Voting-Hard", "Voting-Soft"],
    "Accuracy": [acc_lr, acc_svc, acc_rf, accuracy_score(y_test, vote_clf_hard.predict(X_test)), accuracy_score(y_test, vote_clf_soft.predict(X_test))]
})
```

```{admonition} Difference
- **Individual models**: Logistic regression handles linear patterns, SVM picks margins, random forest handles nonlinear rules.  
- **Voting ensemble**: combines their strengths, reducing the chance of one weak model dominating.  
```

---

### 4.4 (Optional Topic) Boosting: a different ensemble structure

Boosting is sequential: each tree corrects the mistakes of the last. Sometimes it can be stronger than bagging, but needs careful tuning.

```{code-cell} ipython3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

gb_reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=0
).fit(Xr_train, yr_train)

yhat_gb = gb_reg.predict(Xr_test)
print(f"Gradient Boosting R2: {r2_score(yr_test, yhat_gb):.3f}")
print(f"Gradient Boosting MAE: {mean_absolute_error(yr_test, yhat_gb):.2f}")
```

Compare **Random Forest vs Gradient Boosting** directly:

```{code-cell} ipython3
rf_reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(Xr_train, yr_train)
yhat_rf = rf_reg.predict(Xr_test)

pd.DataFrame({
    "Model": ["RandomForest", "GradientBoosting"],
    "R2": [r2_score(yr_test, yhat_rf), r2_score(yr_test, yhat_gb)],
    "MAE": [mean_absolute_error(yr_test, yhat_rf), mean_absolute_error(yr_test, yhat_gb)]
}).round(3)
```

```{code-cell} ipython3
plt.scatter(yr_test, yhat_rf, alpha=0.5, label="RandomForest")
plt.scatter(yr_test, yhat_gb, alpha=0.5, label="GradientBoosting")
lims = [min(yr_test.min(), yhat_rf.min(), yhat_gb.min()), max(yr_test.max(), yhat_rf.max(), yhat_gb.max())]
plt.plot(lims, lims, "k--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Parity plot: RF vs GB")
plt.legend()
plt.show()
```

```{admonition} Compare
- **Random forest**: reduces variance by averaging many deep trees. Stable and easy to tune.  
- **Boosting**: reduces bias by sequentially correcting mistakes. Often achieves higher accuracy but requires careful parameter tuning.  
```

---

## 5. End-to-end recipe for random forest

Now, let's put everything we learn for trees and forests together. Below is a standard workflow for toxicity with a forest. Similar to what we learned from last lecture but handled with RF model.

```{code-cell} ipython3
# 1) Data
X = df_clf[["MolWt", "LogP", "TPSA", "NumRings"]]
y = df_clf["Toxicity"].str.lower().map(label_map).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, stratify=y
)

# 2) Model
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, max_features="sqrt",
    min_samples_leaf=3, oob_score=True, random_state=15, n_jobs=-1
).fit(X_train, y_train)

# 3) Evaluate
y_hat = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]
print(f"OOB score: {rf.oob_score_:.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_hat):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```
Additional plots demonstrating the performance:

```{code-cell} ipython3
# Confusion matrix and ROC
cm = confusion_matrix(y_test, y_hat)
plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

fpr, tpr, thr = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, lw=2); plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
plt.show()
```

---

## 6. Quick reference

```{admonition} Common options
- DecisionTreeClassifier/Regressor: `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion`, `random_state`
- RandomForestClassifier/Regressor: add `n_estimators`, `max_features`, `oob_score`, `n_jobs`
- Use `feature_importances_` for built-in importance and `permutation_importance` for model-agnostic view
```
```{admonition} When to use
- Tree: simple rules, quick to interpret on small depth
- Forest: stronger accuracy, more stable, less sensitive to a single split
```

---

## 7. In-class activities

### 7.1 Tree vs Forest on log-solubility

- Create `y_log = log10(Solubility_mol_per_L + 1e-6)`
- Use features `[MolWt, LogP, TPSA, NumRings]`
- Train a `DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)` and a `RandomForestRegressor(n_estimators=300, min_samples_leaf=5)`
- Report test **R2** for both and draw both parity plots

```python
# TO DO
```

### 7.2 Pruning with `min_samples_leaf`

- Fix `max_depth=None` for `DecisionTreeClassifier` on toxicity
- Sweep `min_samples_leaf` in `[1, 2, 3, 5, 8, 12, 20]`
- Plot test **accuracy** vs `min_samples_leaf`

```python
# TO DO
```

### 7.3 OOB sanity check

- Train `RandomForestClassifier` with `oob_score=True` on toxicity
- Compare OOB score to test accuracy over seeds `[0, 7, 21, 42]`

```python
# TO DO
```

### 7.4 Feature importance agreement

- On melting point, compute forest `feature_importances_` and `permutation_importance`
- Plot both and comment where they agree or disagree

```python
# TO DO
```

### 7.5 Small tree visualization

- Fit a `DecisionTreeClassifier(max_depth=2)` on toxicity
- Use `plot_tree` to draw it and write down the two split rules in plain language

```python
# TO DO
```

---
