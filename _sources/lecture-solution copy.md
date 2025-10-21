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




## 6. In-class activity

### Q1. Identify PU pieces from LU toy data
Using the LU toy data you built (`X_lab`, `y_lab_noisy`, `X_unlab`), create a PU view by revealing only a fraction of the true positives as labeled. Report the shapes of P and U.

**Task**
1. From `y_lab_noisy`, take indices where `y=1`.
2. Reveal 30% at random as P.
3. Put the rest of `X_lab` plus all `X_unlab` into U.
4. Print shapes of P and U.

```{code-cell} ipython3
rng = np.random.default_rng(17)

pos_idx = np.where(y_lab_noisy == 1)[0]
reveal = rng.random(len(pos_idx)) < 0.30
P_idx = pos_idx[reveal]

X_P = X_lab[P_idx]
mask_lab_rest = np.ones(len(X_lab), dtype=bool)
mask_lab_rest[P_idx] = False
X_U = np.vstack([X_lab[mask_lab_rest], X_unlab])

print("P shape:", X_P.shape, " U shape:", X_U.shape)
```

---

### Q2. Train the s-model and estimate c-hat under SCAR
Train a classifier to predict selection `s` using P vs U. Then estimate `ĉ` by averaging `P(s=1|x)` on a held out split of P.

**Task**
1. Build `X_s`, `s_lab`.
2. Fit `LogisticRegression` or `RandomForestClassifier` (you decide).
3. Split P into train and holdout, compute `ĉ` as mean predicted probability on the holdout.

```{code-cell} ipython3
# Build s-dataset
X_s = np.vstack([X_P, X_U])
s_lab = np.hstack([np.ones(len(X_P), dtype=int), np.zeros(len(X_U), dtype=int)])

sc_s = StandardScaler().fit(X_s)
Xs_s = sc_s.transform(X_s)

s_clf = LogisticRegression(max_iter=400, random_state=17).fit(Xs_s, s_lab)

# Estimate c-hat on a P holdout
XP_tr, XP_ho = train_test_split(X_P, test_size=0.2, random_state=17)
c_hat = s_clf.predict_proba(sc_s.transform(XP_ho))[:, 1].mean() if len(XP_ho) else 0.5
print("c_hat:", round(float(c_hat), 4))
```

---

### Q3. Convert s-scores to PU probabilities and rank candidates
Use the Elkan–Noto link under SCAR. Compute `P(y=1|x) ≈ P(s=1|x)/ĉ` and rank the top 10 U points.

**Task**
1. Score all U with the s-model.
2. Divide by `ĉ` and clip to `[0,1]`.
3. Show indices of the top 10 U points by PU probability.

```{code-cell} ipython3
p_s_u = s_clf.predict_proba(sc_s.transform(X_U))[:, 1]
p_y_u = np.clip(p_s_u / max(1e-6, c_hat), 0, 1)

top10 = np.argsort(-p_y_u)[:10]
pd.DataFrame({"u_index": top10, "p_y_hat": p_y_u[top10]}).reset_index(drop=True)
```

---

### Q4. Threshold selection by prior and by quantile
Pick candidates from U using two rules. Compare how many items each rule selects.

**Task**
1. Prior-guided threshold: estimate `π̂ = mean(p_y)` on the full candidate set `X_lab ∪ X_unlab` then pick top `π̂` fraction of U.
2. Fixed-quantile threshold: pick top 25% of U.
3. Report counts.

```{code-cell} ipython3
# Score everyone to estimate a coarse prior
X_all = np.vstack([X_lab, X_unlab])
p_s_all = s_clf.predict_proba(sc_s.transform(X_all))[:, 1]
p_y_all = np.clip(p_s_all / max(1e-6, c_hat), 0, 1)

pi_hat = float(p_y_all.mean())

# 1) Prior-guided threshold
thr_pi = np.quantile(p_y_u, 1 - min(max(pi_hat, 0.01), 0.99))
sel_pi = np.where(p_y_u >= thr_pi)[0]

# 2) Fixed-quantile 25%
thr_q = np.quantile(p_y_u, 0.75)
sel_q = np.where(p_y_u >= thr_q)[0]

print(f"pi_hat={pi_hat:.3f}  thr_pi={thr_pi:.3f}  selected_by_prior={len(sel_pi)}")
print(f"thr_q(25%)={thr_q:.3f}  selected_by_quantile={len(sel_q)}")
```
