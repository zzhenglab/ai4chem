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

## lecture 13


```{code-cell} ipython3
from torch.utils.data import TensorDataset, DataLoader

dl_desc = DataLoader(
    TensorDataset(torch.from_numpy(Xz.astype(np.float32))),
    batch_size=64,
    shuffle=True,
)

# 3D latent AE and training loop (unpack with (xb,))
ae3 = TinyAE(in_dim=10, hid=64, z_dim=3)
opt = optim.Adam(ae3.parameters(), lr=1e-3)

for ep in range(4):
    for (xb,) in dl_desc:
        xr, z = ae3(xb)
        loss = nn.functional.mse_loss(xr, xb)
        opt.zero_grad(); loss.backward(); opt.step()

# Encode with the trained model
with torch.no_grad():
    Z3 = ae3.encode(torch.from_numpy(Xz.astype(np.float32))).numpy()


fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")

p = ax.scatter(
    Z3[:, 0], Z3[:, 1], Z3[:, 2],
    c=df_small["LogP"].values,
    s=12, alpha=0.85
)
ax.set_xlabel("z0"); ax.set_ylabel("z1"); ax.set_zlabel("z2")
ax.set_title("AE latent space (3D), color = LogP")
cb = fig.colorbar(p, ax=ax, shrink=0.7, pad=0.1)
cb.set_label("LogP")
plt.show()


```


q2

```{code-cell} ipython3
for t in [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2]:
    raw = sample_smiles(n=800, temp=t)
    val = len(canonicalize_batch(raw)) / max(1, len(raw))
    print(f"T={t}: validity {val:.2f}")

```

#Lecture 14

```{code-cell} ipython3
# Shared pieces
U_cand = candidate_cloud(m=6000, seed=1)

def rf_mean_std(rf, Xc):
    preds = np.stack([est.predict(Xc) for est in rf.estimators_], axis=1)
    mu = preds.mean(axis=1)
    sd = preds.std(axis=1)
    return mu, sd

def run_rf_bo_ei_estimators(n_estimators=100, n_iter=20, seed=42, noise_sd=1.8):
    rng = np.random.RandomState(seed)
    # same initial 8 runs for fairness
    U0 = rng.rand(8, 3)
    lab0 = np.array([decode_3d(u) for u in U0])
    y0 = suzuki_yield(lab0[:,0], lab0[:,1], lab0[:,2], rng=rng)

    U = U0.copy()
    y = y0.copy()
    best_hist = [y.max()]
    rf_snap = None

    for t in range(n_iter):
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=seed + t,
            n_jobs=-1
        )
        rf.fit(U, y)
        mu_rf, sd_rf = rf_mean_std(rf, U_cand)
        ei = acq_ei(mu_rf, sd_rf, y_best=y.max(), xi=0.01)
        u_next, _ = argmax_on_grid(ei, U_cand)
        lab_next = decode_3d(u_next.ravel())
        y_next = suzuki_yield(lab_next[0], lab_next[1], lab_next[2], rng=rng)
        U = np.vstack([U, u_next])
        y = np.hstack([y, y_next])
        best_hist.append(y.max())
        if t == n_iter - 1:
            rf_snap = (U.copy(), y.copy(), mu_rf.copy(), sd_rf.copy())
    return np.array(best_hist), rf_snap

def run_gp_bo_ei(n_iter=20, seed=42):
    rng = np.random.RandomState(seed)
    U0 = rng.rand(8, 3)
    lab0 = np.array([decode_3d(u) for u in U0])
    y0 = suzuki_yield(lab0[:,0], lab0[:,1], lab0[:,2], rng=rng)

    U = U0.copy()
    y = y0.copy()
    best_hist = [y.max()]

    kernel3 = C(50.0) * Matern(length_scale=[0.2,0.2,0.2], nu=2.5) + WhiteKernel(1.0)
    gp3 = GaussianProcessRegressor(kernel=kernel3, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
    gp_snap = None

    for t in range(n_iter):
        gp3.fit(U, y)
        mu, sd = gp3.predict(U_cand, return_std=True)
        ei = acq_ei(mu, sd, y_best=y.max(), xi=0.01)
        u_next, _ = argmax_on_grid(ei, U_cand)
        lab_next = decode_3d(u_next.ravel())
        y_next = suzuki_yield(lab_next[0], lab_next[1], lab_next[2], rng=rng)
        U = np.vstack([U, u_next])
        y = np.hstack([y, y_next])
        best_hist.append(y.max())
        if t == n_iter - 1:
            gp_snap = (U.copy(), y.copy(), mu.copy(), sd.copy())
    return np.array(best_hist), gp_snap

# 1) RF 100
hist_rf100, snap_rf100 = run_rf_bo_ei_estimators(n_estimators=100,  n_iter=20, seed=10)
# 2) RF 1000
hist_rf1000, snap_rf1000 = run_rf_bo_ei_estimators(n_estimators=1000, n_iter=20, seed=10)
# 3) GP
hist_gp, snap_gp = run_gp_bo_ei(n_iter=20, seed=10)

# 4) Compare best-so-far
plt.figure(figsize=(7,4))
plt.plot(hist_rf100,  marker="o", label="RF+EI 100 trees")
plt.plot(hist_rf1000, marker="s", label="RF+EI 1000 trees")
plt.plot(hist_gp,     marker="^", label="GP+EI")
plt.xlabel("Iteration"); plt.ylabel("Best observed yield")
plt.title("Suzuki BO: RF size vs GP")
plt.legend(); plt.grid(False); plt.show()

# 5) FYI: bands for RF-100 vs RF-1000 on U_cand
for tag, snap in [("RF 100", snap_rf100), ("RF 1000", snap_rf1000)]:
    U_obs, y_obs, mu_final, sd_final = snap
    plt.figure(figsize=(7,4))
    plt.hist(sd_final, bins=40)
    plt.title(f"{tag}: distribution of RF posterior sd on candidate cloud")
    plt.xlabel("sd across trees"); plt.ylabel("count"); plt.show()
```



