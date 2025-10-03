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

