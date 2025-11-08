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

# Lecture 22 - Vision Languge Models

```{contents}
:local:
:depth: 1
```

## Learning goals
- Understand the core idea of **CLIP**: aligning image and text representations with a contrastive loss.
- Build a tiny **zero-shot classifier** for crystal images and compute simple metrics.
- Train a small **linear probe** on top of frozen CLIP embeddings.
- Visualize **embedding structure** with PCA and t-SNE.
- Start a **vision chat** with a GPT model using the Responses API.

 [![Colab](https://img.shields.io/badge/Open-Colab-orange)](https://colab.research.google.com/drive/1ihq56j_5Khl2PY1pDqUlTb1QDu2Uf1Ck?usp=sharing) 



## 1. Setup

```{code-cell} ipython3
:tags: [hide-input]
# pip installs are kept separate so you can see which step fails.
try:
    import torch, torchvision, PIL, sklearn, matplotlib, numpy  # quick smoke test
except Exception as e:
    print("Installing required packages...")
#     %pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#     %pip -q install ftfy regex tqdm onnxruntime matplotlib scikit-learn pillow einops requests

# Install CLIP from the official repository
try:
    import clip  # noqa
except Exception:
    %pip -q install git+https://github.com/openai/CLIP.git

from openai import OpenAI  # noqa

#     %pip -q install openai>=1.40.0
# 0.2 Imports and basic config
import os, json, math, random, io, textwrap, time, pathlib, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import requests
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from openai import OpenAI
from io import BytesIO

import clip  # OpenAI CLIP
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
```

## 2. CLIP: Connecting Text and Images

CLIP learns two encoders:
1. an **image encoder** $f_{\theta}(\cdot)$
2. a **text encoder** $g_{\phi}(\cdot)$

Each image–text pair in a batch is projected into a common space. Similar pairs should have **high cosine similarity**, mismatched pairs should be low.


We scale cosine similarities by a learnable temperature $\tau$ and use a symmetric cross entropy.
For a batch of size $N$ with normalized embeddings
$U = \{u_i\}$ for images and $V = \{v_i\}$ for texts,

$
s_{ij} = \frac{u_i^\top v_j}{\tau}
$

The loss is:
$
\mathcal{L} = \frac{1}{2}
\left[
  \frac{1}{N}\sum_{i=1}^N -\log \frac{e^{s_{ii}}}{\sum_{j=1}^N e^{s_{ij}}}
  +
  \frac{1}{N}\sum_{i=1}^N -\log \frac{e^{s_{ii}}}{\sum_{j=1}^N e^{s_{ji}}}
\right]
$


In plain words, matching image–text pairs get pushed together and everything else gets pushed apart.
This simple recipe is very effective for **zero-shot** use.

We start with a small ViT model to keep memory use friendly.

```{code-cell} ipython3

model_name = "ViT-B/32"
model, preprocess = clip.load(model_name, device=device, jit=False)
model.eval()
print(type(model))
#print(preprocess)  
```

The `preprocess` transform will resize, center-crop, and normalize images. We apply it before encoding.

```{code-cell} ipython3
# Peek at text tokenizer behavior
tokens = clip.tokenize(["a microscopy image of a crystal", "a photo of a dog"]).to(device)
print("Token tensor shape:", tokens.shape)
print(tokens[0, :10])
```

Now, let's bring in crystal images from a GitHub folder.

We will use the same dataset from the last lecture.

```{code-cell} ipython3
GITHUB_DIR_API = "https://api.github.com/repos/zzhenglab/ai4chem/contents/book/_data?ref=main"

# Labels are determined from filename prefixes.
# Images whose names start with "crystal_" will be labeled "crystal",
# and those starting with "nocrystal_" will be labeled "nocrystal".
LABELS = ["crystal", "nocrystal"]

print("Folder (GitHub API):", GITHUB_DIR_API)
print("Labels:", LABELS)

def fetch_image(url: str):
    """
    Downloads an image from the given URL and returns it as a Pillow Image object.
    Converts the image to RGB format for consistency.
    If the download or decoding fails, returns None instead of crashing.
    """
    try:
        r = requests.get(url, timeout=10)   # Fetch the image file from GitHub
        r.raise_for_status()                # Raise an error for bad responses (e.g., 404)
        return Image.open(io.BytesIO(r.content)).convert("RGB")  # Convert bytes to image
    except Exception as e:
        print("Skip:", url, "|", e)         # Log and skip any problematic file
        return None

def list_relevant_files():
    """
    Queries the GitHub folder for all files and filters them to include only
    PNG images whose filenames start with 'crystal_' or 'nocrystal_'.
    Returns a list of dictionaries containing filenames and direct download URLs.
    """
    r = requests.get(GITHUB_DIR_API, timeout=15)  # Get folder contents from GitHub API
    r.raise_for_status()
    items = r.json()                              # Parse JSON response into Python objects
    wanted = []
    for it in items:
        if it.get("type") != "file":              # Skip subfolders or non-file items
            continue
        name = it.get("name", "")
        if not name.lower().endswith(".png"):     # Skip non-PNG files
            continue
        low = name.lower()
        if low.startswith("crystal_") or low.startswith("nocrystal_"):
            # If file name matches one of the two prefixes, save it
            url = it.get("download_url")          # Direct link to the raw file
            if url:
                wanted.append({"name": name, "url": url})
    return wanted

def make_dataset():
    """
    Builds a dataset by downloading all relevant PNG files,
    assigning each a label based on its filename prefix,
    and storing both the image and its metadata.
    Returns a list of records (dicts).
    """
    records = []
    for f in list_relevant_files():
        name = f["name"]
        url = f["url"]
        # Infer the label from the filename prefix
        label = "crystal" if name.lower().startswith("crystal_") else "nocrystal"
        img = fetch_image(url)                  # Download and decode the image
        if img is None:
            continue                            # Skip missing or broken images
        records.append({"label": label, "url": url, "name": name, "pil": img})
    return records

# Build the dataset by fetching and labeling all matching images
dataset = make_dataset()
print("Images loaded:", len(dataset))

# Display one sample image (in Jupyter or similar environment)
if len(dataset) > 0:
    display(dataset[0]["pil"])

# Create a tidy DataFrame summarizing the dataset for later analysis
df = pd.DataFrame(
    {
        "name": [r["name"] for r in dataset],       # filename
        "label": [r["label"] for r in dataset],     # crystal or nocrystal
        "url": [r["url"] for r in dataset],         # direct GitHub URL
        "width": [r["pil"].width for r in dataset], # image width in pixels
        "height": [r["pil"].height for r in dataset]# image height in pixels
    }
)

# Show the first few rows to confirm the data structure
df.head()
```

Now we will encode images and texts with CLIP.

We compute normalized embeddings for both modalities. Shapes matter a lot, so we print them.

```{code-cell} ipython3
# 4.1 Build a small image batch
image_tensors = []
image_labels = []
image_urls = []
for row in dataset:
    image_tensors.append(preprocess(row["pil"]))
    image_labels.append(row["label"])
    image_urls.append(row["url"])

if len(image_tensors) == 0:
    print("No images in dataset. Please fix paths in CLASS_TO_FILES.")
else:
    image_batch = torch.stack(image_tensors).to(device)
    print("image_batch:", tuple(image_batch.shape), "dtype:", image_batch.dtype)

# 4.2 Build text prompts
class_names = ["crystal", "nocrystal"]

nice_label = {
    "crystal":   "a crystal",
    "nocrystal": "nothing present"
}

templates = [
    "an image showing {label}",
    "a photo of {label} structure",
    "a scientific image of {label}"
]

def build_prompts(class_names, templates, nice_label):
    # Order: for each class i, all templates j  -> matches zero_shot_classify pooling
    prompts = []
    for cname in class_names:
        text_label = nice_label.get(cname, cname)
        for t in templates:
            prompts.append(t.format(label=text_label))
    return prompts

prompts = build_prompts(class_names, templates, nice_label)
tokenized = clip.tokenize(prompts).to(device)
print(f"{len(prompts)} prompts built:", prompts[:4], "...")
print("Number of prompts:", len(prompts), "tokenized:", tuple(tokenized.shape))

# 4.3 Forward through CLIP to get embeddings
if len(image_tensors) > 0:
    with torch.no_grad():
        img_feats = model.encode_image(image_batch)
        txt_feats = model.encode_text(tokenized)

    # Normalize to unit length
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    print("img_feats:", tuple(img_feats.shape))
    print("txt_feats:", tuple(txt_feats.shape))
```

Below you will see how the image embeddings looks like for images.
We have 240 images and each are 512D.

```{code-cell} ipython3
print("img_feats:", tuple(img_feats.shape))
img_feats
```

Below you will see how the text embeddings looks like.
We have 8 text embeddings and each are 512D.

```{code-cell} ipython3
print("txt_feats:", tuple(txt_feats.shape))
txt_feats
```

What do the embeddings look like?

We reduce dimensionality to 2D so we can plot and inspect cluster structure.
We expect images from the same label to be near each other.

```{code-cell} ipython3
if len(image_tensors) > 0:
    X = img_feats.detach().cpu().numpy()
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(5,4))
    for lab in sorted(set(image_labels)):
        idx = [i for i, y in enumerate(image_labels) if y == lab]
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=lab, alpha=0.8)
    plt.legend()
    plt.title("CLIP image embeddings - PCA")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(True)
    plt.show()
else:
    print("No images to plot")
```

```{admonition} Note
PCA is linear and fast. t-SNE can separate clusters further but is slower and higher variance.
```

```{code-cell} ipython3
if len(image_tensors) > 0 and len(image_tensors) <= 200:
    tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(image_tensors)-1))
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(5,4))
    for lab in sorted(set(image_labels)):
        idx = [i for i, y in enumerate(image_labels) if y == lab]
        plt.scatter(X_tsne[idx,0], X_tsne[idx,1], label=lab, alpha=0.8)
    plt.legend()
    plt.title("CLIP image embeddings - t-SNE")
    plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2")
    plt.grid(True)
    plt.show()
```

## 3. Prediction based on Embeddings

Before we evaluate performance, we’ll compare each image embedding to the text embeddings created from our prompts.

Both image and text vectors come from the same pretrained model, so we can measure how similar they are using cosine similarity.

Each image is assigned to the class whose text description is most similar to it — no training, just comparison in the shared embedding space.
Then we check how well those predictions match the true labels from our dataset.

```{code-cell} ipython3
def zero_shot_classify(img_feats, txt_feats, class_names, templates, temperature=1.0):
    # Normalize both sets of features so cosine similarity = dot product
    img_z = F.normalize(img_feats, dim=1)            # [N, d] image embeddings
    txt_z = F.normalize(txt_feats, dim=1)            # [C*T, d] text embeddings (C classes × T templates)

    # Group text prompts belonging to each class
    T = len(templates)
    per_class = [list(range(i*T, (i+1)*T)) for i in range(len(class_names))]

    # Compute image–text similarities and average over prompts of the same class
    sims = img_z @ txt_z.t()                         # [N, C*T] cosine similarities
    sims = sims / max(1e-8, temperature)             # optional scaling (lower = sharper differences)
    pooled = torch.stack([sims[:, idxs].mean(dim=1) for idxs in per_class], dim=1)  # [N, C]

    # Take the class with the highest mean similarity as prediction
    preds = pooled.argmax(dim=1).cpu().numpy()
    return preds, pooled.cpu().numpy()

# Run zero-shot classification and evaluate
if len(image_tensors) > 0:
    # Convert string labels in the dataset to numeric indices
    y_true = np.array([class_names.index(r["label"]) for r in dataset])

    # Predict using the zero-shot classifier
    preds, pooled = zero_shot_classify(img_feats, txt_feats, class_names, templates, temperature=0.01)

    # Report accuracy and detailed metrics
    print("Accuracy:", accuracy_score(y_true, preds))
    print(classification_report(y_true, preds, target_names=class_names, digits=3, zero_division=0))
```

Now, we will run linear probe on frozen embeddings.

A simple linear model can be trained quickly on top of CLIP features when you have a few labeled images.

Note: below we only use `img_feats` for the prediction.

```{code-cell} ipython3
if len(image_tensors) > 0:
    # Features and integer labels
    X = img_feats.detach().cpu().numpy()
    y = np.array([class_names.index(c) for c in image_labels])

    # Try a stratified 80:20 split for balanced class representation
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
    except ValueError:
        # Happens when at least one class has only a single sample
        # Fall back to a plain random 80:20 split without stratify
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=42, shuffle=True
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")
    print("Linear probe test accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))
```

Below shows the confusion matrix and the ROC curve for our test set.

```{code-cell} ipython3
# Confusion matrix (counts and normalized)
cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=ax[0], colorbar=False)
ax[0].set_title("Confusion matrix")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
disp.plot(ax=ax[1], colorbar=False, values_format=".2f")
ax[1].set_title("Confusion matrix (normalized)")
plt.tight_layout()
plt.show()

# ROC curve for binary case (optional)
if len(class_names) == 2:
    try:
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_test)
        else:
            scores = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=class_names.index(class_names[1]))
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("ROC skipped:", e)
```

To better visualize this. First, we prep features, score zero shot, pick up to two correct hits per class, and print quick snapshots of the first 8 dims for image, text, and the simple average merge.

```{code-cell} ipython3
# === Part 1: prep, scoring, picks, and 8-dim snapshots ===
# Idea:
# 1) normalize features for cosine math
# 2) compute zero-shot scores per class (average over templates)
# 3) choose up to 2 correct examples per class
# 4) print a compact peek of the first 8 dims for image, text, and merged

import numpy as np, torch, torch.nn.functional as F
np.set_printoptions(precision=4, suppress=True)

# Repro for any random sampling below
rng = np.random.default_rng(9)

# ----- Normalize features -----
# img_feats: [N, d], txt_feats: [C*T, d]
img_test = img_feats[test_idx] if isinstance(img_feats, torch.Tensor) else torch.tensor(img_feats[test_idx])
img_test_z = F.normalize(img_test, dim=1)  # [N_test, d]

T, C = len(templates), len(class_names)
txt_z_in = txt_feats if isinstance(txt_feats, torch.Tensor) else torch.tensor(txt_feats)
txt_z = F.normalize(txt_z_in, dim=1)       # [C*T, d]

# ----- Zero-shot scores and predictions -----
# sims: pairwise cosine similarities to all prompts
sims = img_test_z @ txt_z.t()  # [N_test, C*T]

# pool templates per class by mean
pooled = torch.stack([sims[:, range(c*T, (c+1)*T)].mean(1) for c in range(C)], 1)
preds = pooled.argmax(1).cpu().numpy()

# ----- Pick up to 2 correct examples per class -----
picks = []
for c in range(C):
    ok = np.where((y_test == c) & (preds == c))[0]
    if len(ok) > 0:
        picks.extend(rng.choice(ok, size=min(2, len(ok)), replace=False).tolist())

# ----- Print 8-dim snapshots -----
print("=== Embedding snapshots (first 8 dims) ===")
for j, loc in enumerate(picks):
    c = int(preds[loc])
    v_img   = img_test_z[loc].cpu().numpy()
    v_text  = txt_z[c*T].cpu().numpy()
    v_merge = F.normalize((img_test_z[loc] + txt_z[c*T]) / 2, dim=0).cpu().numpy()

    # Clear labeling helps when scanning outputs
    print(f"\nExample {j+1} | Label={class_names[c]}")
    print("image :", v_img[:8])
    print("text  :", v_text[:8])
    print("merged:", v_merge[:8])

# Quick sanity display
print(f"\nPicked {len(picks)} examples across {C} classes.")
```

between steps, we also want a 2D view to compare spaces with a shared projector. next, we fit PCA on the image-only space, then project the text and merged points with the same transform. to make this cell display something even without plotting, it prints explained variance and a tiny table of projected coords for the chosen examples.

```{code-cell} ipython3
# === Part 2: PCA fit on image space; project text and merged ===
# Plan: fit PCA on image embeddings, then transform text-only and merged pairs
# We also print explained variance and a small table of the 2D coords.

from sklearn.decomposition import PCA

# Fit PCA on image-only space
pca = PCA(n_components=2, random_state=42)
X2_img = pca.fit_transform(img_test_z.cpu().numpy())  # [N_test, 2]

# Project the chosen text and merged vectors using the same PCA
X2_text, X2_merge = [], []
for loc in picks:
    c = int(preds[loc])
    v_text  = txt_z[c*T].cpu().numpy()[None, :]  # shape [1, d]
    v_merge = F.normalize((img_test_z[loc] + txt_z[c*T]) / 2, dim=0).cpu().numpy()[None, :]

    X2_text.append(pca.transform(v_text)[0])
    X2_merge.append(pca.transform(v_merge)[0])

X2_text  = np.array(X2_text)
X2_merge = np.array(X2_merge)

# ----- Display: explained variance and a tiny coordinate table -----
evr = pca.explained_variance_ratio_
print("=== PCA explained variance ratio ===")
print(f"PC1: {evr[0]:.4f}, PC2: {evr[1]:.4f}")

print("\n=== Sample 2D coords for picks ===")
print("idx | label             | text_x   text_y   | merge_x  merge_y")
for (tx, ty), (mx, my), loc in zip(X2_text, X2_merge, picks):
    lab = class_names[int(y_test[loc])]
    print(f"{loc:3d} | {lab:<16} | {tx:7.3f} {ty:7.3f} | {mx:7.3f} {my:7.3f}")
```

Finally, the side by side figure. Left is the image cloud with thumbnails for the chosen points. Middle places the projected text points. Right shows the merged points.

```{code-cell} ipython3
# === Part 3: 3-panel plot: image-only, text-only, merged ===
# Notes:
# - left: image embeddings with per-class scatter and thumbnails for picks
# - middle: text points for the picks
# - right: merged (image + predicted-class text) for the picks

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ----- Panel 1: image-only with thumbnails -----
ax = axes[0]
for c, name in enumerate(class_names):
    pts = X2_img[y_test == c]
    ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.3, label=name)

# Add thumbnails for chosen examples for quick visual inspection
for loc in picks:
    p = X2_img[loc]
    img = np.array(dataset[test_idx[loc]]["pil"])
    ab = AnnotationBbox(
        OffsetImage(img, zoom=0.25),
        p,
        frameon=True,
        bboxprops=dict(boxstyle="round,pad=0.2", lw=0.6),
    )
    ax.add_artist(ab)

ax.set_title("Image embeddings")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(frameon=False, fontsize=8, ncol=2)

# ----- Panel 2: text-only at chosen points (one template per pick) -----
ax = axes[1]
ax.scatter(X2_img[:, 0], X2_img[:, 1], s=6, alpha=0.1, label="_all_")
ax.scatter(X2_text[:, 0], X2_text[:, 1], s=80, marker="P", edgecolor="k", lw=0.6)
for (x, y), loc in zip(X2_text, picks):
    ax.text(x, y, class_names[int(y_test[loc])], fontsize=8, ha="center", va="bottom")
ax.set_title("Text embeddings")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

# ----- Panel 3: merged points (image + predicted-class text) -----
ax = axes[2]
ax.scatter(X2_img[:, 0], X2_img[:, 1], s=6, alpha=0.1, label="_all_")
ax.scatter(X2_merge[:, 0], X2_merge[:, 1], marker="*", s=120, edgecolor="k", lw=0.6)
for (x, y), loc in zip(X2_merge, picks):
    ax.text(x, y, class_names[int(y_test[loc])], fontsize=8, ha="center", va="bottom")
ax.set_title("Merged embeddings")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

plt.tight_layout()
plt.show()
```

Also, to view the results, we can display the images and show the predicted value vs. the ground truth.

```{code-cell} ipython3
images = [r["pil"] for r in dataset]  # same order as X/y

def show_predictions_linear_idx(clf, X_test, y_test, test_idx, images, class_names, n=12, seed=300):
    import random, numpy as np, matplotlib.pyplot as plt
    random.seed(0)
    y_pred = clf.predict(X_test)
    picks = random.sample(range(len(y_test)), min(n, len(y_test)))

    plt.figure(figsize=(10, 6))
    for i, k in enumerate(picks):
        orig_i = test_idx[k]                  # map back to original dataset index
        img = images[orig_i]                  # already a PIL.Image in RGB
        img_np = np.array(img)

        true = class_names[y_test[k]]
        pred = class_names[y_pred[k]]
        color = "green" if pred == true else "red"

        plt.subplot(3, 4, i + 1)
        plt.imshow(img_np)
        plt.title(f"Pred: {pred}\nTrue: {true}", color=color, fontsize=9)
        plt.axis("off")

    plt.suptitle("Random Test Samples with Linear Model Predictions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


show_predictions_linear_idx(clf, X_test, y_test, test_idx, images, class_names)
```

## 4. Image–text Retrieval

We can also use the fact that we have embeddings across two modalities (image and text) to conduct image–text retrieval:

> Given an image, find the best matching prompt.

or

> Given a prompt, find the best matching image.


Note that below you will see the `@` operator there performs matrix multiplication — specifically between the image feature matrix and the transpose of the text feature matrix.

For `img_feats` has shape `[N_img, D]` (one D-dimensional embedding per image)

and `txt_feats` has shape `[N_txt, D]` (one D-dimensional embedding per text prompt)

doing `img_feats @ txt_feats.T` yields a similarity matrix of shape `[N_img, N_txt]`.

```{code-cell} ipython3
# Given an image, find the best matching text prompts

import matplotlib.pyplot as plt
import numpy as np

if len(image_tensors) > 0:
    sims = (img_feats @ txt_feats.T).detach().cpu().numpy()  # [n_img, n_prompts]
    k = 3  # top-k prompts to show
    img_examples = [100, 200]  # two example images, No 101 and No 201

    for idx_img in img_examples:
        order = np.argsort(-sims[idx_img])[:k]
        print(f"\nImage {idx_img} ({image_labels[idx_img]}): top-{k} prompts")
        for j in order:
            print(f"  -> {prompts[j]}  score={sims[idx_img, j]:.3f}")

        # show the image
        plt.figure(figsize=(3, 3))
        plt.imshow(dataset[idx_img]["pil"])
        plt.title(f"Query Image {idx_img}: {image_labels[idx_img]}")
        plt.axis("off")
        plt.show()
```

```{code-cell} ipython3
# Given a text prompt, find the best matching images

import matplotlib.pyplot as plt
import numpy as np

if len(image_tensors) > 0:
    sims = (img_feats @ txt_feats.T).detach().cpu().numpy()
    k = 3  # top-k images to show
    text_examples = [0, 1]  # two example prompts

    for idx_text in text_examples:
        order = np.argsort(-sims[:, idx_text])[:k]
        print(f"\nPrompt: '{prompts[idx_text]}' → top-{k} matching images")
        for j in order:
            print(f"  -> Image {j:>3}: {image_labels[j]}  score={sims[j, idx_text]:.3f}")

        # show top-k retrieved images
        plt.figure(figsize=(9, 3))
        for i, j in enumerate(order):
            plt.subplot(1, k, i + 1)
            plt.imshow(dataset[j]["pil"])
            plt.title(f"{image_labels[j]}\n{round(sims[j, idx_text],3)}", fontsize=9)
            plt.axis("off")
        plt.suptitle(f"Top-{k} images for: '{prompts[idx_text]}'", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()
```

## 5. Vision Chat with a GPT Model

We now send an image URL together with a question using the **Responses API**.
This part requires a valid API key.

First let's load one image from `wikipedia`.

```python
from IPython.display import Image, display

url = "https://upload.wikimedia.org/wikipedia/commons/8/85/NiPc_MOF_wiki.png"
display(Image(url=url))

from openai import OpenAI

# set your OpenAI API key
OPENAI_API_KEY = "sk-.............."


client = OpenAI(api_key=OPENAI_API_KEY)


# Note: This key will expire after the lecture.
# To run this code later, generate a new API key at:
# https://platform.openai.com/api-keys


# Pick the first image in our small dataset
img_url = "https://upload.wikimedia.org/wikipedia/commons/8/85/NiPc_MOF_wiki.png"
user_question = "Describe this image in 1 sentence."

# Build the input as a list with one user item containing text and image
inp = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_question},
                {"type": "input_image", "image_url": img_url} #we put image here after the text
            ]
        }
]

try:
    resp = client.responses.create(model="gpt-4o-mini" , input=inp, temperature=0)
    print(resp.output_text)
except Exception as e:
    print("Vision chat failed:", e)


#Please run the code in Google Colab
```

Alternatively, you can also load image from your local folder.

![image.png](data:image/png;base64,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)

```python
import base64


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "NiPc_MOF_wiki.png"

# Getting the Base64 string
base64_image = encode_image(image_path)


response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what's reaction condition in this image?" },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

#print(response.output_text)
#Please  run this in Google Colab
```

Why this is useful for chemist?

This approach is efficient when dealing with a large body of scientific literature—say, 6,000 papers in a specific field—where manually reviewing each figure or diagram would be impossible.

You can guide vision languge models annotate and classify them according to their content (for example, identifying microscopy images, spectra, or molecular structures). Once labeled, the collection becomes a searchable dataset that allows you to quickly locate and analyze only the figures relevant to a specific research question or data-mining goal. Such automated image annotation and retrieval accelerate insight generation and support large-scale visual pattern discovery in scientific texts.


(See* Digital Discovery*, **2024**, 3, 491–501 for further discussion.)

```{code-cell} ipython3
from IPython.display import Image, display

url = "https://pubs.rsc.org/image/article/2024/dd/d3dd00239j/d3dd00239j-f1_hi-res.gif"
display(Image(url=url, width=800, height=800))
```

## 6. Glossary


```{glossary}
**CLIP**
    Contrastive Language–Image Pretraining. Learns aligned image and text spaces.

**Contrastive loss**
    Objective that raises similarity of matched pairs and lowers mismatched pairs.

**InfoNCE**
    A popular contrastive objective that treats each batch example as the only positive.

**Projection head**
    The final linear layer that maps encoder features into the shared space.

**Zero-shot**
    Using text prompts alone to classify images without training on task labels.

**Linear probe**
    A linear classifier trained on frozen embeddings.

**Cosine similarity**
    The dot product of two unit vectors.

**Temperature ($\tau$)**
    Scales logits in the contrastive loss. Lower means sharper distributions.

**Retrieval**
    Ranking images for a text query or texts for an image query based on similarity.
```