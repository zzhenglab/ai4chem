---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# Lecture 01 — Test: Machine Learning in Chemistry

```{contents}
:depth: 2
```

Welcome to our test lecture showcasing **MyST Markdown** features used in *Jupyter Book*. The scientific theme is *machine learning (ML) in chemistry*, but the real goal is to demonstrate every major styling element you might want in your own notes.

## 1. Learning Goals


123


## 2. Background: Why ML?

> *“Machine learning is rapidly becoming the third pillar of chemical discovery, alongside theory and experiment.”*\
> — *Someone on the Internet*

### 2.1 A quick equation

\(\hat y = f(\mathbf x;\,\theta)\)

where \$\theta\$ are the trainable parameters of our model.

### 2.2 A mini table

| Data modality | Common ML task | Example               |
| ------------- | -------------- | --------------------- |
| Spectra       | Regression     | Predict concentration |
| SMILES string | Classification | Toxic vs. non‑toxic   |
| MD trajectory | Clustering     | Conformational states |

## 3. Hands‑on: Linear Regression

### 3.1 Code cell (hidden *input*)

```{code-cell}
:tags: [hide-input]
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1.0],[2.0],[3.0],[4.0]])
y = np.array([2.1, 4.1, 6.2, 8.0])
model = LinearRegression().fit(X, y)
print(f'Slope = {model.coef_[0]:.2f}')
print(f'Intercept = {model.intercept_:.2f}')
```

### 3.2 Code cell with a plot (visible)

```{code-cell}
import matplotlib.pyplot as plt
plt.scatter(X, y, label='data')
plt.plot(X, model.predict(X), label='fit')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
```

### 3.3 Exercise

1234


## 4. Sidebar notes

\:::{sidebar} Key terms **Descriptor** – a numerical representation of molecular structure.

**Overfitting** – memorizing noise instead of learning signal. :::

## 5. Embedding an image


## 6. Hidden *output* example

```{code-cell}
:tags: [hide-output]
# This cell's printout will be hidden in the final book
print('Training complete! Accuracy = 0.95')
```

## 7. Conclusion

1111

---

*End of sample file.*

