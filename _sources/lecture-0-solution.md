## 11. In-class activity solution


### 11.1 Ethanol mass to moles
Compute the moles of ethanol C2H6O in 9.2 g using only arithmetic and variables.

```{code-cell} ipython3

# Atomic weights (g mol^-1)
C = 12.011
H = 1.008
O = 15.999

# Molar mass of ethanol: C2H6O
M_ethanol = 2*C + 6*H + 1*O   # TODO

# Mass to moles
mass_g = 9.2
moles = mass_g / M_ethanol       # TODO

print("M_ethanol =", M_ethanol, "g mol^-1")
print("moles in", mass_g, "g =", moles)
```
---

### 11.2 Classify pH values in a list
Given several pH readings "2.5", "7.0", "8.1", "6.9", "7.3", print a line for each saying acidic, basic, or neutral. Do not define a helper function.

```{code-cell} ipython3

pH_values = [2.5, 7.0, 8.1, 6.9, 7.3] #TODO

for pH in pH_values:     # TODO
    if pH < 7:               # TODO
        status = "acidic"
    elif pH > 7:             # TODO
        status = "basic"
    else:
        status = "neutral"
    print(pH, "->", status)
```

---

### 11.3 Molar mass from a counts dictionary
Compute the molar mass of glucose using a small dictionary of atomic weights and a counts dictionary. Do not reuse any earlier functions.

```{code-cell} ipython3

# Atomic weights
aw = {"H": 1.008, "C": 12.011, "O": 15.999}

# Counts for C6H12O6
counts = {"C": 6, "H": 12, "O": 6}

M_glucose = 0.0
for elem, n in counts.items():       # TODO
    M_glucose + aw[elem] * n       # TODO 

print("M_glucose =", M_glucose, "g mol^-1")
```


### 11.4 Read leading integer from a string, then convert C to K
A temperature string has digits followed by a letter, for example "25C" or "298K". Read the leading digits using a while-loop and convert Celsius to Kelvin. If the unit is K, leave it as is.

```{code-cell} ipython3
s = "25C"   # try "298K" too

# Parse leading integer value
i = 0
value = 0
while i < len(s) and s[i].isdigit():
    value = value * 10 + int(s[i])
    i += 1

unit = s[i:]  # the rest of the string, e.g. "C" or "K"

if unit == "C":                  # TODO
    temp_K = value + 273.15       # TODO
else:
    temp_K = value

print("Parsed:", value, unit)
print("Temperature in K:", temp_K)
```

---

### 11.5 Which sample has more moles
You are given a mixture as a list of (mass_g, name, counts) where counts is a dictionary of element -> count in the formula. Compute the moles for each, then print the name of the sample with the largest moles. Do not call any functions from above; write the few lines you need here.

```{code-cell} ipython3
aw = {"H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007}

mixture = [
    (2.00, "CO2", {"C": 1, "O": 2}),
    (3.00, "H2O", {"H": 2, "O": 1}),
    (4.00, "NH3", {"N": 1, "H": 3}),
]

max_moles = -1.0
winner = None

for mass_g, name, counts in mixture:         # TODO
    # Compute molar mass from counts
    M = 0.0
    for elem, n in counts.items():    # TODO
        M = M + aw[elem] * n   # TODO
    n_moles = mass_g / M        # TODO

    if n_moles > max_moles:          # TODO
        max_moles = n_moles
        winner = name

print("Largest moles:", winner, "with", max_moles, "mol")
```



































## 9. Solutions

Search for `# TO DO:` in each block above and compare to the full code here.



### Solution 8.1

```{code-cell} ipython3

import pandas as pd

path = "sample_beer_lambert.csv"
# TO DO: read the CSV
df = pd.read_csv(path)

print(df.head())
print(df.info())
print(df.describe())

# TO DO: boolean mask and count
mask = df["absorbance_A"] > 0
count_positive = mask.sum()
print("rows with condition:", count_positive)
```

---

### Solution 8.2

```{code-cell} ipython3
import matplotlib.pyplot as plt

xcol = "concentration_mol_L"
ycol = "absorbance_A"

# TO DO: style controls
point_size = 30
alpha_val  = 0.8
marker_sym = "o"

plt.figure(figsize=(6, 4))
plt.scatter(df[xcol], df[ycol], s=point_size, alpha=alpha_val, marker=marker_sym)
plt.xlabel(xcol)
plt.ylabel(ycol)
plt.title("Scatter: y vs x with style tweaks")
plt.grid(True)
```

---

### Solution 8.3

```{code-cell} ipython3
key_col = "concentration_mol_L"
val_col = "absorbance_A"

summary = (
    df.groupby(key_col)[val_col]
      .agg(["mean", "std", "count"])
      .reset_index()
)

x = summary[key_col].to_numpy()
y = summary["mean"].to_numpy()
# TO DO: standard deviation array
yerr = summary["std"].to_numpy()

plt.figure(figsize=(6, 4))
plt.errorbar(x, y, yerr=yerr, fmt="o-")
plt.xlabel(key_col)
plt.ylabel(f"mean {val_col}")
plt.title("Group means with error bars")
plt.grid(True)
```

---

### Solution 8.4

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use provided file if present
path = "https://raw.githubusercontent.com/zzhenglab/ai4chem/main/book/_data/organic_synthesis_yields.csv" 

# TO DO: read CSV
df_y = pd.read_csv(path)
df_y.head()
```

```{code-cell} ipython3
# TO DO: scatter temperature vs yield
plt.figure(figsize=(6, 4))
plt.scatter(df_y["temperature_C"], df_y["yield_percent"], alpha=0.7)
plt.xlabel("temperature, C")
plt.ylabel("yield, %")
plt.title("Yield vs temperature")
plt.grid(True)
```

```{code-cell} ipython3
# TO DO: color by time, choose a colormap
plt.figure(figsize=(6, 4))
plt.scatter(df_y["temperature_C"], df_y["yield_percent"],
            c=df_y["time_min"], cmap="viridis", alpha=0.8)
plt.colorbar(label="time, min")
plt.xlabel("temperature, C")
plt.ylabel("yield, %")
plt.title("Yield vs temperature colored by time")
plt.grid(True)
```

```{code-cell} ipython3
# TO DO: pivot and heatmap
pivot = df_y.pivot_table(index="temperature_C", columns="time_min",
                         values="yield_percent", aggfunc="mean")
plt.figure(figsize=(7, 5))
plt.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
plt.colorbar(label="yield, %")
plt.yticks(range(pivot.shape[0]), pivot.index)
plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45)
plt.xlabel("time, min")
plt.ylabel("temperature, C")
plt.title("Mean yield heatmap")
plt.grid(False)
```

```{code-cell} ipython3
# TO DO: histogram with chosen bins
plt.figure(figsize=(6, 4))
plt.hist(df_y["yield_percent"], bins=25, alpha=0.9)
plt.xlabel("yield, %")
plt.ylabel("count")
plt.title("Yield distribution")
plt.grid(True)
```

---

### Solution 8.5

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TO DO: define bins and labels
bins = [40, 60, 80, 100, 120]
labels = ["40-60", "60-80", "80-100", "100-120"]

df_y = df_y.copy()
df_y["temp_bin"] = pd.cut(df_y["temperature_C"], bins=bins, labels=labels, include_lowest=True)

# TO DO: build groups and draw plot
groups = [grp["yield_percent"].to_numpy() for _, grp in df_y.groupby("temp_bin")]

plt.figure(figsize=(6, 4))
plt.violinplot(groups, showmeans=True)
plt.xticks(range(1, len(labels) + 1), labels)
plt.ylabel("yield, %")
plt.title("Yield by temperature bin")
plt.grid(True)
```


In case you are curious how `"organic_synthesis_yields.csv"` is prepared (it's synthesized dataset of synthesis!), here is the code:
```python

rng = np.random.default_rng(42)
n = 180
temperature_C = rng.choice(np.arange(40, 121, 5), size=n, replace=True, p=np.linspace(2, 1, 17) / np.linspace(2, 1, 17).sum())
time_min = rng.choice(np.arange(10, 241, 10), size=n, replace=True, p=np.linspace(2, 1, 24) / np.linspace(2, 1, 24).sum())
temp_scale = (temperature_C - 30) / 60.0
time_scale = time_min / 180.0
base = 100 * (1 - np.exp(-temp_scale)) * (1 - np.exp(-time_scale))
noise_normal = rng.normal(0, 7, size=n)
noise_neg = -rng.gamma(shape=1.2, scale=4, size=n)
yield_percent = np.clip(np.round(base + noise_normal + noise_neg, 1), 3, 96)
df_new = pd.DataFrame({"reaction_id": np.arange(1, n + 1),
                       "temperature_C": temperature_C,
                       "time_min": time_min,
                       "yield_percent": yield_percent}).sample(frac=1, random_state=123).reset_index(drop=True)
df_new.to_csv("organic_ssynthesis_yields.csv", index=False)
```
