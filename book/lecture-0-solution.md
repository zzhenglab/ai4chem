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
