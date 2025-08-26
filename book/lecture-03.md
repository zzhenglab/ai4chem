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

# Lecture 3 - SMILES and RDKit

> Start with the SMILES language, practice tiny steps, then use RDKit to draw, edit, and analyze molecules. Finish with PubChem lookups.


```{contents}
:local:
:depth: 1
```

## Learning goals

- Read SMILES strings with confidence: atoms, bonds, branches, rings, aromaticity, charges, simple stereochemistry.
- Use RDKit to parse SMILES, draw structures, add hydrogens, and compute basic properties.
- Make small edits: replace atoms, neutralize groups, split salts, add a methyl group with a graph edit.
- Query PubChem after you can edit molecules locally, then round-trip to SMILES and files.

---


## 1. SMILES step by step

If you use Colab, run the install cell below first.

```{code-cell} ipython3
# Install only if needed
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors
except Exception:
    %pip install rdkit
```


### 1.1 Atoms

- Organic set without brackets: `B C N O P S F Cl Br I`.  
- Hydrogens are usually implicit.  
- Charges or unusual valence use brackets.




```{code-cell} ipython3
# Plain strings to focus on notation
ethanol = "CCO"         # C-C-O
mol_1 = Chem.MolFromSmiles(ethanol)
Draw.MolToImage(mol_1)
```


```{code-cell} ipython3
acetic = "CC(=O)O"      # C-C with a double bonded O and an OH
benzene = "c1ccccc1"    # aromatic ring

mol_2 = Chem.MolFromSmiles(acetic)
mol_3 = Chem.MolFromSmiles(benzene)

# Draw both molecules side by side
Draw.MolsToImage([mol_2, mol_3], molsPerRow=2, subImgSize=(100,100))
```


```{code-cell} ipython3
charged1 = "[NH4+]"     # ammonium
charged2 = "C(=O)[O-]"  # carboxylate
mol_c1 = Chem.MolFromSmiles(charged1)
mol_c2 = Chem.MolFromSmiles(charged2)

Draw.MolsToImage([mol_c1, mol_c2], molsPerRow=1, subImgSize=(100,100))

```

### 1.2 Bonds

- Single is implied.  
- `=` is double.  
- `#` is triple.

```{code-cell} ipython3
single = "CC"
double = "C=C"
triple = "C#N"

mols = [Chem.MolFromSmiles(bond) for bond in [single, double, triple]]

Draw.MolsToImage(mols, molsPerRow=1, subImgSize=(100,100))


```

### 1.3 Branches

- Parentheses create side branches.

```{code-cell} ipython3
isopropanol_a = "CC(O)CC"
isopropanol_b = "CC(C)OC"    # same structure, different order
print(isopropanol_a, isopropanol_b)

mol_a = Chem.MolFromSmiles(isopropanol_a)
mol_b = Chem.MolFromSmiles(isopropanol_b)

Draw.MolsToImage([mol_a, mol_b], molsPerRow=2, subImgSize=(100,100))


```

### 1.4 Rings

- Numbers open and close rings.  
- Same digit appears twice to close that ring.

```{code-cell} ipython3
cyclohexane = "C1CCCCC1"
benzene = "c1ccccc1"
naphthalene = "c1cccc2c1cccc2"

mols = [Chem.MolFromSmiles(ring) for ring in [cyclohexane, benzene, naphthalene]]

Draw.MolsToImage(mols, molsPerRow=3, subImgSize=(100,100))

```

### 1.5 Aromatic vs aliphatic

- Aromatic atoms are lower case.  
- Aliphatic are upper case.

```{code-cell} ipython3

mol_aromatic = Chem.MolFromSmiles("c1ccccc1")
mol_aliphatic = Chem.MolFromSmiles("C1CCCCC1")

Draw.MolsToImage([ mol_aromatic, mol_aliphatic], molsPerRow=3, subImgSize=(100,100))
```

```{note}
What the digits mean:
A number marks a ring connection. The same digit appears twice on the two atoms that are bonded to each other to close that ring.
For fused systems, you can reuse different digits to show where each ring closes.
```

```{code-cell} ipython3
ring1 = Chem.MolFromSmiles("C1CCC2CCCCC2C1")
ring2 = Chem.MolFromSmiles("C1CC(C)C1")
ring3 = Chem.MolFromSmiles("c1ccc(C2CCCCC2)cc1")
ring4 = Chem.MolFromSmiles("c1ccc(-c2ccc3ccccc3c2)cc1")
ring5 = Chem.MolFromSmiles("C1=CC2C=COC2=C1")
ring6 = Chem.MolFromSmiles("CC(C)c1cccc(C2CC2)c1")

Draw.MolsToImage([ring1,ring2, ring3, ring4, ring5, ring6], ubImgSize=(100,100))
```




### 1.6 Charges and salts

- Use `.` to separate parts in a salt.  
- Place charge in brackets on the atom.

```{code-cell} ipython3
salt = "C[NH+](C)C.[Cl-]"    # trimethylammonium chloride
Draw.MolToImage(Chem.MolFromSmiles(salt ), molsPerRow=1, subImgSize=(100,100))

```

### 1.7 Simple stereochemistry

- E and Z for alkenes use slashes.  
- `Cl/C=C/Cl` is E. `Cl/C=C\Cl` is Z.

```{code-cell} ipython3


mol_E = Chem.MolFromSmiles("Cl/C=C/Cl")
mol_Z = Chem.MolFromSmiles("Cl/C=C\\Cl")

Draw.MolsToImage([ mol_E, mol_Z], molsPerRow=3, subImgSize=(100,100))


```

```{admonition} Try
Search online SMILES for these and print them:
- isopropyl alcohol
- benzoate anion
- cyclopropane
- pyridine
```

---

## 2. RDKit quick start



```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors

smi = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin
mol = Chem.MolFromSmiles(smi)
Draw.MolToImage(mol, size=(350, 250))
```

```{code-cell} ipython3
# Add hydrogens for clarity
mol_H = Chem.AddHs(mol)
Draw.MolToImage(mol_H, size=(350, 250))
```

```{code-cell} ipython3
# Quick properties in one place
mw = Descriptors.MolWt(mol)
logp = Crippen.MolLogP(mol)
hbd = rdMolDescriptors.CalcNumHBD(mol)
hba = rdMolDescriptors.CalcNumHBA(mol)
tpsa = rdMolDescriptors.CalcTPSA(mol)
print("MolWt", round(mw,2), "LogP", round(logp,2), "HBD", hbd, "HBA", hba, "TPSA", round(tpsa,1))
```

```{note}
**MolWt** → The molecular weight (molar mass) of the compound, measured in grams per mole.

**LogP** → The logarithm of the partition coefficient (octanol/water); higher values mean more lipophilic (hydrophobic).

**HBD** (Hydrogen Bond Donors) → Atoms (often OH or NH groups) that can donate a hydrogen in hydrogen bonding.

**HBA** (Hydrogen Bond Acceptors) → Atoms (such as oxygen or nitrogen) that can accept a hydrogen bond.

**TPSA** (Topological Polar Surface Area) → A measure of the molecule’s polar area, correlated with solubility and permeability.
```

```{code-cell} ipython3
# Show atom numbers to plan edits
img = Draw.MolToImage(mol, size=(350, 250), includeAtomNumbers=True)
img
```

```{admonition} Practice
Change `smi` to caffeine or acetaminophen. Compare MolWt and TPSA.
```

---

## 3. Small edits in RDKit

We will avoid pattern languages here. We will use plain molecules to find and replace common pieces.

### 3.1 Replace atom type by matching a small molecule

Replace all chlorine atoms with fluorine in an aryl chloride.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw

qry = Chem.MolFromSmiles("Cl")     # what to find
rep = Chem.MolFromSmiles("F")      # what to place
mol = Chem.MolFromSmiles("Clc1ccc(cc1)C(=O)O")

out = Chem.ReplaceSubstructs(mol, qry, rep, replaceAll=True)[0]
Draw.MolToImage(out, size=(350, 250))
```

### 3.2 Neutralize a carboxylate

```{code-cell} ipython3
mol = Chem.MolFromSmiles("CC(=O)[O-]")
find = Chem.MolFromSmiles("[O-]")  # anionic oxygen as a molecule
put  = Chem.MolFromSmiles("O")
mol_neutral = Chem.ReplaceSubstructs(mol, find, put, replaceAll=True)[0]
Draw.MolToImage(mol_neutral, size=(320, 220))
```

### 3.3 Add a methyl group with a graph edit

```{code-cell} ipython3
mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
em = Chem.EditableMol(mol)

idx_C = em.AddAtom(Chem.Atom("C"))
idx_H1 = em.AddAtom(Chem.Atom("H"))
idx_H2 = em.AddAtom(Chem.Atom("H"))
idx_H3 = em.AddAtom(Chem.Atom("H"))

em.AddBond(2, idx_C, order=Chem.BondType.SINGLE)  # attach at atom index 2
em.AddBond(idx_C, idx_H1, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H2, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H3, order=Chem.BondType.SINGLE)

mol2 = em.GetMol()
Chem.SanitizeMol(mol2)
Draw.MolToImage(mol2, size=(350, 250), includeAtomNumbers=True)
```

```{admonition} Tip
After graph edits, call `Chem.SanitizeMol` to check valence and aromaticity.
```

---

## 4. PubChem after you can edit


```note
Goal: given a common name or a PubChem CID, get a SMILES string you can feed into RDKit.
```

### 4.1 Install and imports

```{code-cell} ipython3
# Install requests if you do not have it
try:
    import requests
except Exception:
    %pip -q install requests
    import requests

from urllib.parse import quote_plus  # for safe URL encoding
```

```[note]
Why encode? Names can contain spaces or symbols. `quote_plus("acetic acid")` -> `acetic+acid`.  
Unencoded spaces can cause HTTP 400 errors.
```

---

### 4.2 Resolve a **name** to a CID

```python
import requests
from urllib.parse import quote_plus

name = "acetaminophen"

# Step 1: resolve the name to one or more PubChem CIDs
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote_plus(name)}/cids/JSON"
print("URL:", url)  # helps debug if you get HTTP 400

r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()

cid_list = data.get("IdentifierList", {}).get("CID", [])
if not cid_list:
    raise ValueError(f"No CID found for {name}")

cid = cid_list[0]  # take the first hit
print("CID:", cid)
```

```{note}
CID is PubChem’s numeric identifier for a molecule. Taking the first hit is simple and works well for common drugs.
```

---

### 4.3 Get properties from the CID

#### 4.3.1 Get the **IUPAC name**

```python
fields = "IUPACName"
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{fields}/JSON"
print("URL:", url)

r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()

props = data["PropertyTable"]["Properties"][0]
print("IUPAC:", props.get("IUPACName"))
```

```{note}
The IUPAC name is the standardized systematic name for the compound, defined by the International Union of Pure and Applied Chemistry.
```

---

#### 4.3.2 Get the **Canonical SMILES**

```python
fields = "CanonicalSMILES"
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{fields}/JSON"
print("URL:", url)

r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()

props = data["PropertyTable"]["Properties"][0]
print("Canonical SMILES:", props.get("CanonicalSMILES"))
```

```{note}
Canonical SMILES is a normalized form of the molecule's SMILES string.  
It provides a unique representation but does not retain stereochemistry.
```

---

#### 4.3.3 Get the **Isomeric SMILES**

```python
fields = "IsomericSMILES"
url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{fields}/JSON"
print("URL:", url)

r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()

props = data["PropertyTable"]["Properties"][0]
print("Isomeric SMILES:", props.get("IsomericSMILES"))
```

```{note}
Isomeric SMILES includes stereochemistry and isotopic information if PubChem has it.  
This is useful for distinguishing molecules that have the same atoms but different spatial arrangements.
```

---

### 4.4 Make it safe: minimal error handling

```{code-cell} ipython3
def safe_get(url, timeout=30):
    """Return JSON for a URL or raise a friendly ValueError."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error from PubChem. URL was:\n{url}\nMessage: {e}") from e
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error while contacting PubChem: {e}") from e
```

```note
If students see HTTP 400, print the URL and check if the name was encoded.  
Timeouts keep notebooks from hanging.
```

---

### 4.5 Function: by **name**

```{code-cell} ipython3
def pubchem_smiles_by_name(name, isomeric=True):
    """
    Look up a compound by common name. Returns a dict with CID, SMILES, and IUPAC.
    
    name: string like "ibuprofen" or "acetic acid"
    isomeric: True -> IsomericSMILES; False -> CanonicalSMILES
    """
    fields = "CanonicalSMILES,IsomericSMILES,CID,IUPACName"
    encoded = quote_plus(str(name).strip())
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/{fields}/JSON"
    data = safe_get(url)
    try:
        p = data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError) as e:
        raise ValueError(f"No results found for name: {name}") from e

    smiles = p.get("IsomericSMILES") if isomeric else p.get("CanonicalSMILES")
    return {"name": name, "cid": p["CID"], "smiles": smiles, "iupac": p.get("IUPACName", "")}
```

**Example**

```{code-cell} ipython3
pubchem_smiles_by_name("ibuprofen")
```

```note
This picks the first PubChem hit. That is fine for common compounds in an intro class.  
For ambiguous names you could add a second step that lists all hits.
```

---

### 4.6 Function: by **CID**

```{code-cell} ipython3
def pubchem_smiles_by_cid(cid, isomeric=True):
    """
    Look up a compound by PubChem CID (integer). Returns a dict with SMILES and IUPAC.
    """
    fields = "CanonicalSMILES,IsomericSMILES,IUPACName"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{int(cid)}/property/{fields}/JSON"
    data = safe_get(url)
    try:
        p = data["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError) as e:
        raise ValueError(f"No results found for CID: {cid}") from e

    smiles = p.get("IsomericSMILES") if isomeric else p.get("CanonicalSMILES")
    return {"cid": int(cid), "smiles": smiles, "iupac": p.get("IUPACName", "")}
```

**Example**

```{code-cell} ipython3
pubchem_smiles_by_cid(2244)  # aspirin
```

```note
Using `int(cid)` helps catch strings like "2244 " early.
```

---

### 4.7 Use with RDKit (optional, short)

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw

res = pubchem_smiles_by_name("acetaminophen")
mol = Chem.MolFromSmiles(res["smiles"])
Draw.MolToImage(mol, size=(180, 180))
```

```note
If `MolFromSmiles` returns `None`, print the SMILES and check for copy errors.  
If depiction warns about kekulization on fused aromatics, pass `kekulize=False` to the drawing call.
```

---

### 4.8 All together: one helper that accepts either **name** or **CID**

```{code-cell} ipython3
def pubchem_smiles(query, isomeric=True):
    """
    Look up SMILES from PubChem by name or CID.
    - If query is an int (or digits), uses CID.
    - Otherwise, uses name.
    Returns a dict with keys: 'cid', 'smiles', and 'iupac'. Includes 'name' when searching by name.
    """
    # Decide route
    if isinstance(query, int) or (isinstance(query, str) and query.strip().isdigit()):
        cid = int(query)
        return pubchem_smiles_by_cid(cid, isomeric=isomeric)
    else:
        return pubchem_smiles_by_name(str(query), isomeric=isomeric)

# Examples
print(pubchem_smiles("ibuprofen"))  # by name
print(pubchem_smiles(2244))         # by CID (aspirin)
```

```note
Teach students to try `pubchem_smiles("acetic acid")` and `pubchem_smiles("sodium chloride")`.  
These show why URL encoding matters and how salts look in SMILES.
```

---

### 4.9 Quick fixes for common errors

- **HTTPError: 400 Bad Request**  
  Use `quote_plus(name)`. Print the URL to see what was sent.
- **No results**  
  Check spelling. Try a different name or use a CID.
- **Timeout**  
  Try again. Classroom Wi-Fi can be slow. You can increase `timeout=60` in `safe_get`.
- **Multiple matches**  
  In an intro class, use the first result. For an advanced class, add a route that lists CIDs and lets students pick.

```note
If you want to keep the very first lesson short, skip 4.8 and just teach 4.5 (by name) first.  
Add 4.6 (by CID) and 4.8 (combined) in a later exercise.
```


## 5. Save and export

```{code-cell} ipython3
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
print("canonical:", Chem.MolToSmiles(mol))
print("isomeric:", Chem.MolToSmiles(mol, isomericSmiles=True))
```

```{code-cell} ipython3
# SDF with 2D coordinates
m = Chem.AddHs(mol)
from rdkit.Chem import AllChem
AllChem.Compute2DCoords(m)
w = Chem.SDWriter("molecule.sdf")
w.write(m); w.close()
"Saved molecule.sdf"
```

```{code-cell} ipython3
# PNG depiction
img = Draw.MolToImage(mol, size=(400, 300))
img.save("molecule.png")
"Saved molecule.png"
```

---

## 6. Quick reference

```{admonition} SMILES
- Atoms: upper case aliphatic, lower case aromatic
- Bonds: implicit single, =, #
- Branches: parentheses
- Rings: digits to open and close
- Charges: bracket the atom, e.g., [O-], [NH4+]
- Salts: separate parts with a dot
- E or Z: use slashes around the double bond
```

```{admonition} RDKit
- Parse: `Chem.MolFromSmiles`
- Draw: `Draw.MolToImage(..., includeAtomNumbers=True)`
- Hydrogens: `Chem.AddHs`
- Properties: `Descriptors.MolWt`, `Crippen.MolLogP`, `CalcNumHBA/HBD`, `CalcTPSA`
- Replace piece with piece: `Chem.ReplaceSubstructs(mol, findMol, repMol)`
- Salt split: `Chem.GetMolFrags(..., asMols=True)`
- Graph edit: `Chem.EditableMol`
- Save: `Chem.MolToSmiles`, `SDWriter`, PNG via `MolToImage(...).save(...)`
```

---

## 7. Glossary

```{glossary}
SMILES
  Text line notation for molecules. Example: ethanol is CCO.

aromatic
  Conjugated ring system represented with lower case atom symbols in SMILES, for example c1ccccc1.

CID
  PubChem Compound ID for a unique compound record.

sanitize
  RDKit process that checks valence, aromaticity, and stereochemistry.

descriptor
  Computed molecular property such as molecular weight or LogP.

EditableMol
  RDKit object that exposes low level atom and bond editing.
```

---

## 8. In-class activity

Each task mirrors the examples above. Fill in the `...` lines. Work in pairs. Solutions are in Section 9.

### 8.1 Read a SMILES and inspect

Given `smi = "O=C(O)c1ccccc1Cl"`.  
a) Draw with atom numbers.  
b) Count number of rings.  
c) Print the list of bonds with begin and end atom indices and bond orders.

```python
from rdkit import Chem
from rdkit.Chem import Draw

smi = ...  # TO DO

mol = Chem.MolFromSmiles(smi)
display(Draw.MolToImage(mol, size=(350, 250), includeAtomNumbers=True))

num_rings = ...   # TO DO: Chem.GetSSSR(mol)
print("rings:", num_rings)

for b in mol.GetBonds():
    print("bond", b.GetIdx(), b.GetBeginAtomIdx(), "-", b.GetEndAtomIdx(), "order", int(b.GetBondTypeAsDouble()))
```

---

### 8.2 Make a small properties table

Use names `["caffeine", "acetaminophen", "ibuprofen"]`. For each, fetch SMILES from PubChem, then compute MolWt, LogP, HBD, HBA, and TPSA.

```python
import pandas as pd
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

names = ...  # TO DO
rows = []
for nm in names:
    info = ...  # TO DO: pubchem_smiles_by_name
    smi = info["smiles"]
    m = Chem.MolFromSmiles(smi)
    rows.append({
        "name": nm,
        "smiles": smi,
        "MolWt": ...,
        "LogP": ...,
        "HBD": ...,
        "HBA": ...,
        "TPSA": ...
    })

pd.DataFrame(rows)
```

---

### 8.3 Replace chlorine with fluorine

Replace Cl with F in `Clc1ccc(cc1)C(=O)O` and print the result SMILES.

```python
find = Chem.MolFromSmiles(... )  # TO DO: "Cl"
put  = Chem.MolFromSmiles(... )  # TO DO: "F"
mol  = Chem.MolFromSmiles("Clc1ccc(cc1)C(=O)O")
out  = Chem.ReplaceSubstructs(mol, find, put, replaceAll=True)[0]
print(Chem.MolToSmiles(out))
```

---

### 8.4 Add a methyl group with a graph edit

Add a methyl at atom index 2 of benzene.

```python
mol = Chem.MolFromSmiles("c1ccccc1")
em = Chem.EditableMol(mol)

idx_C = em.AddAtom(Chem.Atom("C"))
idx_H1 = em.AddAtom(Chem.Atom("H"))
idx_H2 = em.AddAtom(Chem.Atom("H"))
idx_H3 = em.AddAtom(Chem.Atom("H"))

em.AddBond(..., idx_C, order=Chem.BondType.SINGLE)  # TO DO: use 2
em.AddBond(idx_C, idx_H1, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H2, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H3, order=Chem.BondType.SINGLE)

mol2 = em.GetMol()
Chem.SanitizeMol(mol2)
Draw.MolToImage(mol2, size=(350, 250), includeAtomNumbers=True)
```

---

### 8.5 Split a salt and keep the largest fragment

```python
mix = Chem.MolFromSmiles("C[NH+](C)C.Cl-")
frags = Chem.GetMolFrags(mix, asMols=True, sanitizeFrags=True)
keep = max(frags, key=lambda m: m.GetNumAtoms())
print("fragments:", [Chem.MolToSmiles(f) for f in frags])
print("kept:", Chem.MolToSmiles(keep))
```

---

### 8.6 E and Z check

Draw both and print double bond stereo flags.

```python
s_e = ...  # TO DO: "Cl/C=C/Cl"
s_z = ...  # TO DO: "Cl/C=C\\Cl"

for s in [s_e, s_z]:
    m = Chem.MolFromSmiles(s)
    display(Draw.MolToImage(m, size=(300, 220)))
    flags = [b.GetStereo() for b in m.GetBonds() if b.GetBondType().name == "DOUBLE"]
    print(s, flags)
```

---

## 9. Solutions

Open after you try Section 8.

### Solution 8.1

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw

smi = "O=C(O)c1ccccc1Cl"

mol = Chem.MolFromSmiles(smi)
display(Draw.MolToImage(mol, size=(350, 250), includeAtomNumbers=True))

num_rings = Chem.GetSSSR(mol)
print("rings:", num_rings)

for b in mol.GetBonds():
    print("bond", b.GetIdx(), b.GetBeginAtomIdx(), "-", b.GetEndAtomIdx(), "order", int(b.GetBondTypeAsDouble()))
```

### Solution 8.2

```{code-cell} ipython3
import pandas as pd
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

names = ["caffeine", "acetaminophen", "ibuprofen"]
rows = []
for nm in names:
    info = pubchem_smiles_by_name(nm)
    smi = info["smiles"]
    m = Chem.MolFromSmiles(smi)
    rows.append({
        "name": nm,
        "smiles": smi,
        "MolWt": Descriptors.MolWt(m),
        "LogP": Crippen.MolLogP(m),
        "HBD": rdMolDescriptors.CalcNumHBD(m),
        "HBA": rdMolDescriptors.CalcNumHBA(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m)
    })

pd.DataFrame(rows)
```

### Solution 8.3

```{code-cell} ipython3
find = Chem.MolFromSmiles("Cl")
put  = Chem.MolFromSmiles("F")
mol  = Chem.MolFromSmiles("Clc1ccc(cc1)C(=O)O")
out  = Chem.ReplaceSubstructs(mol, find, put, replaceAll=True)[0]
print(Chem.MolToSmiles(out))
Draw.MolToImage(out, size=(350, 250))
```

### Solution 8.4

```{code-cell} ipython3
mol = Chem.MolFromSmiles("c1ccccc1")
em = Chem.EditableMol(mol)

idx_C = em.AddAtom(Chem.Atom("C"))
idx_H1 = em.AddAtom(Chem.Atom("H"))
idx_H2 = em.AddAtom(Chem.Atom("H"))
idx_H3 = em.AddAtom(Chem.Atom("H"))

em.AddBond(2, idx_C, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H1, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H2, order=Chem.BondType.SINGLE)
em.AddBond(idx_C, idx_H3, order=Chem.BondType.SINGLE)

mol2 = em.GetMol()
Chem.SanitizeMol(mol2)
Draw.MolToImage(mol2, size=(350, 250), includeAtomNumbers=True)
```

### Solution 8.5

```{code-cell} ipython3
mix = Chem.MolFromSmiles("C[NH+](C)C.Cl-")
frags = Chem.GetMolFrags(mix, asMols=True, sanitizeFrags=True)
keep = max(frags, key=lambda m: m.GetNumAtoms())
print("fragments:", [Chem.MolToSmiles(f) for f in frags])
print("kept:", Chem.MolToSmiles(keep))
```

### Solution 8.6

```{code-cell} ipython3
s_e = "Cl/C=C/Cl"
s_z = "Cl/C=C\\Cl"

for s in [s_e, s_z]:
    m = Chem.MolFromSmiles(s)
    display(Draw.MolToImage(m, size=(300, 220)))
    flags = [b.GetStereo() for b in m.GetBonds() if b.GetBondType().name == "DOUBLE"]
    print(s, flags)
```

---

## 10. Save your work

```{admonition} Save
- Keep a CSV of PubChem pulls and computed properties if you plan to model.  
- Save key molecules to SDF and PNG.  
- Store edited SMILES with `Chem.MolToSmiles` for reproducibility.
```
