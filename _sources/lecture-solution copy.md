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

names = ["Cn1cnc2N(C)C(=O)N(C)C(=O)c12", "CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"]
rows = []
for nm in names:
    m = Chem.MolFromSmiles(nm)
    rows.append({
        "smiles": nm,
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
# Paste the SMILES you obtained from PubChem Draw structure
smi1 = "CC1=CC=CC=C1N=NC2=C(C=CC3=CC=CC=C32)O"  # for image 1
smi2 = "C1=CC(=C(C=C1I)C(=O)O)O"  # for image 2
smi3 = "C1=CC=C(C=C1)C2(C(=O)NC(=O)N2)C3=CC=CC=C3"  # for image 3

m1 = Chem.MolFromSmiles(smi1)
m2 = Chem.MolFromSmiles(smi2)
m3 = Chem.MolFromSmiles(smi3)

Draw.MolsToGridImage([m1, m2, m3], legends=["img1","img2","img3"], molsPerRow=3, subImgSize=(220,200), useSVG=True)
```
```{code-cell} ipython3
# Compute quick properties for the three molecules
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import pandas as pd

def props(m):
    return dict(
        MolWt=Descriptors.MolWt(m),
        LogP=Crippen.MolLogP(m),
        HBD=rdMolDescriptors.CalcNumHBD(m),
        HBA=rdMolDescriptors.CalcNumHBA(m),
        TPSA=rdMolDescriptors.CalcTPSA(m)
    )

df = pd.DataFrame([
    {"name":"img1","smiles":smi1, **props(m1)},
    {"name":"img2","smiles":smi2, **props(m2)},
    {"name":"img3","smiles":smi3, **props(m3)}
]).round(3)

df
```



---
## 10. Solutions

### Solution 9.1

```{code-cell} ipython3
targets = ["caffeine", "ibuprofen", "acetaminophen"]
for t in targets:
    pc_key = pubchem_by_name(t)["inchikey"]
    cir_key = cir_get(t, "stdinchikey")
    print(f"{t:14s} PubChem={pc_key}  CIR={cir_key}  match={pc_key==cir_key}")
```

### Solution 9.2

```{code-cell} ipython3
if Chem is not None:
    import pandas as pd
    rows = []
    names = ["caffeine", "acetaminophen", "ibuprofen"]
    for nm in names:
        info = pubchem_by_name(nm)
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
else:
    print("RDKit is not available here.")
```

### Solution 9.3

```{code-cell} ipython3
hits = pug_view_find_sections(2244, "GHS")
print(pug_view_text_from_hit(hits[0], max_lines=8))
```

### Solution 9.4

```{code-cell} ipython3
if Chem is not None:
    find = Chem.MolFromSmiles("Cl")
    put  = Chem.MolFromSmiles("F")
    mol  = Chem.MolFromSmiles("Clc1ccc(cc1)C(=O)O")
    out  = Chem.ReplaceSubstructs(mol, find, put, replaceAll=True)[0]
    print(Chem.MolToSmiles(out))
    Draw.MolToImage(out, size=(300, 220))
else:
    print("RDKit is not available here.")
```

### Solution 9.5

```{code-cell} ipython3
items = ["446157", "2244", "CCO", "482752", "c1ccccc1"]

for q in items:
    if q.isdigit():
        info = pubchem_by_cid(int(q))
        smi, cid = info["smiles"], info["cid"]
    elif Chem is not None and Chem.MolFromSmiles(q):
        smi, cid = q, None
    else:
        info = pubchem_by_name(q)
        smi, cid = info["smiles"], info["cid"]
    print(f"input={q} CID={cid} SMILES={smi}")
    if Chem is not None:
        m = Chem.MolFromSmiles(smi)
        display(Draw.MolToImage(m, size=(220, 180)))
```
