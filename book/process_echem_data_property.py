# pip install rdkit-pypi pandas openpyxl
import math, hashlib, random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

# ========= config =========
INPUT_PATH = r"C:\Users\52377\OneDrive - Washington University in St. Louis\CHEM 508\ai4chem\book\_data"
INPUT_FILE = "C_H_oxidation_ dataset.xlsx"    # change if your Excel file has a different name
ASSUMED_MP_C = 150.0         # fallback melting point for toy solubility (°C)
NOISE_STD_PKA = 0.25
NOISE_STD_TOX = 0.12
NOISE_STD_LOGS = 0.30
DETERMINISTIC_NOISE = True
# ==========================

def _seeded_gauss(key: str, std: float) -> float:
    if std <= 0:
        return 0.0
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = random.Random(seed)
    return rng.gauss(0, std)

# SMARTS patterns reused for toy features
SMARTS = {
    "carboxylic_acid": Chem.MolFromSmarts("C(=O)[O;H1]"),
    "phenol": Chem.MolFromSmarts("c[O;H1]"),
    "sulfonamide": Chem.MolFromSmarts("S(=O)(=O)N"),
    "imide": Chem.MolFromSmarts("C(=O)NC(=O)"),
    "alkyl_amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
    "tertiary_amine": Chem.MolFromSmarts("[NX3;H0;!$(NC=O)]"),
    "aniline": Chem.MolFromSmarts("cN"),
    "imidazole_like": Chem.MolFromSmarts("n1cc[nH]c1"),
    "nitro": Chem.MolFromSmarts("[N+](=O)[O-]"),
    "anilide": Chem.MolFromSmarts("cNC(=O)"),
    "michael_acceptor": Chem.MolFromSmarts("C=CC=O"),
    "epoxide": Chem.MolFromSmarts("C1OC1"),
    "azide": Chem.MolFromSmarts("N=[N+]=N"),
}
def _count(mol, key):
    patt = SMARTS[key]
    return len(mol.GetSubstructMatches(patt)) if patt is not None else 0

# ========== toy properties, matching earlier logic but collapsed to 1 col each ==========

def toy_logS_from_smiles(smiles: str,
                         mp_celsius: float = ASSUMED_MP_C,
                         noise_std: float = NOISE_STD_LOGS,
                         deterministic: bool = DETERMINISTIC_NOISE) -> float:
    """
    General Solubility Equation at 25 C
    log10(S [mol/L]) = 0.5 - 0.01*(MP - 25) - logP + noise
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    logp = Crippen.MolLogP(mol)
    logS = 0.5 - 0.01 * (mp_celsius - 25.0) - logp
    logS += _seeded_gauss(smiles + "|logs", noise_std) if deterministic else random.gauss(0, noise_std)
    return float(logS)

def toy_pKa_single(smiles: str,
                   noise_std: float = NOISE_STD_PKA,
                   deterministic: bool = DETERMINISTIC_NOISE) -> float:
    """
    Produce a single pKa estimate from the same toy features as before.
    We compute an acidic and a basic estimate, then return the one closer to physiological range.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan

    logP = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)

    # Acidic estimate
    acids = (2.5 * _count(mol, "carboxylic_acid") +
             1.8 * _count(mol, "imide") +
             1.2 * _count(mol, "sulfonamide") +
             0.8 * _count(mol, "phenol"))
    pKa_acid = 10.0 - 2.2 * acids + 0.03 * logP - 0.005 * tpsa
    pKa_acid = float(np.clip(pKa_acid, -2.0, 15.0))

    # Basic estimate
    bases = (2.5 * _count(mol, "alkyl_amine") +
             3.0 * _count(mol, "tertiary_amine") +
             1.0 * _count(mol, "imidazole_like") -
             1.5 * _count(mol, "aniline"))
    pKa_base = 6.0 + 1.8 * bases + 0.05 * hba - 0.15 * rings + 0.02 * logP
    pKa_base = float(np.clip(pKa_base, -2.0, 15.0))

    # choose the pKa closer to 7 as a single representative
    pick = pKa_acid if abs(pKa_acid - 7.0) < abs(pKa_base - 7.0) else pKa_base
    pick += _seeded_gauss(smiles + "|pka", noise_std) if deterministic else random.gauss(0, noise_std)
    return round(float(pick), 2)

def toy_toxicity_label(smiles: str,
                       noise_std: float = NOISE_STD_TOX,
                       deterministic: bool = DETERMINISTIC_NOISE) -> str:
    """
    Toy toxicity classifier label using alerts and physchem features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    mw = Descriptors.MolWt(mol)
    logP = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)

    alert_sum = (_count(mol, "nitro") + _count(mol, "anilide") +
                 _count(mol, "michael_acceptor") + _count(mol, "epoxide") +
                 _count(mol, "azide"))

    z = (0.35 * logP + 0.02 * (mw / 100.0) + 0.15 * rings +
         0.10 * hba + 0.10 * hbd - 0.003 * tpsa + 0.9 * alert_sum) - 1.0
    z += _seeded_gauss(smiles + "|tox", noise_std) if deterministic else random.gauss(0, noise_std)
    prob = 1.0 / (1.0 + math.exp(-z))
    return "toxic" if prob >= 0.5 else "non_toxic"

def process_file(input_folder: Path, input_filename: str):
    in_path = input_folder / input_filename
    if not in_path.exists():
        raise FileNotFoundError(f"Cannot find {in_path}")

    df = pd.read_excel(in_path)
    if "SMILES" not in df.columns or "Reactivity" not in df.columns:
        raise ValueError("Input must have columns 'SMILES' and 'Reactivity'")

    # compute once per SMILES into a small DataFrame with exactly 3 columns
    props = df["SMILES"].astype(str).apply(
        lambda smi: pd.Series({
            "Solubility_logS_25C": toy_logS_from_smiles(smi),
            "pKa_est": toy_pKa_single(smi),
            "Tox_Label": toy_toxicity_label(smi),
        })
    )

    # insert three columns right after SMILES and before Reactivity
    smiles_idx = df.columns.get_loc("SMILES")
    # keep order Solubility_logS_25C, pKa_est, Tox_Label
    df.insert(smiles_idx + 1, "Solubility_logS_25C", props["Solubility_logS_25C"])
    df.insert(smiles_idx + 2, "pKa_est", props["pKa_est"])
    df.insert(smiles_idx + 3, "Tox_Label", props["Tox_Label"])

    # write CSV next to the Excel
    out_csv = in_path.with_suffix("").parent / (in_path.stem + "_with_props.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    process_file(Path(INPUT_PATH), INPUT_FILE)
