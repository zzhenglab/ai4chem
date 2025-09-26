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








#Lecture 10


### Solution Q1
```{code-cell} ipython3
# Q4 solution
df_reg = df[["SMILES","Melting Point"]].dropna().copy()
df_reg.to_csv("mp_data.csv", index=False)

!chemprop train \
  --data-path mp_data.csv \
  -t regression \
  -s SMILES \
  --target-columns "Melting Point" \
  -o mp_model_q4 \
  --num-replicates 1 \
  --epochs 20 \
  --metrics mae rmse r2 \
  --tracking-metric r2

pd.DataFrame({"SMILES": ["CCO","c1ccccc1","CC(=O)O","CCN(CC)CC","O=C(O)C(O)C"]}).to_csv("q4_smiles.csv", index=False)

!chemprop predict \
  --test-path q4_smiles.csv \
  --model-paths mp_model_q4/replicate_0/model_0/best.pt \
  --preds-path q4_preds.csv

pd.read_csv("q4_preds.csv")
```

### Solution Q2
```{code-cell} ipython3
# Q5 solution
df = pd.read_csv(url)
df = df[["SMILES","Toxicity"]].dropna().copy()
df["Toxicity_bin"] = df["Toxicity"].str.lower().map({"toxic":1, "non_toxic":0}).astype(int)
df[["SMILES","Toxicity_bin"]].to_csv("tox_data.csv", index=False)

!chemprop train \
  --data-path tox_data.csv \
  -t classification \
  -s SMILES \
  --target-columns Toxicity_bin \
  -o tox_model \
  --num-replicates 1 \
  --epochs 20 \
  --class-balance \
  --metrics roc prc accuracy \
  --tracking-metric roc

pd.DataFrame({"SMILES": ["CCO","c1ccccc1","O=[N+](=O)[O-]","ClCCl","CC(=O)Cl"]}).to_csv("q5_smiles.csv", index=False)

!chemprop predict \
  --test-path q5_smiles.csv \
  --model-paths tox_model/replicate_0/model_0/best.pt \
  --preds-path q5_preds.csv

pd.read_csv("q5_preds.csv")
```
