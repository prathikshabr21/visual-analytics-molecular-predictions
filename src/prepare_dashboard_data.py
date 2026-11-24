
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen


# ==================================================
# 1. Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "dashboard_data"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Load original ESOL dataset
# ==================================================

df = pd.read_csv(
    DATA_DIR / "delaney-processed.csv"
)

df = df.reset_index().rename(
    columns={"index": "molecule_id"}
)

assert len(df) == 1128
assert df["molecule_id"].is_unique

print("\n--- ORIGINAL DATASET ---")
print(f"Total molecules: {len(df)}")
print(f"Original columns: {len(df.columns)}")


# ==================================================
# 3. Load saved graph dataset
# ==================================================

splits = torch.load(
    DATA_DIR / "esol_graphs.pt",
    weights_only=False,
    map_location="cpu",
)


# ==================================================
# 4. Recover actual dataset split membership
# ==================================================

split_records = []

split_names = {
    "train": "Training",
    "val": "Validation",
    "test": "Testing",
}

for split_key, split_label in split_names.items():

    for graph in splits[split_key]:

        split_records.append({
            "molecule_id": int(graph.molecule_id),
            "split": split_label,
            "graph_target": float(graph.y.item()),
            "Number of Atoms": int(graph.x.shape[0]),
            "Number of Bonds": int(
                graph.edge_index.shape[1] // 2
            ),
        })

split_df = pd.DataFrame(split_records)

assert len(split_df) == 1128
assert split_df["molecule_id"].is_unique

df = df.merge(
    split_df,
    on="molecule_id",
    how="left",
    validate="one_to_one",
)

assert df["split"].notna().all()

assert np.allclose(
    df["measured log solubility in mols per litre"],
    df["graph_target"],
    atol=1e-5,
    rtol=0,
)

df = df.drop(columns=["graph_target"])

expected_counts = {
    "Training": 902,
    "Validation": 113,
    "Testing": 113,
}

assert df["split"].value_counts().to_dict() == expected_counts

print("\n--- DATASET SPLITS ---")
print(df["split"].value_counts().to_string())


# ==================================================
# 5. Calculate additional molecular descriptors
# ==================================================

def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "LogP": Crippen.MolLogP(mol),

        "Number of H-Bond Acceptors":
            Lipinski.NumHAcceptors(mol),

        "Number of Aromatic Rings":
            Lipinski.NumAromaticRings(mol),

        "Heavy Atom Count":
            Lipinski.HeavyAtomCount(mol),

        "Fraction Csp3":
            Lipinski.FractionCSP3(mol),

        "Number of Heteroatoms":
            Lipinski.NumHeteroatoms(mol),

        "Molar Refractivity":
            Crippen.MolMR(mol),
    }


print("\nCalculating molecular descriptors...")

descriptor_records = [
    calculate_descriptors(smiles)
    for smiles in df["smiles"]
]

descriptors_df = pd.DataFrame(
    descriptor_records,
    index=df.index,
)

assert len(descriptors_df) == len(df)

assert np.isfinite(
    descriptors_df.to_numpy(dtype=float)
).all()

df = pd.concat(
    [df, descriptors_df],
    axis=1,
)


# ==================================================
# 6. Add readable solubility column
# ==================================================

df["Measured logS"] = df[
    "measured log solubility in mols per litre"
]


# ==================================================
# 7. Create dataset split summary
# ==================================================

split_summary = (
    df.groupby("split")
    .agg(
        molecules=("molecule_id", "count"),
        mean_logS=("Measured logS", "mean"),
        std_logS=("Measured logS", "std"),
        min_logS=("Measured logS", "min"),
        max_logS=("Measured logS", "max"),
    )
    .reindex(["Training", "Validation", "Testing"])
    .reset_index()
)

split_summary["percentage"] = (
    split_summary["molecules"] / len(df) * 100
)


# ==================================================
# 8. Save dashboard-ready datasets
# ==================================================

df.to_csv(
    OUTPUT_DIR / "all_molecules.csv",
    index=False,
)

split_summary.to_csv(
    OUTPUT_DIR / "dataset_split_summary.csv",
    index=False,
)


# ==================================================
# 9. Final verification
# ==================================================

print("\n--- DASHBOARD DATASET SUMMARY ---")

print(f"Total molecules: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Unique molecule IDs: {df['molecule_id'].nunique()}")
print(f"Missing values: {df.isna().sum().sum()}")

print("\n--- ADDITIONAL DESCRIPTORS ---")

for column in descriptors_df.columns:
    print(f"- {column}")

print("\n--- SPLIT STATISTICS ---")

print(
    split_summary.round(4).to_string(index=False)
)

print("\n--- GENERATED FILES ---")

print(OUTPUT_DIR / "all_molecules.csv")
print(OUTPUT_DIR / "dataset_split_summary.csv")

print("\nDashboard data preparation completed successfully.")
