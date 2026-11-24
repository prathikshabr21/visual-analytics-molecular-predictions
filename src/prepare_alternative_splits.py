from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import torch
import rdkit

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "experiments" / "split_comparison"

OUTPUT.mkdir(parents=True, exist_ok=True)

RDLogger.DisableLog("rdApp.warning")

PARTITIONS = ("train", "val", "test")

df = pd.read_csv(DATA / "delaney-processed.csv")
ids = set(range(len(df)))

print("RDKit version:", rdkit.__version__)
print("Total molecules:", len(df))


# 1. Load existing graphs

def load_graphs(filename):

    dataset = torch.load(
        DATA / filename,
        weights_only=False,
        map_location="cpu"
    )

    graphs = {
        int(g.molecule_id): g
        for p in PARTITIONS
        for g in dataset[p]
    }

    assert set(graphs) == ids
    assert sum(len(dataset[p]) for p in PARTITIONS) == len(df)

    return dataset, graphs


original, atom_graphs = load_graphs("esol_graphs.pt")
bonds, bond_graphs = load_graphs("esol_graphs_bonds.pt")

for i in ids:

    assert torch.equal(atom_graphs[i].x, bond_graphs[i].x)
    assert torch.equal(
        atom_graphs[i].edge_index,
        bond_graphs[i].edge_index
    )
    assert torch.equal(atom_graphs[i].y, bond_graphs[i].y)

    assert np.isclose(
        atom_graphs[i].y.item(),
        df.iloc[i]["measured log solubility in mols per litre"]
    )

print("Existing graphs verified.")


# 2. Preserve original split

original_ids = {
    p: [int(g.molecule_id) for g in original[p]]
    for p in PARTITIONS
}

for p in PARTITIONS:

    assert original_ids[p] == [
        int(g.molecule_id) for g in bonds[p]
    ]


# 3. Alternative random split

train, temp = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=123
)

val, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=123
)

random_ids = {
    "train": train.tolist(),
    "val": val.tolist(),
    "test": test.tolist()
}

print("Alternative random split generated.")


# 4. Corrected scaffold extraction

def extract_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if mol.GetRingInfo().NumRings() == 0:
        return ""

    mol_copy = Chem.Mol(mol)

    Chem.Kekulize(
        mol_copy,
        clearAromaticFlags=True
    )

    scaffold = MurckoScaffold._pyGetScaffoldForMol(
        mol_copy
    )

    Chem.SanitizeMol(scaffold)

    assert scaffold.GetRingInfo().NumRings() > 0

    return Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=False
    )


# 5. Extract scaffolds for all molecules

groups = defaultdict(list)
scaffolds = {}
failed = []

for i, smiles in enumerate(df["smiles"]):

    try:

        scaffold = extract_scaffold(smiles)

        scaffolds[i] = scaffold
        groups[scaffold].append(i)

    except Exception as e:

        failed.append(i)
        print(f"Failed molecule {i}: {e}")

print("\n--- SCAFFOLD VERIFICATION ---")
print("Successful:", len(scaffolds))
print("Failed:", len(failed))
print("Unique scaffold groups:", len(groups))
print("Acyclic molecules:", len(groups.get("", [])))

assert not failed, f"Failed molecule IDs: {failed}"
assert len(scaffolds) == len(df)


# 6. Create scaffold split

ordered_groups = list(groups.values())

random.Random(42).shuffle(ordered_groups)

ordered_groups.sort(key=len, reverse=True)

scaffold_ids = {p: [] for p in PARTITIONS}

targets = {
    p: len(original_ids[p])
    for p in PARTITIONS
}

for group in ordered_groups:

    partition = max(
        PARTITIONS,
        key=lambda p: targets[p] - len(scaffold_ids[p])
    )

    scaffold_ids[partition].extend(group)

print("Scaffold split generated.")


# 7. Verify and save all splits

all_splits = {
    "random42": original_ids,
    "random123": random_ids,
    "scaffold42": scaffold_ids
}

membership = []
summary = []

for split_name, split in all_splits.items():

    train_set = set(split["train"])
    val_set = set(split["val"])
    test_set = set(split["test"])

    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set

    assert train_set | val_set | test_set == ids

    assert sum(len(split[p]) for p in PARTITIONS) == len(df)

    assert all(len(split[p]) > 0 for p in PARTITIONS)

    if split_name == "scaffold42":

        a = {scaffolds[i] for i in split["train"]}
        b = {scaffolds[i] for i in split["val"]}
        c = {scaffolds[i] for i in split["test"]}

        assert not a & b
        assert not a & c
        assert not b & c

        print("No scaffold overlap detected.")

    print(f"Integrity checks passed: {split_name}")

    for p in PARTITIONS:

        partition_ids = split[p]

        values = [
            atom_graphs[i].y.item()
            for i in partition_ids
        ]

        summary.append({
            "split_id": split_name,
            "partition": p,
            "molecules": len(partition_ids),
            "percentage": len(partition_ids) / len(df) * 100,
            "mean_logS": np.mean(values),
            "std_logS": np.std(values),
            "min_logS": np.min(values),
            "max_logS": np.max(values),
            "unique_scaffolds": len({
                scaffolds[i] for i in partition_ids
            })
        })

        for i in partition_ids:

            membership.append({
                "split_id": split_name,
                "partition": p,
                "molecule_id": i,
                "scaffold": scaffolds[i]
            })

    if split_name == "random42":
        continue

    for feature_set, graphs in {
        "original": atom_graphs,
        "bonds": bond_graphs
    }.items():

        payload = {
            p: [graphs[i] for i in split[p]]
            for p in PARTITIONS
        }

        for p in PARTITIONS:

            payload[f"{p}_indices"] = np.asarray(split[p])

        payload["split_id"] = split_name
        payload["feature_set"] = feature_set
        payload["seed"] = 123 if split_name == "random123" else 42
        payload["rdkit_version"] = rdkit.__version__

        filename = f"esol_{split_name}_{feature_set}.pt"

        torch.save(payload, DATA / filename)

        print("Saved:", filename)


# 8. Save experiment documentation

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUTPUT / "split_summary.csv",
    index=False
)

pd.DataFrame(membership).to_csv(
    OUTPUT / "split_membership.csv",
    index=False
)

pd.DataFrame([
    {"scaffold": s, "molecules": len(g)}
    for s, g in groups.items()
]).sort_values(
    "molecules",
    ascending=False
).to_csv(
    OUTPUT / "scaffold_groups.csv",
    index=False
)

pd.DataFrame([{
    "rdkit_version": rdkit.__version__,
    "scaffold_method": "_pyGetScaffoldForMol",
    "kekulize_before_extraction": True,
    "clear_aromatic_flags": True,
    "include_chirality": False,
    "acyclic_grouping": "single_empty_scaffold",
    "scaffold_split_seed": 42,
    "alternative_random_seed": 123
}]).to_csv(
    OUTPUT / "split_configuration.csv",
    index=False
)

print("\n--- DATASET SPLIT COMPARISON ---")
print(summary_df.round(4).to_string(index=False))

print("\nAll scaffold extraction checks passed.")
print("All split integrity checks passed.")
print("Original datasets remain unchanged.")
print("Alternative datasets saved successfully.")