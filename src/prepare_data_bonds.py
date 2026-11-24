
from pathlib import Path

import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.data import Data


# ==========================================
# 1. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "delaney-processed.csv"
)

ORIGINAL_PATH = (
    PROJECT_ROOT / "data" / "esol_graphs.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "esol_graphs_bonds.pt"
)


# ==========================================
# 2. Bond categories
# ==========================================

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

NUM_BOND_FEATURES = 14


# ==========================================
# 3. One-hot encoding
# ==========================================

def one_hot_encode(value, categories):

    encoding = [
        int(value == category)
        for category in categories
    ]

    # Unknown or other category
    encoding.append(
        int(value not in categories)
    )

    return encoding


# ==========================================
# 4. Original six atomic features
# ==========================================

def atom_features(atom):

    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]


# ==========================================
# 5. Chemical bond features
# ==========================================

def bond_features(bond):

    features = one_hot_encode(
        bond.GetBondType(),
        BOND_TYPES
    )

    features.extend([
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ])

    features.extend(
        one_hot_encode(
            bond.GetStereo(),
            STEREO_TYPES
        )
    )

    return features


# ==========================================
# 6. Convert SMILES to molecular graph
# ==========================================

def smiles_to_graph(smiles, target):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    node_features = [
        atom_features(atom)
        for atom in mol.GetAtoms()
    ]

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    edges = []
    edge_features = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        features = bond_features(bond)

        # Forward directed edge
        edges.append([i, j])
        edge_features.append(features)

        # Reverse directed edge
        edges.append([j, i])
        edge_features.append(features)

    if edges:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_attr = torch.empty(
            (0, NUM_BOND_FEATURES),
            dtype=torch.float
        )

    y = torch.tensor(
        [target],
        dtype=torch.float
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )


# ==========================================
# 7. Generate bond-aware dataset
# ==========================================

def main():

    df = pd.read_csv(DATA_PATH)

    original_splits = torch.load(
        ORIGINAL_PATH,
        weights_only=False,
        map_location="cpu"
    )

    # Preserve original split metadata
    bond_splits = dict(original_splits)

    for split_name in ["train", "val", "test"]:

        bond_graphs = []

        for old_graph in original_splits[split_name]:

            molecule_id = int(old_graph.molecule_id)

            row = df.iloc[molecule_id]

            graph = smiles_to_graph(
                row["smiles"],
                row[
                    "measured log solubility in mols per litre"
                ]
            )

            if graph is None:

                raise ValueError(
                    f"Invalid molecule: {molecule_id}"
                )

            graph.molecule_id = molecule_id

            # Verify unchanged atomic features
            assert torch.equal(
                graph.x,
                old_graph.x
            ), f"Atom features changed: {molecule_id}"

            # Verify unchanged graph connectivity
            assert torch.equal(
                graph.edge_index,
                old_graph.edge_index
            ), f"Edges changed: {molecule_id}"

            # Verify unchanged target values
            assert torch.equal(
                graph.y,
                old_graph.y
            ), f"Target changed: {molecule_id}"

            # Verify bond feature dimensions
            assert graph.edge_attr.shape[1] == (
                NUM_BOND_FEATURES
            )

            # Verify one feature vector per edge
            assert graph.edge_attr.shape[0] == (
                graph.edge_index.shape[1]
            )

            # Verify numerical validity
            assert torch.isfinite(graph.x).all()
            assert torch.isfinite(graph.edge_attr).all()

            # Verify forward and reverse edge features
            assert torch.equal(
                graph.edge_attr[0::2],
                graph.edge_attr[1::2]
            )

            bond_graphs.append(graph)

        bond_splits[split_name] = bond_graphs


    # ======================================
    # 8. Verify identical dataset splits
    # ======================================

    for split_name in ["train", "val", "test"]:

        original_ids = [
            int(graph.molecule_id)
            for graph in original_splits[split_name]
        ]

        bond_ids = [
            int(graph.molecule_id)
            for graph in bond_splits[split_name]
        ]

        assert original_ids == bond_ids, (
            f"Split mismatch: {split_name}"
        )

    bond_splits["feature_set"] = "bonds"
    bond_splits["num_node_features"] = 6
    bond_splits["num_edge_features"] = NUM_BOND_FEATURES


    # ======================================
    # 9. Save the new dataset
    # ======================================

    torch.save(
        bond_splits,
        OUTPUT_PATH
    )


    # ======================================
    # 10. Display verification results
    # ======================================

    print("\n--- BOND-AWARE DATASET ---")

    print("Total molecules:", len(df))

    print(
        "Atom features:",
        bond_splits["train"][0].x.shape[1]
    )

    print(
        "Bond features:",
        bond_splits["train"][0].edge_attr.shape[1]
    )

    print(
        "Train graphs:",
        len(bond_splits["train"])
    )

    print(
        "Validation graphs:",
        len(bond_splits["val"])
    )

    print(
        "Test graphs:",
        len(bond_splits["test"])
    )

    print("\nOriginal example graph:")
    print(original_splits["train"][0])

    print("\nBond-aware example graph:")
    print(bond_splits["train"][0])

    print("\nExample bond feature vector:")
    print(
        bond_splits["train"][0].edge_attr[0]
        if bond_splits["train"][0].edge_attr.shape[0] > 0
        else "No bonds"
    )

    print("\nAll verification checks passed.")
    print("Original atomic features preserved.")
    print("Original graph connectivity preserved.")
    print("Train, validation and test splits preserved.")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
