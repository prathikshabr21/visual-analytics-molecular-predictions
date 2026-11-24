
from pathlib import Path

import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.data import Data


# ==========================================
# 1. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "delaney-processed.csv"

ORIGINAL_PATH = PROJECT_ROOT / "data" / "esol_graphs.pt"

OUTPUT_PATH = PROJECT_ROOT / "data" / "esol_graphs_enhanced.pt"


# ==========================================
# 2. Categorical chemical properties
# ==========================================

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

CHIRALITY_TYPES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]


# ==========================================
# 3. One-hot encoding
# ==========================================

def one_hot_encode(value, categories):

    encoding = [
        int(value == category)
        for category in categories
    ]

    # Additional category for unknown values
    encoding.append(
        int(value not in categories)
    )

    return encoding


# ==========================================
# 4. Enhanced atom features
# ==========================================

def atom_features(atom):

    # Original six features
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]

    # Hybridization: seven dimensions
    features.extend(
        one_hot_encode(
            atom.GetHybridization(),
            HYBRIDIZATION_TYPES
        )
    )

    # Chirality: four dimensions
    features.extend(
        one_hot_encode(
            atom.GetChiralTag(),
            CHIRALITY_TYPES
        )
    )

    # Additional numerical features
    features.extend([
        atom.GetTotalValence(),
        atom.GetMass() / 100.0,
        atom.GetNumRadicalElectrons(),
    ])

    return features


# ==========================================
# 5. Convert SMILES into enhanced graphs
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

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append([i, j])
        edges.append([j, i])

    if edges:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    y = torch.tensor(
        [target],
        dtype=torch.float
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=y
    )


# ==========================================
# 6. Create enhanced dataset
# ==========================================

def main():

    df = pd.read_csv(DATA_PATH)

    original_splits = torch.load(
        ORIGINAL_PATH,
        weights_only=False,
        map_location="cpu"
    )

    # Preserve original split metadata
    enhanced_splits = dict(original_splits)

    # Reconstruct graphs using the original
    # molecule IDs and split membership
    for split_name in ["train", "val", "test"]:

        enhanced_graphs = []

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

            # Verify that graph structure is unchanged
            assert torch.equal(
                graph.edge_index,
                old_graph.edge_index
            ), f"Edges changed for molecule {molecule_id}"

            # Verify that targets are unchanged
            assert torch.equal(
                graph.y,
                old_graph.y
            ), f"Target changed for molecule {molecule_id}"

            # Verify that original features are unchanged
            assert torch.equal(
                graph.x[:, :6],
                old_graph.x
            ), f"Original features changed for {molecule_id}"

            # Verify enhanced feature dimensions
            assert graph.x.shape[1] == 20

            assert torch.isfinite(graph.x).all()

            enhanced_graphs.append(graph)

        enhanced_splits[split_name] = enhanced_graphs

    # ======================================
    # 7. Verify identical dataset splits
    # ======================================

    for split_name in ["train", "val", "test"]:

        original_ids = [
            int(graph.molecule_id)
            for graph in original_splits[split_name]
        ]

        enhanced_ids = [
            int(graph.molecule_id)
            for graph in enhanced_splits[split_name]
        ]

        assert original_ids == enhanced_ids, (
            f"Split mismatch: {split_name}"
        )

    enhanced_splits["feature_set"] = "enhanced"
    enhanced_splits["num_node_features"] = 20

    # ======================================
    # 8. Save enhanced dataset
    # ======================================

    torch.save(
        enhanced_splits,
        OUTPUT_PATH
    )

    print("\n--- ENHANCED DATASET ---")

    print("Total molecules:", len(df))

    print(
        "Original atom features:",
        original_splits["train"][0].x.shape[1]
    )

    print(
        "Enhanced atom features:",
        enhanced_splits["train"][0].x.shape[1]
    )

    print(
        "Train graphs:",
        len(enhanced_splits["train"])
    )

    print(
        "Validation graphs:",
        len(enhanced_splits["val"])
    )

    print(
        "Test graphs:",
        len(enhanced_splits["test"])
    )

    print("\nOriginal example graph:")
    print(original_splits["train"][0])

    print("\nEnhanced example graph:")
    print(enhanced_splits["train"][0])

    print("\nAll verification checks passed.")
    print("Original dataset preserved.")
    print("Train, validation and test splits preserved.")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
