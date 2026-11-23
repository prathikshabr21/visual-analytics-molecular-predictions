
import random
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#get the atom features
def atom_features(atom):
    """Create numerical features for an atom."""
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]

#convert smiles to graph using RDKit
def smiles_to_graph(smiles, target):
    """Convert a SMILES string into a PyG graph."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None
    #We loop through every atom and extract its features
    node_features = [
        atom_features(atom)
        for atom in mol.GetAtoms()
    ]

    x = torch.tensor(node_features, dtype=torch.float)

    edges = []
    #edge index: This creates connections between atoms
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # PyG message passing uses directed edges.
        edges.append([i, j])
        edges.append([j, i])

    if edges:
        edge_index = torch.tensor(
            edges, dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )

    y = torch.tensor([target], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

#Convert SMILES to graph tensors
def main():
    df = pd.read_csv("data/delaney-processed.csv")

    graphs = []
    invalid_smiles = []

    for idx, row in df.iterrows():
        graph = smiles_to_graph(
            row["smiles"],
            row["measured log solubility in mols per litre"]
        )

        if graph is None:
            invalid_smiles.append(idx)
        else:
            graph.molecule_id = int(idx)
            graphs.append(graph)

    print("Total molecules:", len(df))
    print("Valid graphs:", len(graphs))
    print("Invalid SMILES:", len(invalid_smiles))

    indices = np.arange(len(graphs))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=SEED,
        shuffle=True
    )

    splits = {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "seed": SEED,
    }

    torch.save(splits, "data/esol_graphs.pt")

    print("Train graphs:", len(train_idx))
    print("Validation graphs:", len(val_idx))
    print("Test graphs:", len(test_idx))

    print("\nExample graph:")
    print(graphs[0])

    print("\nSaved to data/esol_graphs.pt")


if __name__ == "__main__":
    main()
