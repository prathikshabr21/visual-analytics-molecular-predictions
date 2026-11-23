
import torch
import torch_geometric
import rdkit
import pandas as pd

print("PyTorch:", torch.__version__)
print("PyTorch Geometric:", torch_geometric.__version__)
print("RDKit:", rdkit.__version__)

df = pd.read_csv("data/delaney-processed.csv")

print("Dataset shape:", df.shape) #(1128, 10)
print("Columns:", df.columns.tolist()) #
['Compound ID', 'ESOL predicted log solubility in mols per litre', 'Minimum Degree', 'Molecular Weight', 'Number of H-Bond Donors', 'Number of Rings', 
 'Number of Rotatable Bonds', 'Polar Surface Area', 'measured log solubility in mols per litre', 'smiles']
print("Missing values:", df.isnull().sum().sum()) #None

print("Setup successful!")

