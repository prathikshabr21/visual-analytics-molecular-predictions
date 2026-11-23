
from pathlib import Path
import torch

# 1. Locate the project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2. Define input and output file paths
INPUT_FILE = PROJECT_ROOT / "data" / "esol_graphs.pt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "esol_graphs_readable.txt"

# 3. Load the saved molecular graphs
# Only use weights_only=False for files you trust.
data = torch.load(INPUT_FILE, weights_only=False)

# 4. Export all molecular graphs to a readable text file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    f.write("ESOL MOLECULAR GRAPH DATA\n")
    f.write("=========================\n\n")

    for split in ["train", "val", "test"]:

        graphs = data[split]

        f.write(f"\n{'=' * 50}\n")
        f.write(f"{split.upper()} DATASET\n")
        f.write(f"Total molecules: {len(graphs)}\n")
        f.write(f"{'=' * 50}\n")

        for i, graph in enumerate(graphs):

            f.write(f"\n--- Molecule {i + 1} ---\n")

            f.write(f"Molecule ID: {graph.molecule_id}\n")

            f.write(f"Number of atoms: {graph.x.shape[0]}\n")

            f.write(
                f"Number of directed edges: "
                f"{graph.edge_index.shape[1]}\n"
            )

            f.write("\nAtom Features:\n")
            f.write(str(graph.x.tolist()) + "\n")

            f.write("\nEdge Connections:\n")
            f.write(str(graph.edge_index.tolist()) + "\n")

            f.write("\nMeasured Solubility:\n")
            f.write(str(graph.y.tolist()) + "\n")

            f.write("\n" + "-" * 50 + "\n")

# 5. Confirm successful export
print("Export completed successfully!")
print(f"Readable file saved to: {OUTPUT_FILE}")
