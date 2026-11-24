
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "dashboard_data"
    / "all_molecules.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dashboard"
    / "assets"
    / "molecules"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

assert len(df) == 1128
assert df["molecule_id"].is_unique

generated = 0
failed = []

print("Generating molecular structures...")

for row in df.itertuples(index=False):

    molecule_id = int(row.molecule_id)
    smiles = row.smiles

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        failed.append(molecule_id)
        continue

    rdDepictor.Compute2DCoords(mol)

    drawer = Draw.MolDraw2DSVG(450, 320)

    options = drawer.drawOptions()
    options.padding = 0.12
    options.bondLineWidth = 2.0

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()

    output_path = OUTPUT_DIR / f"{molecule_id}.svg"

    output_path.write_text(
        svg,
        encoding="utf-8",
    )

    generated += 1

    if generated % 100 == 0:
        print(f"Generated {generated}/{len(df)} structures")


total_size = sum(
    file.stat().st_size
    for file in OUTPUT_DIR.glob("*.svg")
)

print("\n--- MOLECULAR STRUCTURE EXPORT ---")
print(f"Total molecules: {len(df)}")
print(f"Generated structures: {generated}")
print(f"Failed structures: {len(failed)}")
print(f"Total size: {total_size / (1024 ** 2):.2f} MB")
print(f"Output directory: {OUTPUT_DIR}")

if failed:
    print(f"Failed molecule IDs: {failed}")

assert generated == len(df)
assert not failed

print("\nMolecular structure export completed successfully.")
