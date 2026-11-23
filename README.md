# Visual Analytics Project — Molecular Solubility Prediction

This project uses a **Graph Convolutional Network (GCN)** to predict aqueous solubility (`log S`) for molecules from the **ESOL (Delaney)** dataset. A **Streamlit** dashboard displays test predictions and explores how prediction errors relate to molecular properties.

> **Current status:** One GCN baseline and a working dashboard. GNN architecture comparisons and split experiments are planned, not yet implemented.

## 1. Project structure

```text
Visual Analytics Project/
├── .venv/                          # Local Python environment
├── data/
│   ├── delaney-processed.csv       # Original ESOL data
│   └── esol_graphs.pt              # Prepared graph data and splits
├── src/
│   ├── check_setup.py             # Check environment and CSV
│   ├── prepare_data.py            # Convert SMILES to graphs; split data
│   ├── model.py                   # Two-layer GCN architecture
│   ├── train.py                   # Train, validate, test, save outputs
│   └── visualize_baseline.py      # Generate static plots
├── dashboard/
│   └── app.py                     # Interactive Streamlit dashboard
├── checkpoints/
│   └── gcn_baseline.pt            # Best validation checkpoint
└── experiments/
    ├── gcn_baseline_history.csv
    ├── gcn_baseline_predictions.csv
    └── figures/                   # Static plots
```

Other optional scripts, such as `src/analyze_errors.py`, may exist depending on which steps you have completed.

## 2. Open the project and activate the environment (macOS)

Open the **Visual Analytics Project** folder in VS Code. Open **Terminal → New Terminal** and run:

```bash
cd "/Users/prathikshabr/Desktop/Visual Analytics Project"
source .venv/bin/activate
```

Confirm that the correct Python is active:

```bash
python --version
python -c "import torch, torch_geometric, rdkit, pandas, streamlit, plotly; print('Dependencies available')"
```

If the virtual environment is missing, create it and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch torch-geometric rdkit pandas numpy scikit-learn matplotlib seaborn streamlit plotly
```

For reproducibility, use the existing `.venv` whenever possible. The project does not yet have a pinned dependency lockfile.

## 3. Fast demonstration: run the dashboard

**Use this during the project review if the saved prediction CSV already exists.** You do **not** need to retrain the model to demonstrate the dashboard.

From the project root with `.venv` activated:

```bash
python -m streamlit run dashboard/app.py
```

Open the local URL shown in the terminal (typically `http://localhost:8501`).

Demonstrate:

1. The app loads **113 held-out test molecules** from `experiments/gcn_baseline_predictions.csv`.
2. The metric cards show **test RMSE ~1.1915** and **test MAE ~0.9574**.
3. **Predicted vs Actual Solubility:** hover over dots to inspect individual predictions and absolute errors. The dashed diagonal represents perfect predictions.
4. **Molecular Properties vs Prediction Error:** choose **Molecular Weight**, **Polar Surface Area**, **Number of Rings**, or another descriptor from the dropdown; hover to inspect molecule details.
5. Expand **View Prediction Data** to show the underlying joined dataset.

To stop the app, return to the terminal and press **Ctrl+C**.

## 4. Full execution sequence (from scratch)

**Run these commands from the project root, in order.** The source CSV must already exist at `data/delaney-processed.csv`.

### Step 1 — Check the environment and dataset

```bash
python src/check_setup.py
```

Expected dataset: **1,128 rows × 10 columns**, with no missing values.

### Step 2 — Prepare molecular graphs

```bash
python src/prepare_data.py
```

What it does:
- Parses each SMILES string using RDKit.
- Builds PyTorch Geometric molecular graphs with **six atom features** per atom and bidirectional bond connections.
- Uses seed **42** for an **80% / 10% / 10%** train/validation/test split.
- Saves `data/esol_graphs.pt`.

Expected split: **902 training**, **113 validation**, **113 test** graphs.

### Step 3 — Train and evaluate the GCN

```bash
python src/train.py
```

Current baseline configuration: **two GCN layers**, **64 hidden features**, **dropout 0.2**, **Adam**, **learning rate 0.001**, **batch size 32**, **up to 150 epochs**. The best checkpoint is selected by validation RMSE and evaluated on the test set.

Outputs:

- `checkpoints/gcn_baseline.pt` — saved model parameters
- `experiments/gcn_baseline_history.csv` — epoch-by-epoch metrics
- `experiments/gcn_baseline_predictions.csv` — test predictions and errors

Previous baseline run (results may change if code/settings change):

| Metric | Value |
|---|---:|
| Best epoch | 145 |
| Best validation RMSE | 1.2395 |
| Test RMSE | 1.1915 |
| Test MAE | 0.9574 |

### Step 4 — Generate static plots

```bash
python src/visualize_baseline.py
```

Creates figures in `experiments/figures/` showing training/validation RMSE, predicted vs measured solubility, and prediction-error distribution.

### Step 5 — Launch the interactive dashboard

```bash
python -m streamlit run dashboard/app.py
```

The dashboard reads the saved test predictions and merges them with the original ESOL molecular descriptors using `molecule_id`.

## 5. What each data file means

- **`delaney-processed.csv`**: original molecular records, SMILES, measured log solubility, and descriptors.
- **`esol_graphs.pt`**: binary PyTorch file containing molecular graph objects and data splits. It is **not** a text file; load it with PyTorch if inspection is needed. Only use `weights_only=False` on files you trust.
- **`gcn_baseline.pt`**: saved trained GCN weights, not a chart or text document.
- **`gcn_baseline_history.csv`**: learning progress by epoch.
- **`gcn_baseline_predictions.csv`**: test-set `molecule_id`, actual and predicted `log S`, signed error, and absolute error.

## 6. How to explain the project in the review

> We use the ESOL dataset of 1,128 molecules. RDKit converts SMILES strings into graphs, where atoms are nodes and chemical bonds define edges. A two-layer GCN learns molecular representations and predicts measured log solubility. We use a reproducible 80/10/10 split, train with mean squared error, select the best checkpoint on validation RMSE, and evaluate it on held-out test molecules. Our Streamlit dashboard shows prediction accuracy and lets us explore how errors vary with molecular properties.

**Research Question 1:** Where does the model perform well, and where does it fail? The dashboard supports exploratory analysis; apparent relationships in scatter plots are **not** proof of causation.

## 7. Troubleshooting

| Problem | What to check |
|---|---|
| `No module named ...` | Activate `.venv` and verify packages are installed in that environment. |
| `FileNotFoundError` | Run commands from the **project root**, not inside `src/` or `dashboard/`. |
| Dashboard missing prediction CSV | Run `python src/prepare_data.py`, then `python src/train.py`. |
| Streamlit URL not opening | Check the URL printed in the terminal and keep the terminal running. |
| Port 8501 already in use | Stop the old Streamlit process with **Ctrl+C**, or use `--server.port 8502`. |
| `.pt` file looks unreadable in VS Code | This is expected: `.pt` is a binary serialized file. |

## 8. Next steps after the first dashboard milestone

- Compare **GCN**, **GraphSAGE**, and **GAT** under consistent settings.
- Investigate hyperparameters and alternative data splits.
- Extend the dashboard to compare models and inspect individual molecules.
- Document findings and select a shared model/checkpoint for later explanation analysis.
