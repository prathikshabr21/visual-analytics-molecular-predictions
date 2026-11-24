
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from torch_geometric.loader import DataLoader

from model import (
    GCNRegressor,
    GATRegressor,
    GINERegressor,
)


# ==================================================
# 1. Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENT_DIR / "dashboard_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Load datasets
# ==================================================

datasets = {
    "original": torch.load(
        DATA_DIR / "esol_graphs.pt",
        weights_only=False,
        map_location="cpu",
    ),

    "bonds": torch.load(
        DATA_DIR / "esol_graphs_bonds.pt",
        weights_only=False,
        map_location="cpu",
    ),
}

molecules = pd.read_csv(
    OUTPUT_DIR / "all_molecules.csv"
)

split_names = {
    "train": "Training",
    "val": "Validation",
    "test": "Testing",
}

expected_counts = {
    "train": 902,
    "val": 113,
    "test": 113,
}


# ==================================================
# 3. Verify dataset consistency
# ==================================================

for split_key, split_label in split_names.items():

    expected_ids = set(
        molecules.loc[
            molecules["split"] == split_label,
            "molecule_id"
        ]
    )

    for dataset_name, dataset in datasets.items():

        graphs = dataset[split_key]

        graph_ids = {
            int(graph.molecule_id)
            for graph in graphs
        }

        assert len(graphs) == expected_counts[split_key]

        assert graph_ids == expected_ids

    print(
        f"{split_label} verified: "
        f"{len(expected_ids)} molecules"
    )


# ==================================================
# 4. Define existing model checkpoints
# ==================================================

experiments = [
    {
        "experiment": "E00",
        "model": "GCN",
        "seed": 42,
        "dataset": "original",
        "checkpoint": "gcn_baseline.pt",
    },
    {
        "experiment": "E11",
        "model": "GAT",
        "seed": 42,
        "dataset": "original",
        "checkpoint":
            "gat_h64_d0.1_lr0.001_bs32_e300_p40_seed42_poolcombined.pt",
    },
    {
        "experiment": "E15",
        "model": "GAT",
        "seed": 123,
        "dataset": "original",
        "checkpoint":
            "gat_h64_d0.1_lr0.001_bs32_e300_p40_seed123_poolcombined_dataoriginal.pt",
    },
    {
        "experiment": "E16",
        "model": "GAT",
        "seed": 2026,
        "dataset": "original",
        "checkpoint":
            "gat_h64_d0.1_lr0.001_bs32_e300_p40_seed2026_poolcombined_dataoriginal.pt",
    },
    {
        "experiment": "E14",
        "model": "GINE",
        "seed": 42,
        "dataset": "bonds",
        "checkpoint":
            "gine_h64_d0.1_lr0.001_bs32_e300_p40_seed42_poolcombined_databonds.pt",
    },
    {
        "experiment": "E17",
        "model": "GINE",
        "seed": 123,
        "dataset": "bonds",
        "checkpoint":
            "gine_h64_d0.1_lr0.001_bs32_e300_p40_seed123_poolcombined_databonds.pt",
    },
    {
        "experiment": "E18",
        "model": "GINE",
        "seed": 2026,
        "dataset": "bonds",
        "checkpoint":
            "gine_h64_d0.1_lr0.001_bs32_e300_p40_seed2026_poolcombined_databonds.pt",
    },
]


# ==================================================
# 5. Reconstruct model architectures
# ==================================================

def create_model(model_name):

    if model_name == "GCN":

        return GCNRegressor(
            input_dim=6,
            hidden_dim=64,
            dropout=0.2,
            pooling="mean",
        )

    if model_name == "GAT":

        return GATRegressor(
            input_dim=6,
            hidden_dim=64,
            dropout=0.1,
            heads=4,
            pooling="combined",
        )

    if model_name == "GINE":

        return GINERegressor(
            input_dim=6,
            hidden_dim=64,
            dropout=0.1,
            pooling="combined",
            edge_dim=14,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ==================================================
# 6. Generate molecule-level predictions
# ==================================================

@torch.no_grad()
def predict(model, graphs):

    model.eval()

    loader = DataLoader(
        graphs,
        batch_size=32,
        shuffle=False,
    )

    records = []

    for batch in loader:

        outputs = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
        )

        molecule_ids = (
            batch.molecule_id.view(-1).tolist()
        )

        actual_values = (
            batch.y.view(-1).tolist()
        )

        predicted_values = (
            outputs.view(-1).tolist()
        )

        for molecule_id, actual, predicted in zip(
            molecule_ids,
            actual_values,
            predicted_values,
        ):

            error = predicted - actual

            records.append({
                "molecule_id": int(molecule_id),
                "actual_logS": actual,
                "predicted_logS": predicted,
                "error": error,
                "absolute_error": abs(error),
                "squared_error": error ** 2,
            })

    df = pd.DataFrame(records)

    assert len(df) == len(graphs)

    assert df["molecule_id"].is_unique

    assert np.isfinite(
        df["predicted_logS"]
    ).all()

    return df


# ==================================================
# 7. Calculate split-specific metrics
# ==================================================

def calculate_metrics(df):

    actual = df["actual_logS"]
    predicted = df["predicted_logS"]

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "mae": float(
            mean_absolute_error(actual, predicted)
        ),

        "r2": float(
            r2_score(actual, predicted)
        ),

        "mean_signed_error": float(
            df["error"].mean()
        ),
    }


# ==================================================
# 8. Evaluate all existing checkpoints
# ==================================================

all_predictions = []
all_metrics = []

for experiment in experiments:

    experiment_id = experiment["experiment"]
    model_name = experiment["model"]
    seed = experiment["seed"]

    print("\n" + "=" * 60)

    print(
        f"{experiment_id} | "
        f"{model_name} | Seed {seed}"
    )

    print("=" * 60)

    model = create_model(model_name)

    checkpoint_path = (
        CHECKPOINT_DIR / experiment["checkpoint"]
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            weights_only=True,
            map_location="cpu",
        )
    )

    dataset = datasets[experiment["dataset"]]

    for split_key, split_label in split_names.items():

        predictions = predict(
            model,
            dataset[split_key],
        )

        predictions["experiment"] = experiment_id
        predictions["model"] = model_name
        predictions["seed"] = seed
        predictions["split"] = split_label

        metrics = calculate_metrics(predictions)

        all_predictions.append(predictions)

        all_metrics.append({
            "experiment": experiment_id,
            "model": model_name,
            "seed": seed,
            "split": split_label,
            "molecules": len(predictions),
            **metrics,
        })

        print(
            f"{split_label:12s} | "
            f"N: {len(predictions):4d} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"MAE: {metrics['mae']:.4f} | "
            f"R2: {metrics['r2']:.4f}"
        )


# ==================================================
# 9. Combine all prediction records
# ==================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

metrics_df = pd.DataFrame(all_metrics)

assert len(predictions_df) == 1128 * 7

assert len(metrics_df) == 7 * 3

assert not predictions_df.duplicated(
    subset=["experiment", "molecule_id"]
).any()

assert predictions_df.isna().sum().sum() == 0

reference_targets = molecules.set_index(
    "molecule_id"
)["Measured logS"]

assert np.allclose(
    predictions_df["actual_logS"],
    predictions_df["molecule_id"].map(
        reference_targets
    ),
    atol=1e-5,
    rtol=0,
)

reference_splits = molecules.set_index(
    "molecule_id"
)["split"]

assert (
    predictions_df["split"]
    == predictions_df["molecule_id"].map(
        reference_splits
    )
).all()


# ==================================================
# 10. Verify against previous final test evaluation
# ==================================================

reference_test = pd.read_csv(
    EXPERIMENT_DIR
    / "final_evaluation"
    / "all_test_predictions.csv"
)

new_test = predictions_df[
    predictions_df["split"] == "Testing"
]

comparison = new_test.merge(
    reference_test,
    on=["experiment", "molecule_id"],
    suffixes=("_new", "_reference"),
    validate="one_to_one",
)

assert len(comparison) == 791

assert np.allclose(
    comparison["actual_logS_new"],
    comparison["actual_logS_reference"],
    atol=1e-5,
    rtol=0,
)

assert np.allclose(
    comparison["predicted_logS_new"],
    comparison["predicted_logS_reference"],
    atol=1e-4,
    rtol=0,
)

print("\nPrevious test predictions verified successfully.")


# ==================================================
# 11. Export dashboard files
# ==================================================

predictions_df.to_csv(
    OUTPUT_DIR / "all_split_predictions.csv",
    index=False,
)

metrics_df.to_csv(
    OUTPUT_DIR / "all_split_metrics.csv",
    index=False,
)


# ==================================================
# 12. Final summary
# ==================================================

print("\n" + "=" * 60)
print("DASHBOARD PREDICTION EXPORT")
print("=" * 60)

print(
    f"Total prediction records: "
    f"{len(predictions_df)}"
)

print(
    f"Unique molecules: "
    f"{predictions_df['molecule_id'].nunique()}"
)

print(
    f"Model checkpoints: "
    f"{predictions_df['experiment'].nunique()}"
)

print(
    f"Metric records: {len(metrics_df)}"
)

print("\nPrediction records per split:")

print(
    predictions_df["split"]
    .value_counts()
    .to_string()
)

print("\nGenerated files:")

print(OUTPUT_DIR / "all_split_predictions.csv")
print(OUTPUT_DIR / "all_split_metrics.csv")

print("\nDashboard prediction export completed successfully.")
