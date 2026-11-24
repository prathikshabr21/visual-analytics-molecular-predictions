
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
from model import GATRegressor, GINERegressor


# ==================================================
# 1. Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

OUTPUT_DIR = EXPERIMENT_DIR / "final_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Load saved datasets
# ==================================================

original = torch.load(
    DATA_DIR / "esol_graphs.pt",
    weights_only=False,
    map_location="cpu",
)

bonds = torch.load(
    DATA_DIR / "esol_graphs_bonds.pt",
    weights_only=False,
    map_location="cpu",
)

datasets = {
    "original": original,
    "bonds": bonds,
}


# ==================================================
# 3. Verify identical dataset splits
# ==================================================

for split_name in ["train", "val", "test"]:

    original_graphs = original[split_name]
    bond_graphs = bonds[split_name]

    original_ids = [
        int(g.molecule_id) for g in original_graphs
    ]

    bond_ids = [
        int(g.molecule_id) for g in bond_graphs
    ]

    assert original_ids == bond_ids, (
        f"Molecule IDs differ in {split_name}"
    )

    original_targets = torch.cat(
        [g.y for g in original_graphs]
    )

    bond_targets = torch.cat(
        [g.y for g in bond_graphs]
    )

    assert torch.equal(
        original_targets, bond_targets
    ), f"Targets differ in {split_name}"

    print(
        f"{split_name.upper()} split verified: "
        f"{len(original_ids)} molecules"
    )


train_ids = {
    int(g.molecule_id) for g in original["train"]
}

val_ids = {
    int(g.molecule_id) for g in original["val"]
}

test_ids = {
    int(g.molecule_id) for g in original["test"]
}

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

assert len(original["test"]) == 113
assert len(test_ids) == 113

print("All split verification checks passed.")


# ==================================================
# 4. Define shortlisted experiments
# ==================================================

experiments = [

    {
        "id": "E11",
        "model": "GAT",
        "seed": 42,
        "dataset": "original",
        "checkpoint":
            "gat_h64_d0.1_lr0.001_bs32_e300_p40_seed42_poolcombined.pt",
    },

    {
        "id": "E15",
        "model": "GAT",
        "seed": 123,
        "dataset": "original",
        "checkpoint":
            "gat_h64_d0.1_lr0.001_bs32_e300_p40_seed123_poolcombined_dataoriginal.pt",
    },

    {
        "id": "E16",
        "model": "GAT",
        "seed": 2026,
        "dataset": "original",
        "checkpoint":
            "gat_h64_d0.1_lr0.001_bs32_e300_p40_seed2026_poolcombined_dataoriginal.pt",
    },

    {
        "id": "E14",
        "model": "GINE",
        "seed": 42,
        "dataset": "bonds",
        "checkpoint":
            "gine_h64_d0.1_lr0.001_bs32_e300_p40_seed42_poolcombined_databonds.pt",
    },

    {
        "id": "E17",
        "model": "GINE",
        "seed": 123,
        "dataset": "bonds",
        "checkpoint":
            "gine_h64_d0.1_lr0.001_bs32_e300_p40_seed123_poolcombined_databonds.pt",
    },

    {
        "id": "E18",
        "model": "GINE",
        "seed": 2026,
        "dataset": "bonds",
        "checkpoint":
            "gine_h64_d0.1_lr0.001_bs32_e300_p40_seed2026_poolcombined_databonds.pt",
    },
]


# ==================================================
# 5. Verify required checkpoints and summaries
# ==================================================

for experiment in experiments:

    checkpoint_path = (
        CHECKPOINT_DIR / experiment["checkpoint"]
    )

    summary_path = EXPERIMENT_DIR / (
        checkpoint_path.stem + "_summary.csv"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

print("All shortlisted checkpoints and summaries found.")


# ==================================================
# 6. Calculate evaluation metrics
# ==================================================

def calculate_metrics(actual, predicted):

    return {
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
    }


# ==================================================
# 7. Evaluate a saved model
# ==================================================

@torch.no_grad()
def evaluate_model(model, loader):

    model.eval()

    predictions = []
    targets = []
    molecule_ids = []

    for batch in loader:

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            edge_attr=batch.edge_attr,
        )

        predictions.extend(output.cpu().tolist())

        targets.extend(
            batch.y.view(-1).cpu().tolist()
        )

        molecule_ids.extend(
            batch.molecule_id.view(-1).cpu().tolist()
        )

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    assert len(predictions) == len(targets)
    assert np.isfinite(predictions).all()

    errors = predictions - targets

    return pd.DataFrame({
        "molecule_id": molecule_ids,
        "actual_logS": targets,
        "predicted_logS": predictions,
        "error": errors,
        "absolute_error": np.abs(errors),
    })


# ==================================================
# 8. Recover original GCN baseline
# ==================================================

results = []
all_predictions = []

baseline_path = (
    EXPERIMENT_DIR / "gcn_baseline_predictions.csv"
)

baseline = pd.read_csv(baseline_path)

assert len(baseline) == 113
assert baseline["molecule_id"].is_unique
assert set(baseline["molecule_id"]) == test_ids

reference_targets = {
    int(g.molecule_id): float(g.y.item())
    for g in original["test"]
}

expected_targets = baseline["molecule_id"].map(
    reference_targets
)

assert np.allclose(
    baseline["actual_logS"],
    expected_targets,
    atol=1e-5,
    rtol=0,
)

baseline_metrics = calculate_metrics(
    baseline["actual_logS"],
    baseline["predicted_logS"],
)

baseline["experiment"] = "E00"
baseline["model"] = "GCN"
baseline["seed"] = 42

baseline.to_csv(
    OUTPUT_DIR / "E00_gcn_seed42_predictions.csv",
    index=False,
)

all_predictions.append(baseline)

results.append({
    "experiment": "E00",
    "model": "GCN",
    "seed": 42,
    "dataset": "original",
    "best_epoch": 145,
    "val_rmse": 1.2395,
    "val_mae": np.nan,
    "test_rmse": baseline_metrics["rmse"],
    "test_mae": baseline_metrics["mae"],
    "test_r2": baseline_metrics["r2"],
})

print("\n--- ORIGINAL GCN BASELINE ---")

print(f"Test RMSE: {baseline_metrics['rmse']:.4f}")
print(f"Test MAE:  {baseline_metrics['mae']:.4f}")
print(f"Test R2:   {baseline_metrics['r2']:.4f}")


# ==================================================
# 9. Evaluate all six shortlisted checkpoints
# ==================================================

for experiment in experiments:

    experiment_id = experiment["id"]
    model_name = experiment["model"]
    seed = experiment["seed"]
    dataset_name = experiment["dataset"]

    print("\n" + "=" * 55)
    print(f"Experiment: {experiment_id}")
    print(f"Model: {model_name}")
    print(f"Training seed: {seed}")
    print("=" * 55)

    splits = datasets[dataset_name]

    test_loader = DataLoader(
        splits["test"],
        batch_size=32,
        shuffle=False,
    )

    val_loader = DataLoader(
        splits["val"],
        batch_size=32,
        shuffle=False,
    )

    # Reconstruct the original architecture.

    if model_name == "GAT":

        model = GATRegressor(
            input_dim=6,
            hidden_dim=64,
            dropout=0.1,
            heads=4,
            pooling="combined",
        )

    elif model_name == "GINE":

        model = GINERegressor(
            input_dim=6,
            hidden_dim=64,
            dropout=0.1,
            pooling="combined",
            edge_dim=14,
        )

    else:
        raise ValueError(model_name)

    # Load trained weights without retraining.

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

    summary_path = EXPERIMENT_DIR / (
        checkpoint_path.stem + "_summary.csv"
    )

    summary = pd.read_csv(summary_path).iloc[0]

    # Verify the checkpoint against validation results.

    validation_df = evaluate_model(
        model, val_loader
    )

    validation_metrics = calculate_metrics(
        validation_df["actual_logS"],
        validation_df["predicted_logS"],
    )

    assert np.isclose(
        validation_metrics["rmse"],
        float(summary["val_rmse"]),
        atol=0.0005,
        rtol=0,
    ), f"Validation RMSE mismatch: {experiment_id}"

    assert np.isclose(
        validation_metrics["mae"],
        float(summary["val_mae"]),
        atol=0.0005,
        rtol=0,
    ), f"Validation MAE mismatch: {experiment_id}"

    print("Checkpoint validation verified.")

    # Final held-out test evaluation.

    prediction_df = evaluate_model(
        model, test_loader
    )

    assert len(prediction_df) == 113
    assert prediction_df["molecule_id"].is_unique
    assert set(prediction_df["molecule_id"]) == test_ids

    expected_targets = prediction_df[
        "molecule_id"
    ].map(reference_targets)

    assert np.allclose(
        prediction_df["actual_logS"],
        expected_targets,
        atol=1e-5,
        rtol=0,
    )

    metrics = calculate_metrics(
        prediction_df["actual_logS"],
        prediction_df["predicted_logS"],
    )

    prediction_df["experiment"] = experiment_id
    prediction_df["model"] = model_name
    prediction_df["seed"] = seed

    prediction_path = OUTPUT_DIR / (
        f"{experiment_id}_"
        f"{model_name.lower()}_"
        f"seed{seed}_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    all_predictions.append(prediction_df)

    results.append({
        "experiment": experiment_id,
        "model": model_name,
        "seed": seed,
        "dataset": dataset_name,
        "best_epoch": int(summary["best_epoch"]),
        "val_rmse": validation_metrics["rmse"],
        "val_mae": validation_metrics["mae"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
    })

    print(
        f"Validation RMSE: "
        f"{validation_metrics['rmse']:.4f}"
    )

    print(
        f"Validation MAE:  "
        f"{validation_metrics['mae']:.4f}"
    )

    print(f"Test RMSE:       {metrics['rmse']:.4f}")
    print(f"Test MAE:        {metrics['mae']:.4f}")
    print(f"Test R2:         {metrics['r2']:.4f}")


# ==================================================
# 10. Save individual experiment results
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "final_model_results.csv",
    index=False,
)

combined_predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)

assert len(combined_predictions) == 7 * 113

combined_predictions.to_csv(
    OUTPUT_DIR / "all_test_predictions.csv",
    index=False,
)


# ==================================================
# 11. Calculate architecture-level statistics
# ==================================================

architecture_results = (
    results_df.groupby("model")
    .agg(
        runs=("seed", "count"),

        mean_val_rmse=("val_rmse", "mean"),
        std_val_rmse=("val_rmse", "std"),

        mean_val_mae=("val_mae", "mean"),
        std_val_mae=("val_mae", "std"),

        mean_test_rmse=("test_rmse", "mean"),
        std_test_rmse=("test_rmse", "std"),

        mean_test_mae=("test_mae", "mean"),
        std_test_mae=("test_mae", "std"),

        mean_test_r2=("test_r2", "mean"),
        std_test_r2=("test_r2", "std"),
    )
    .reset_index()
)

architecture_results.to_csv(
    OUTPUT_DIR / "architecture_comparison.csv",
    index=False,
)


# ==================================================
# 12. Calculate improvement over GCN baseline
# ==================================================

architecture_results["rmse_reduction_percent"] = (
    (
        baseline_metrics["rmse"]
        - architecture_results["mean_test_rmse"]
    )
    / baseline_metrics["rmse"]
    * 100
)

architecture_results["mae_reduction_percent"] = (
    (
        baseline_metrics["mae"]
        - architecture_results["mean_test_mae"]
    )
    / baseline_metrics["mae"]
    * 100
)

architecture_results.to_csv(
    OUTPUT_DIR / "baseline_comparison.csv",
    index=False,
)


# ==================================================
# 13. Print final results
# ==================================================

print("\n" + "=" * 70)
print("FINAL INDIVIDUAL MODEL RESULTS")
print("=" * 70)

print(
    results_df.round(4).to_string(index=False)
)

print("\n" + "=" * 70)
print("ARCHITECTURE COMPARISON")
print("=" * 70)

print(
    architecture_results.round(4).to_string(index=False)
)

print("\nPrimary architecture: GAT")
print("Selected using mean validation RMSE.")
print("Test results are for final evaluation only.")

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nFinal evaluation completed successfully.")
