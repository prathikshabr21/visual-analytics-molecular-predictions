
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch_geometric.loader import DataLoader

from model import (
    GCNRegressor,
    GraphSAGERegressor,
    GATRegressor,
    EdgeGATRegressor,
    GINERegressor,
)


# ==========================================
# 1. Command-line arguments
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=[
        "gcn",
        "graphsage",
        "gat",
        "edgegat",
        "gine"
    ],
    required=True
)

parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--patience", type=int, default=40)

parser.add_argument(
    "--pooling",
    choices=["mean", "sum", "max", "combined"],
    default="mean"
)

parser.add_argument(
    "--dataset",
    choices=["original", "enhanced", "bonds"],
    default="original"
)

parser.add_argument(
    "--evaluate_test",
    action="store_true"
)

args = parser.parse_args()


# ==========================================
# 2. Validate model and dataset compatibility
# ==========================================

EDGE_MODELS = ["edgegat", "gine"]

if args.model in EDGE_MODELS and args.dataset != "bonds":

    parser.error(
        f"{args.model} requires --dataset bonds"
    )

if args.dataset == "bonds" and args.model not in EDGE_MODELS:

    parser.error(
        "Use edgegat or gine with the bond-aware dataset."
    )


# ==========================================
# 3. Reproducibility
# ==========================================

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ==========================================
# 4. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)

DATASET_PATHS = {
    "original": PROJECT_ROOT / "data" / "esol_graphs.pt",

    "enhanced": (
        PROJECT_ROOT / "data" / "esol_graphs_enhanced.pt"
    ),

    "bonds": (
        PROJECT_ROOT / "data" / "esol_graphs_bonds.pt"
    ),
}

DATASET_PATH = DATASET_PATHS[args.dataset]

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


# ==========================================
# 5. Experiment filenames
# ==========================================

run_name = (
    f"{args.model}"
    f"_h{args.hidden_dim}"
    f"_d{args.dropout}"
    f"_lr{args.lr}"
    f"_bs{args.batch_size}"
    f"_e{args.epochs}"
    f"_p{args.patience}"
    f"_seed{args.seed}"
    f"_pool{args.pooling}"
    f"_data{args.dataset}"
)

checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"

history_path = (
    EXPERIMENT_DIR / f"{run_name}_history.csv"
)

predictions_path = (
    EXPERIMENT_DIR / f"{run_name}_predictions.csv"
)

summary_path = (
    EXPERIMENT_DIR / f"{run_name}_summary.csv"
)


# ==========================================
# 6. Load selected dataset
# ==========================================

splits = torch.load(
    DATASET_PATH,
    weights_only=False,
    map_location="cpu"
)

example_graph = splits["train"][0]

num_node_features = example_graph.x.shape[1]

num_edge_features = (
    example_graph.edge_attr.shape[1]
    if example_graph.edge_attr is not None
    else 0
)

if args.model in EDGE_MODELS and num_edge_features == 0:

    raise ValueError(
        "Selected model requires bond features."
    )


# ==========================================
# 7. Data loaders
# ==========================================

generator = torch.Generator()
generator.manual_seed(args.seed)

train_loader = DataLoader(
    splits["train"],
    batch_size=args.batch_size,
    shuffle=True,
    generator=generator
)

train_eval_loader = DataLoader(
    splits["train"],
    batch_size=args.batch_size,
    shuffle=False
)

val_loader = DataLoader(
    splits["val"],
    batch_size=args.batch_size,
    shuffle=False
)

test_loader = DataLoader(
    splits["test"],
    batch_size=args.batch_size,
    shuffle=False
)


# ==========================================
# 8. Select model architecture
# ==========================================

model_classes = {
    "gcn": GCNRegressor,
    "graphsage": GraphSAGERegressor,
    "gat": GATRegressor,
    "edgegat": EdgeGATRegressor,
    "gine": GINERegressor,
}

model_kwargs = {
    "input_dim": num_node_features,
    "hidden_dim": args.hidden_dim,
    "dropout": args.dropout,
    "pooling": args.pooling,
}

if args.model in EDGE_MODELS:

    model_kwargs["edge_dim"] = num_edge_features

model = model_classes[args.model](
    **model_kwargs
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)

criterion = nn.MSELoss()

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


# ==========================================
# 9. Display experiment configuration
# ==========================================

print("\n--- EXPERIMENT CONFIGURATION ---")

print(f"Experiment: {run_name}")
print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"Atom features: {num_node_features}")
print(f"Bond features: {num_edge_features}")
print(f"Hidden dimensions: {args.hidden_dim}")
print(f"Dropout: {args.dropout}")
print(f"Pooling: {args.pooling}")
print(f"Learning rate: {args.lr}")
print(f"Batch size: {args.batch_size}")
print(f"Maximum epochs: {args.epochs}")
print(f"Early stopping patience: {args.patience}")
print(f"Training seed: {args.seed}")
print(f"Trainable parameters: {parameter_count}")

print(f"Training molecules: {len(splits['train'])}")
print(f"Validation molecules: {len(splits['val'])}")
print(f"Test molecules: {len(splits['test'])}")


# ==========================================
# 10. Unified forward pass
# ==========================================

def forward_batch(batch):

    return model(
        batch.x,
        batch.edge_index,
        batch.batch,
        edge_attr=batch.edge_attr
    )


# ==========================================
# 11. Training function
# ==========================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:

        optimizer.zero_grad()

        predictions = forward_batch(batch)

        targets = batch.y.view(-1)

        loss = criterion(
            predictions,
            targets
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_graphs += targets.numel()

    return total_loss / total_graphs


# ==========================================
# 12. Evaluation function
# ==========================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    predictions = []
    targets = []

    for batch in loader:

        output = forward_batch(batch)

        predictions.extend(
            output.cpu().tolist()
        )

        targets.extend(
            batch.y.view(-1).cpu().tolist()
        )

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    errors = predictions - targets

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    mae = float(
        np.mean(np.abs(errors))
    )

    return rmse, mae, predictions, targets


# ==========================================
# 13. Training loop and early stopping
# ==========================================

history = []

best_val_rmse = float("inf")
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(1, args.epochs + 1):

    train_loss = train_one_epoch()

    train_rmse, train_mae, _, _ = evaluate(
        train_eval_loader
    )

    val_rmse, val_mae, _, _ = evaluate(
        val_loader
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
    })

    # Save best validation checkpoint
    if val_rmse < best_val_rmse:

        best_val_rmse = val_rmse
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

    else:

        epochs_without_improvement += 1

    # Display training progress
    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )

    # Early stopping
    if epochs_without_improvement >= args.patience:

        print(
            f"Early stopping at epoch {epoch}"
        )

        break


# ==========================================
# 14. Reload best checkpoint
# ==========================================

if best_epoch == 0:

    raise RuntimeError(
        "Training failed to produce a valid checkpoint."
    )

model.load_state_dict(
    torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu"
    )
)


# ==========================================
# 15. Final validation evaluation
# ==========================================

val_rmse, val_mae, _, _ = evaluate(
    val_loader
)


# ==========================================
# 16. Save training history
# ==========================================

pd.DataFrame(history).to_csv(
    history_path,
    index=False
)


# ==========================================
# 17. Experiment summary
# ==========================================

summary = {
    "run_name": run_name,
    "model": args.model,
    "dataset": args.dataset,
    "num_node_features": num_node_features,
    "num_edge_features": num_edge_features,
    "seed": args.seed,
    "hidden_dim": args.hidden_dim,
    "dropout": args.dropout,
    "pooling": args.pooling,
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "max_epochs": args.epochs,
    "patience": args.patience,
    "parameters": parameter_count,
    "best_epoch": best_epoch,
    "epochs_completed": len(history),
    "val_rmse": val_rmse,
    "val_mae": val_mae,
}

print("\n--- EXPERIMENT RESULTS ---")

print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")
print(f"Atom features: {num_node_features}")
print(f"Bond features: {num_edge_features}")
print(f"Pooling: {args.pooling}")
print(f"Best epoch: {best_epoch}")
print(f"Epochs completed: {len(history)}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAE: {val_mae:.4f}")


# ==========================================
# 18. Optional test evaluation
# ==========================================

if args.evaluate_test:

    test_rmse, test_mae, test_preds, test_targets = (
        evaluate(test_loader)
    )

    test_ids = [
        int(graph.molecule_id)
        for graph in splits["test"]
    ]

    pd.DataFrame({
        "molecule_id": test_ids,
        "actual_logS": test_targets,
        "predicted_logS": test_preds,
        "error": test_preds - test_targets,
        "absolute_error": np.abs(
            test_preds - test_targets
        ),
    }).to_csv(
        predictions_path,
        index=False
    )

    summary["test_rmse"] = test_rmse
    summary["test_mae"] = test_mae

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Predictions: {predictions_path}")


# ==========================================
# 19. Save experiment summary
# ==========================================

pd.DataFrame([summary]).to_csv(
    summary_path,
    index=False
)


# ==========================================
# 20. Completion message
# ==========================================

print("\nExperiment completed successfully.")

print(f"Checkpoint: {checkpoint_path}")
print(f"History: {history_path}")
print(f"Summary: {summary_path}")
