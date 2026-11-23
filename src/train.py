
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from model import GCNRegressor


SEED = 42
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

Path("checkpoints").mkdir(exist_ok=True)
Path("experiments").mkdir(exist_ok=True)

# Load our prepared molecular graphs.
# weights_only=False is needed for the trusted local
# PyG graph objects created by prepare_data.py.
splits = torch.load(
    "data/esol_graphs.pt",
    weights_only=False,
    map_location="cpu"
)

train_loader = DataLoader(
    splits["train"], batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    splits["val"], batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    splits["test"], batch_size=BATCH_SIZE, shuffle=False
)

model = GCNRegressor()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)
criterion = nn.MSELoss()


def train_one_epoch():
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:
        optimizer.zero_grad()

        predictions = model(
            batch.x, batch.edge_index, batch.batch
        )
        targets = batch.y.view(-1)

        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_graphs += targets.numel()

    return total_loss / total_graphs


@torch.no_grad()
def evaluate(loader):
    model.eval()
    predictions = []
    targets = []

    for batch in loader:
        output = model(
            batch.x, batch.edge_index, batch.batch
        )

        predictions.extend(output.cpu().tolist())
        targets.extend(batch.y.view(-1).cpu().tolist())

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    errors = predictions - targets

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    return rmse, mae, predictions, targets


history = []
best_val_rmse = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()

    train_rmse, train_mae, _, _ = evaluate(train_loader)
    val_rmse, val_mae, _, _ = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "val_rmse": val_rmse,
        "val_mae": val_mae
    })

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(
            model.state_dict(),
            "checkpoints/gcn_baseline.pt"
        )
        best_epoch = epoch

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f}"
        )


# Reload the checkpoint selected on validation data.
model.load_state_dict(
    torch.load(
        "checkpoints/gcn_baseline.pt",
        weights_only=True,
        map_location="cpu"
    )
)

test_rmse, test_mae, test_preds, test_targets = evaluate(
    test_loader
)

pd.DataFrame(history).to_csv(
    "experiments/gcn_baseline_history.csv",
    index=False
)

test_ids = [
    graph.molecule_id
    for graph in splits["test"]
]

pd.DataFrame({
    "molecule_id": test_ids,
    "actual_logS": test_targets,
    "predicted_logS": test_preds,
    "error": test_preds - test_targets,
    "absolute_error": np.abs(test_preds - test_targets)
}).to_csv(
    "experiments/gcn_baseline_predictions.csv",
    index=False
)

print("\n--- BASELINE RESULTS ---")
print(f"Best epoch: {best_epoch}")
print(f"Best validation RMSE: {best_val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print("Saved checkpoint, training history and predictions.")
