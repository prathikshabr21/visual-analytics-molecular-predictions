
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv(
    "experiments/gcn_baseline_history.csv"
)

predictions = pd.read_csv(
    "experiments/gcn_baseline_predictions.csv"
)

Path("experiments/figures").mkdir(
    parents=True, exist_ok=True
)

# 1. Training and validation RMSE
plt.figure(figsize=(9, 5))

plt.plot(
    history["epoch"],
    history["train_rmse"],
    label="Training RMSE"
)

plt.plot(
    history["epoch"],
    history["val_rmse"],
    label="Validation RMSE"
)

plt.xlabel("Epoch")
plt.ylabel("RMSE (log S)")
plt.title("GCN Baseline: Training Performance")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "experiments/figures/training_curves.png",
    dpi=200
)
plt.show()


# 2. Predicted versus actual solubility
plt.figure(figsize=(7, 6))

plt.scatter(
    predictions["actual_logS"],
    predictions["predicted_logS"],
    alpha=0.7
)

lower = min(
    predictions["actual_logS"].min(),
    predictions["predicted_logS"].min()
)

upper = max(
    predictions["actual_logS"].max(),
    predictions["predicted_logS"].max()
)

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Measured log S")
plt.ylabel("Predicted log S")
plt.title("GCN Baseline: Test Predictions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "experiments/figures/predicted_vs_actual.png",
    dpi=200
)
plt.show()


# 3. Prediction error distribution
plt.figure(figsize=(8, 5))

plt.hist(
    predictions["error"],
    bins=25,
    edgecolor="black",
    alpha=0.75
)

plt.axvline(0, linestyle="--")

plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Molecules")
plt.title("GCN Baseline: Error Distribution")
plt.tight_layout()

plt.savefig(
    "experiments/figures/error_distribution.png",
    dpi=200
)
plt.show()

print("All three figures saved successfully.")
