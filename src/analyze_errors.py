
import pandas as pd

dataset = pd.read_csv("data/delaney-processed.csv")
predictions = pd.read_csv(
    "experiments/gcn_baseline_predictions.csv"
)

# Connect predictions to the original dataset
results = predictions.merge(
    dataset.reset_index().rename(
        columns={"index": "molecule_id"}
    ),
    on="molecule_id",
    how="left",
    validate="one_to_one"
)

print("\n--- FIVE MOST ACCURATE PREDICTIONS ---")
print(
    results.nsmallest(5, "absolute_error")[
        [
            "Compound ID",
            "actual_logS",
            "predicted_logS",
            "absolute_error"
        ]
    ].to_string(index=False)
)

print("\n--- FIVE LARGEST PREDICTION ERRORS ---")
print(
    results.nlargest(5, "absolute_error")[
        [
            "Compound ID",
            "actual_logS",
            "predicted_logS",
            "absolute_error",
            "Molecular Weight",
            "Polar Surface Area"
        ]
    ].to_string(index=False)
)

print("\n--- ERROR SUMMARY ---")
print("Mean signed error:", round(results["error"].mean(), 4))
print("Median absolute error:",
      round(results["absolute_error"].median(), 4))

results.to_csv(
    "experiments/gcn_baseline_molecules.csv",
    index=False
)

print("\nSaved molecule-level results.")
