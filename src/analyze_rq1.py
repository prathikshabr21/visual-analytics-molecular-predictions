
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "dashboard_data"

OUTPUT = ROOT / "experiments" / "rq1_findings"
OUTPUT.mkdir(parents=True, exist_ok=True)

PROPERTIES = [
    "Molecular Weight",
    "Polar Surface Area",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Number of H-Bond Donors",
    "Minimum Degree",
    "LogP",
    "Number of H-Bond Acceptors",
    "Number of Aromatic Rings",
    "Heavy Atom Count",
    "Fraction Csp3",
    "Number of Heteroatoms",
    "Molar Refractivity",
    "Number of Atoms",
    "Number of Bonds",
]

molecules = pd.read_csv(DATA / "all_molecules.csv")

predictions = pd.read_csv(
    DATA / "all_split_predictions.csv"
)

df = predictions.merge(
    molecules,
    on=["molecule_id", "split"],
    how="left",
    validate="many_to_one",
)

assert len(df) == 7896
assert df["Compound ID"].notna().all()


# ==================================================
# 1. PRIMARY MODEL AND TEST PARTITION
# ==================================================

test = df[
    (df["model"] == "GAT")
    & (df["seed"] == 42)
    & (df["split"] == "Testing")
].copy()

assert len(test) == 113

print("\n" + "=" * 65)
print("RESEARCH QUESTION 1")
print("Where does the model perform well, and where does it fail?")
print("=" * 65)

print("\nModel: GAT")
print("Training seed: 42")
print("Partition: Testing")
print(f"Molecules: {len(test)}")

rmse = (test["error"].pow(2).mean()) ** 0.5
mae = test["absolute_error"].mean()

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Mean signed error: {test['error'].mean():.4f}")


# ==================================================
# 2. PREDICTION ACCURACY THRESHOLDS
# ==================================================

threshold_records = []

for threshold in [0.25, 0.5, 1.0, 1.5, 2.0]:

    count = int(
        (test["absolute_error"] <= threshold).sum()
    )

    threshold_records.append({
        "threshold": threshold,
        "within_threshold": count,
        "above_threshold": len(test) - count,
        "percentage_within": count / len(test) * 100,
    })

thresholds = pd.DataFrame(threshold_records)

print("\n--- PREDICTION ACCURACY THRESHOLDS ---")
print(thresholds.round(4).to_string(index=False))

thresholds.to_csv(
    OUTPUT / "error_thresholds.csv",
    index=False,
)


# ==================================================
# 3. ERRORS ACROSS SOLUBILITY RANGES
# ==================================================

df["Solubility range"] = pd.cut(
    df["actual_logS"],
    bins=[
        float("-inf"),
        -6,
        -4,
        -2,
        0,
        float("inf"),
    ],
    labels=[
        "<= -6",
        "-6 to -4",
        "-4 to -2",
        "-2 to 0",
        "> 0",
    ],
)

solubility_results = (
    df.groupby(
        ["model", "seed", "split", "Solubility range"],
        observed=True,
    )
    .agg(
        molecules=("molecule_id", "count"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        mean_signed_error=("error", "mean"),
    )
    .reset_index()
)

solubility_results["rmse"] = (
    solubility_results["mse"] ** 0.5
)

solubility_results = solubility_results.drop(
    columns="mse"
)

solubility_results.to_csv(
    OUTPUT / "solubility_range_errors.csv",
    index=False,
)

primary_ranges = solubility_results[
    (solubility_results["model"] == "GAT")
    & (solubility_results["seed"] == 42)
    & (solubility_results["split"] == "Testing")
]

print("\n--- ERRORS BY SOLUBILITY RANGE ---")

print(
    primary_ranges[
        [
            "Solubility range",
            "molecules",
            "rmse",
            "mae",
            "mean_signed_error",
        ]
    ].round(4).to_string(index=False)
)


# ==================================================
# 4. MOLECULAR PROPERTY ASSOCIATIONS
# ==================================================

correlation_records = []

for (model, seed, split), group in df.groupby(
    ["model", "seed", "split"]
):

    for property_name in PROPERTIES:

        for error_type in ["absolute_error", "error"]:

            pearson = group[property_name].corr(
                group[error_type]
            )

            spearman = group[property_name].rank().corr(
                group[error_type].rank()
            )

            correlation_records.append({
                "model": model,
                "seed": seed,
                "split": split,
                "property": property_name,
                "error_type": error_type,
                "pearson": pearson,
                "spearman": spearman,
                "molecules": len(group),
            })

correlations = pd.DataFrame(correlation_records)

correlations.to_csv(
    OUTPUT / "property_correlations.csv",
    index=False,
)

primary_correlations = correlations[
    (correlations["model"] == "GAT")
    & (correlations["seed"] == 42)
    & (correlations["split"] == "Testing")
    & (correlations["error_type"] == "absolute_error")
].copy()

primary_correlations["strength"] = (
    primary_correlations["spearman"].abs()
)

primary_correlations = primary_correlations.sort_values(
    "strength",
    ascending=False,
)

print("\n--- MOLECULAR PROPERTY ASSOCIATIONS ---")

print(
    primary_correlations[
        ["property", "pearson", "spearman"]
    ].round(4).to_string(index=False)
)


# ==================================================
# 5. ACCURATE AND INACCURATE PREDICTIONS
# ==================================================

columns = [
    "molecule_id",
    "Compound ID",
    "smiles",
    "actual_logS",
    "predicted_logS",
    "error",
    "absolute_error",
    "Molecular Weight",
    "LogP",
    "Number of Rings",
]

best = test.nsmallest(
    5, "absolute_error"
)[columns]

worst = test.nlargest(
    5, "absolute_error"
)[columns]

best.to_csv(
    OUTPUT / "accurate_examples.csv",
    index=False,
)

worst.to_csv(
    OUTPUT / "inaccurate_examples.csv",
    index=False,
)

display_columns = [
    "molecule_id",
    "Compound ID",
    "actual_logS",
    "predicted_logS",
    "absolute_error",
]

print("\n--- FIVE MOST ACCURATE PREDICTIONS ---")

print(
    best[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n--- FIVE LEAST ACCURATE PREDICTIONS ---")

print(
    worst[display_columns]
    .round(4)
    .to_string(index=False)
)


# ==================================================
# 6. CONSISTENCY ACROSS GAT TRAINING SEEDS
# ==================================================

gat_test = df[
    (df["model"] == "GAT")
    & (df["split"] == "Testing")
].copy()

consistency = (
    gat_test.groupby(
        ["molecule_id", "Compound ID"],
        as_index=False,
    )
    .agg(
        runs=("seed", "nunique"),
        mean_absolute_error=("absolute_error", "mean"),
        min_absolute_error=("absolute_error", "min"),
        max_absolute_error=("absolute_error", "max"),
        std_absolute_error=("absolute_error", "std"),
    )
)

assert len(consistency) == 113
assert consistency["runs"].eq(3).all()

consistency = consistency.sort_values(
    "mean_absolute_error",
    ascending=False,
)

consistency.to_csv(
    OUTPUT / "gat_seed_consistency.csv",
    index=False,
)

print("\n--- HIGHEST MEAN ERRORS ACROSS GAT SEEDS ---")

print(
    consistency.head(5)
    .round(4)
    .to_string(index=False)
)

print("\n--- LOWEST MEAN ERRORS ACROSS GAT SEEDS ---")

print(
    consistency.tail(5)
    .sort_values("mean_absolute_error")
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 65)
print("RQ1 ANALYSIS COMPLETED SUCCESSFULLY")
print(f"Results saved to: {OUTPUT}")
print("=" * 65)
