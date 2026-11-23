
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Configure the dashboard
st.set_page_config(
    page_title="Molecular Solubility Dashboard",
    page_icon="🧪",
    layout="wide"
)

st.title("Molecular Solubility Prediction Dashboard")
st.write("Visual Analytics of GNN Predictions on the ESOL Dataset")

# 2. Load our GCN prediction results
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "experiments" / "gcn_baseline_predictions.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Load the original ESOL molecular properties
MOLECULE_PATH = PROJECT_ROOT / "data" / "delaney-processed.csv"

molecules = pd.read_csv(MOLECULE_PATH)

# Create a molecule_id matching the original CSV row index
molecules = molecules.reset_index().rename(
    columns={"index": "molecule_id"}
)

# Join GCN predictions with molecular properties
df = df.merge(
    molecules,
    on="molecule_id",
    how="left",
    validate="one_to_one"
)

if df["Compound ID"].isna().any():
    st.error("Some predictions could not be matched to ESOL molecules.")
    st.stop()


st.success(f"Successfully loaded {len(df)} test molecules!")

# 3. Display evaluation metrics
rmse = ((df["actual_logS"] - df["predicted_logS"]) ** 2).mean() ** 0.5
mae = (df["actual_logS"] - df["predicted_logS"]).abs().mean()

col1, col2, col3 = st.columns(3)

col1.metric("Test RMSE", f"{rmse:.4f}")
col2.metric("Test MAE", f"{mae:.4f}")
col3.metric("Test Molecules", len(df))

# 4. Create our first interactive chart
st.subheader("Predicted vs Actual Solubility")

fig = px.scatter(
    df,
    x="actual_logS",
    y="predicted_logS",
    hover_data=["molecule_id", "absolute_error"],
    labels={
        "actual_logS": "Measured log Solubility",
        "predicted_logS": "Predicted log Solubility"
    },
    title="GCN Baseline: Predictions vs Ground Truth"
)

# Add the perfect-prediction reference line
minimum = min(df["actual_logS"].min(), df["predicted_logS"].min())
maximum = max(df["actual_logS"].max(), df["predicted_logS"].max())

fig.add_shape(
    type="line",
    x0=minimum,
    y0=minimum,
    x1=maximum,
    y1=maximum,
    line=dict(color="red", dash="dash")
)

st.plotly_chart(fig, use_container_width=True)

# 5. Show the underlying data
with st.expander("View Prediction Data"):
    st.dataframe(df, use_container_width=True)

# 6. Molecular property analysis
st.subheader("Molecular Properties vs Prediction Error")

st.write(
    "Explore how the GCN prediction error varies "
    "with different molecular characteristics."
)

property_options = [
    "Molecular Weight",
    "Polar Surface Area",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Number of H-Bond Donors",
    "Minimum Degree"
]

selected_property = st.selectbox(
    "Select a molecular property:",
    property_options
)

correlation = df[selected_property].corr(df["absolute_error"])

st.metric(
    "Pearson Correlation with Absolute Error",
    f"{correlation:.3f}" if pd.notna(correlation) else "N/A"
)

fig_property = px.scatter(
    df,
    x=selected_property,
    y="absolute_error",
    hover_data=[
        "Compound ID",
        "smiles",
        "actual_logS",
        "predicted_logS"
    ],
    labels={
        "absolute_error": "Absolute Prediction Error (log S units)",
        selected_property: selected_property
    },
    title=f"{selected_property} vs Absolute Prediction Error"
)

st.plotly_chart(fig_property, use_container_width=True)



