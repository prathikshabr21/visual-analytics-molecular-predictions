
# Understanding Molecular Predictions

## Visual Analytics of Molecular Solubility Predictions Using Graph Neural Networks

This project investigates molecular solubility prediction using Graph Neural Networks (GNNs) and interactive visual analytics.

The ESOL (Delaney) dataset is used to train, evaluate and compare multiple GNN architectures. An interactive Streamlit dashboard enables users to investigate model performance, prediction errors, molecular properties and individual molecular structures.

## Live Dashboard

- [Interactive Streamlit Dashboard](https://visual-analytics-molecular-predictions.streamlit.app/)
- [GitHub Repository](https://github.com/prathikshabr21/visual-analytics-molecular-predictions)

The dashboard uses precomputed predictions and molecular visualizations. Model retraining is not required to explore the results.

## 1. Research Objectives

### RQ1 — Model Performance

Where does the model perform well, and where does it fail?

- Evaluate overall prediction performance.
- Investigate errors across different solubility ranges.
- Explore relationships between molecular properties and prediction errors.
- Identify accurately and inaccurately predicted molecules.
- Connect aggregate performance with individual molecular structures.

### RQ2 — Explanation Quality

How well does the assigned explanation method reflect the model's behavior?

Planned work includes quantitative explanation evaluation, fidelity analysis and comparison of explanation quality across prediction accuracy levels.

### RQ3 — Molecular Attributions

What can we learn by investigating individual molecules and their attributions?

Planned work includes interactive exploration of molecular structures, atom-level and bond-level attributions, predictions and explanation scores.

The current implementation primarily addresses RQ1.

## 2. Dataset

The project uses the ESOL dataset introduced by Delaney (2004).

| Characteristic | Description |
|---|---|
| Dataset | ESOL / Delaney |
| Total molecules | 1,128 |
| Original dataset columns | 10 |
| Prediction target | Measured aqueous log solubility |
| Target unit | log10(mol/L) |
| Learning task | Regression |
| Molecular representation | SMILES and molecular graphs |
| Dashboard molecular properties | 15 |

The dataset contains experimentally measured aqueous solubility values and molecular descriptors.

## 3. Molecular Graph Representation

RDKit converts SMILES strings into molecular graphs.

- Nodes represent atoms.
- Edges represent chemical bonds.
- Bonds are represented as bidirectional graph connections.
- Each graph has a measured solubility target.
- Molecule identifiers preserve the relationship between graphs, predictions and original molecular records.

### Atom Features

The original representation contains six atom features:

1. Atomic number
2. Atom degree
3. Formal charge
4. Total number of hydrogen atoms
5. Aromaticity
6. Ring membership

An enhanced representation containing 20 atom features was also investigated.

### Bond Features

Bond-aware architectures use 14 additional edge features representing:

- Bond type
- Conjugation
- Ring membership
- Bond stereochemistry

## 4. Dataset Splitting

The original experiments use a reproducible random 80/10/10 split.

| Partition | Molecules |
|---|---:|
| Training | 902 |
| Validation | 113 |
| Testing | 113 |
| Total | 1,128 |

The validation set is used for checkpoint selection and model comparison. The test set is used for held-out performance evaluation.

### Alternative Dataset Splits

Three partitioning strategies have been prepared:

| Split | Training | Validation | Testing |
|---|---:|---:|---:|
| Random 42 | 902 | 113 | 113 |
| Random 123 | 902 | 113 | 113 |
| Scaffold 42 | 902 | 113 | 113 |

The scaffold-based partition groups molecules according to extracted molecular scaffolds.

Preparation results:

- All 1,128 molecules retained.
- 269 scaffold groups identified.
- 317 acyclic molecules.
- No molecule overlap within each split.
- No scaffold group overlap between scaffold-based partitions.

A workaround using RDKit's alternative scaffold extraction implementation was required because the standard implementation failed in RDKit 2026.09.2.

The workaround passed full-dataset extraction and partition integrity checks. Its equivalence to the standard implementation has not been independently verified.

Alternative split training and dashboard integration are planned for the next development milestone.

## 5. Model Architectures

Five GNN architectures have been investigated.

| Architecture | Description |
|---|---|
| GCN | Graph Convolutional Network |
| GraphSAGE | Neighborhood feature aggregation |
| GAT | Graph Attention Network |
| EdgeGAT | Graph attention incorporating bond features |
| GINE | Graph Isomorphism Network incorporating edge features |

The models generate molecular representations through message passing, graph-level pooling and a regression layer.

### Pooling Strategies

The following graph-level pooling strategies were evaluated:

- Global mean pooling
- Global sum pooling
- Global max pooling
- Combined mean, sum and max pooling

Combined pooling concatenates the three graph-level representations.

## 6. Training Methodology

The experimental framework investigates:

- Model architecture
- Hidden dimension
- Learning rate
- Dropout
- Training duration
- Pooling strategy
- Atom feature representation
- Bond feature representation
- Training seed

Training uses:

- Adam optimizer
- Mean Squared Error loss
- Mini-batch training
- Validation RMSE for checkpoint selection
- Early stopping in comparison experiments

A total of 19 full training experiments have been completed, excluding short implementation smoke tests.

Detailed configurations and experimental findings are available in [EXPERIMENTS.md](EXPERIMENTS.md).

## 7. Evaluation Metrics

Model performance is evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

Lower RMSE and MAE indicate better prediction accuracy.

Higher R² indicates better agreement between measured and predicted target values relative to a constant-mean reference.

Signed prediction error is defined as:

    Error = Predicted logS - Measured logS

Positive errors indicate overprediction, while negative errors indicate underprediction.

## 8. Experimental Results

### Original GCN Baseline

| Metric | Value |
|---|---:|
| Best epoch | 145 |
| Validation RMSE | 1.2395 |
| Validation MAE | 0.9700 |
| Test RMSE | 1.1915 |
| Test MAE | 0.9574 |
| Test R² | 0.6666 |

### Multi-Seed Architecture Comparison

The optimized GAT and GINE configurations were evaluated using training seeds 42, 123 and 2026 on the original fixed dataset split.

| Model | Mean Validation RMSE | Mean Test RMSE | Mean Test MAE | Mean Test R² |
|---|---:|---:|---:|---:|
| GCN baseline | 1.2395 | 1.1915 | 0.9574 | 0.6666 |
| GAT | 0.7371 | 0.7634 | 0.5306 | 0.8630 |
| GINE | 0.7385 | 0.7702 | 0.5303 | 0.8605 |

The GCN baseline represents a single training run. GAT and GINE results represent averages across three training seeds.

### Improvement over the Baseline

The optimized GAT configuration achieved approximately:

- 35.93% reduction in mean test RMSE.
- 44.58% reduction in mean test MAE.

GAT is the recommended candidate architecture based on its mean validation RMSE and observed training-seed stability.

The difference between GAT and GINE is small and does not establish statistical superiority.

The dashboard uses GAT with training seed 42 as its default reference model. Final shared checkpoint selection remains subject to project discussion.

## 9. Interactive Visual Analytics Dashboard

The dashboard contains eight main pages.

### Project Overview

Introduces the project, dataset, research questions and model methodology.

### Dataset Explorer

Supports exploration of molecular properties, dataset partitions, distributions and individual molecular structures.

### Model Performance

Compares model performance across architectures, training seeds and dataset partitions.

### Prediction Analysis

Provides:

- Predicted versus measured solubility
- Signed prediction error distributions
- Prediction bias across solubility levels
- Adjustable accuracy thresholds
- Error analysis across solubility ranges

### Molecular Property Analysis

Investigates relationships between molecular properties and prediction errors using scatter plots, correlation measures and grouped error statistics.

### Molecular Explorer

Supports:

- Individual molecular structure inspection
- Measured and predicted solubility comparison
- Cross-model prediction comparison
- Comparison of two molecules

### RQ1 Key Findings

Summarizes important observations about prediction accuracy, error patterns and difficult molecules.

### Experiment Lab

Documents model optimization experiments, validation performance, training histories and experimental configurations.

## 10. Key RQ1 Findings

The default GAT model with training seed 42 achieved:

| Metric | Value |
|---|---:|
| Test molecules | 113 |
| RMSE | 0.7792 |
| MAE | 0.5319 |
| R² | 0.8574 |
| Mean signed error | -0.0536 |

### Prediction Accuracy

| Absolute Error Threshold | Molecules | Percentage |
|---|---:|---:|
| ≤ 0.25 | 46 | 40.71% |
| ≤ 0.50 | 75 | 66.37% |
| ≤ 1.00 | 95 | 84.07% |
| ≤ 1.50 | 106 | 93.81% |
| ≤ 2.00 | 110 | 97.35% |

These thresholds are exploratory and are not universal chemical accuracy standards.

### Solubility Ranges

The model performs comparatively well for molecules with measured logS between -2 and 0.

Larger errors are observed at extreme solubility values, particularly for the small group of molecules with positive measured logS.

The model tends to overpredict very low solubility values and underpredict high solubility values.

### Molecular Characteristics

Molecular size-related descriptors, including atom count, bond count and molecular weight, exhibit moderate positive associations with absolute prediction error.

These observations represent associations rather than causal explanations.

### Individual Molecules

Accurately predicted examples include:

- Diethyl sulfide
- cis-2-Pentene
- 2,2-Dimethyl-1-butanol

Difficult examples include:

- Norbormide
- Dimecron
- Guanine

Several difficult molecules exhibit consistently high prediction errors across multiple GAT training seeds.

## 11. Project Structure

    Visual Analytics Project/
    |
    |-- data/
    |   |-- delaney-processed.csv
    |   |-- esol_graphs.pt
    |   |-- esol_graphs_enhanced.pt
    |   |-- esol_graphs_bonds.pt
    |   |-- Alternative split graph datasets
    |
    |-- src/
    |   |-- check_setup.py
    |   |-- prepare_data.py
    |   |-- prepare_data_enhanced.py
    |   |-- prepare_data_bonds.py
    |   |-- prepare_alternative_splits.py
    |   |-- model.py
    |   |-- train.py
    |   |-- train_comparison.py
    |   |-- evaluate_final.py
    |   |-- prepare_dashboard_data.py
    |   |-- export_dashboard_predictions.py
    |   |-- generate_molecule_images.py
    |   |-- analyze_rq1.py
    |   |-- visualize_baseline.py
    |
    |-- dashboard/
    |   |-- app.py
    |   |-- assets/
    |       |-- molecules/
    |
    |-- experiments/
    |   |-- Model training histories and summaries
    |   |-- final_evaluation/
    |   |-- dashboard_data/
    |   |-- rq1_findings/
    |   |-- split_comparison/
    |   |-- figures/
    |
    |-- checkpoints/
    |-- requirements.txt
    |-- README.md
    |-- EXPERIMENTS.md
    |-- .gitignore

Generated graph datasets and model checkpoints are stored locally and excluded from Git tracking.

Precomputed dashboard data and molecular SVG images are included for dashboard deployment.

## 12. Installation and Dashboard Execution

### Clone the Repository

    git clone https://github.com/prathikshabr21/visual-analytics-molecular-predictions.git

    cd visual-analytics-molecular-predictions

### Create a Python Environment

    python3 -m venv .venv
    source .venv/bin/activate

### Install Dashboard Dependencies

    python -m pip install -r requirements.txt

### Launch the Dashboard

    python -m streamlit run dashboard/app.py

The application can normally be accessed at:

http://localhost:8501

No model retraining is required for dashboard execution.

## 13. Training Environment and Reproduction

The training environment additionally requires:

    python -m pip install torch torch-geometric rdkit numpy pandas scikit-learn matplotlib seaborn

The root requirements.txt intentionally contains only dashboard dependencies for lightweight Streamlit deployment.

The training environment is not yet fully version-pinned.

### Dataset Preparation

    python src/check_setup.py
    python src/prepare_data.py
    python src/prepare_data_enhanced.py
    python src/prepare_data_bonds.py

### Original Baseline Training

    python src/train.py

### Example Optimized GAT Training

    python src/train_comparison.py --model gat --epochs 300 --hidden_dim 64 --dropout 0.1 --lr 0.001 --batch_size 32 --seed 42 --patience 40 --pooling combined --dataset original

### Alternative Dataset Preparation

    python src/prepare_alternative_splits.py

### Final Evaluation and Dashboard Data Generation

After the required trained checkpoints are available:

    python src/evaluate_final.py
    python src/prepare_dashboard_data.py
    python src/export_dashboard_predictions.py
    python src/generate_molecule_images.py
    python src/analyze_rq1.py

Important: Re-running training or preparation scripts may overwrite existing generated files. Preserve the original checkpoints and experiment records before reproducing experiments.

Some earlier checkpoint filenames follow previous naming conventions. The existing evaluation scripts require their expected checkpoint filenames.

## 14. Limitations

- ESOL is a relatively small molecular dataset.
- Most completed model experiments use one fixed random dataset split.
- Multiple training seeds measure training stability rather than independent dataset generalization.
- Repeated validation-based model selection may introduce validation selection bias.
- The original baseline test performance was observed before subsequent model comparisons.
- Correlations between molecular properties and errors do not establish causality.
- Extreme solubility ranges contain relatively few molecules.
- Scaffold extraction uses an alternative RDKit implementation whose equivalence to the standard method remains unverified.
- Attention weights alone are not treated as validated explanations.
- Explanation quality and molecular attribution analysis remain future work.

## 15. Project Progress

### Completed

- [x] Dataset inspection and preprocessing
- [x] Molecular graph generation
- [x] Original GCN baseline
- [x] Five GNN architectures
- [x] Nineteen full training experiments
- [x] Hyperparameter and pooling comparisons
- [x] Atom and bond feature investigations
- [x] Multi-seed evaluation
- [x] RQ1 prediction error analysis
- [x] Interactive molecular visualization
- [x] Eight-page Streamlit dashboard
- [x] GitHub and Streamlit deployment
- [x] Alternative dataset split preparation

### Planned

- [ ] Alternative split model training
- [ ] Comparison of random and scaffold split performance
- [ ] Dashboard integration of split comparisons
- [ ] Final shared model and checkpoint selection
- [ ] Explanation method evaluation — RQ2
- [ ] Molecular attribution analysis — RQ3

## 16. Technologies

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Git and GitHub

## 17. References

- Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. Journal of Chemical Information and Computer Sciences, 44(3), 1000–1005.
- Kipf, T. N., and Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
- Hamilton, W. L., Ying, R., and Leskovec, J. (2017). Inductive Representation Learning on Large Graphs.
- Veličković, P., et al. (2018). Graph Attention Networks.
- Xu, K., et al. (2019). How Powerful are Graph Neural Networks?
- RDKit Documentation: https://www.rdkit.org/docs/
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

---

University Visual Analytics Project — Understanding Molecular Predictions

Current milestone: RQ1 and model performance investigation.
