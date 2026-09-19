# Egg Hatchability Prediction using CHSL

This repository contains a cleaned implementation of the final paper workflow for predicting egg hatchability with Conditional Tabular Generative Adversarial Networks and Hierarchical Supervised Learning (CHSL). The code retains the reported RF, LightGBM, SVR, HSL, and CHSL structures while removing exploratory and legacy fixed-split experiments.

## Method

The cleaned dataset contains 1,737 egg batches and 125 columns: 124 encoded predictors and the `Hatchability` target. The predictors represent 13 original factors grouped using incubation-domain knowledge:

- Set A: animal-related factors;
- Set AE: animal- and environment-related factors;
- Set AH: animal- and hatchery-related factors;
- Set AEH: all animal-, environment-, and hatchery-related factors.

Continuous variables are scaled to `[0, 1]`. Month is already represented by sine and cosine terms, and nominal variables are already one-hot encoded in the distributed cleaned dataset.

Final model performance is evaluated with an **outer five-fold cross-validation** loop using `KFold(n_splits=5, shuffle=True, random_state=42)`. For every outer fold, scalers are fitted only on that fold's real training observations and then applied to its test observations.

HSL is a two-stage stacking regressor:

1. Stage 0 contains Random Forest, SVR, and LightGBM regressors.
2. Stage 1 uses linear regression to combine their predictions.

`StackingRegressor(cv=5)` performs a separate **internal five-fold cross-validation** on the current outer training fold to produce out-of-fold predictions for the stage-1 learner. This internal procedure constructs the meta-learner; it is not the outer model evaluation.

For CHSL, CTGAN is fitted separately inside every outer fold and only on low-hatchability (`Hatchability < 60`) observations from that fold's training partition. It generates 30 synthetic observations, which are transformed with the training-fold scalers and appended only to that training partition. The outer test fold never participates in scaling or CTGAN fitting.

SHAP uses the same outer-fold CHSL training procedure and explains only unseen outer-test-fold predictions. The five sets of out-of-fold explanations are restored to original row order and combined for global and local analysis. Because `StackingRegressor` is not supported by `TreeExplainer`, the workflow uses SHAP's model-agnostic `Explainer`.

## Repository layout

```text
.
|-- data/
|   `-- Hatchability.csv
|-- notebooks/
|   |-- 01_model_evaluation.ipynb
|   `-- 02_shap_analysis.ipynb
|-- outputs/
|   `-- .gitkeep
|-- scripts/
|   |-- check_project.py
|   |-- run_evaluation.py
|   `-- run_shap.py
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- ctgan_augmentation.py
|   |-- data.py
|   |-- evaluation.py
|   `-- models.py
|-- tests/
|   `-- test_data_contract.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Installation

Python 3.10-3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The original notebooks did not record a complete environment lock file. The dependency ranges in `requirements.txt` describe a compatible environment rather than claiming to reproduce an undocumented package snapshot.

## Running the project

Run commands from the repository root.

1. Validate paths, syntax, notebook structure, and the dataset contract:

   ```bash
   python scripts/check_project.py
   python -m unittest discover -s tests
   python scripts/run_evaluation.py --validate-only
   ```

2. Run the final outer five-fold evaluation:

   ```bash
   python scripts/run_evaluation.py
   ```

   This writes per-fold results to `outputs/fold_metrics.csv` and fold-mean results to `outputs/mean_metrics.csv`.

3. Run the out-of-fold CHSL SHAP analysis:

   ```bash
   python scripts/run_shap.py
   ```

   Model-agnostic SHAP can be computationally expensive. For a quicker diagnostic run, the training-fold background can be sampled without changing the fitted CHSL models:

   ```bash
   python scripts/run_shap.py --background-size 100
   ```

   SHAP arrays, importance values, and plots are written under `outputs/shap/`.

The notebooks provide the same two workflows interactively and delegate reusable logic to `src/`.

## Data source

The data originate from Hendrix Genetics and comprise 1,737 batches of 100-166 eggs incubated between 2010 and 2018. The dataset was published with the Supplementary Materials of:

> Bouba, I., Visser, B., Kemp, B., Rodenburg, T. B., & van den Brand, H. (2021). Predicting hatchability of layer breeders and identifying effects of animal related and environmental factors. *Poultry Science, 100*(10), 101394. https://doi.org/10.1016/j.psj.2021.101394

The article and supplementary-material links are available through [NCBI PubMed Central, PMCID: PMC8385447](https://pmc.ncbi.nlm.nih.gov/articles/PMC8385447/).

The included `data/Hatchability.csv` is an unchanged copy of the supplied cleaned dataset. Before public release, repository maintainers should confirm that redistribution complies with the supplementary material's applicable terms.

## Citation

When using the dataset, cite Bouba et al. (2021) using the reference above. When using the CHSL implementation, also cite the associated CHSL paper. Add its final bibliographic metadata and DOI here once available. For a versioned software citation, archive the GitHub release with Zenodo and cite the resulting DOI.

## Reproduction scope

This repository implements the final paper methodology and intentionally excludes the legacy fixed 60/40 split, exploratory CTGAN sample-size trials, debugging cells, unsuccessful `TreeExplainer` calls, and unrelated LIME/permutation-importance experiments. Full numerical reproduction of the published tables was not attempted during repository preparation.
