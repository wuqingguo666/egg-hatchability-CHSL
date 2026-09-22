# Egg Hatchability Prediction using CHSL

This repository provides code and data for research on egg hatchability prediction using CHSL. It includes model evaluation and explainability workflows associated with the paper. The final evaluation uses outer five-fold cross-validation.

## Method

The workflow compares RF, LightGBM, SVR, and hierarchical ensemble learning (HSL) across domain-informed feature sets. CHSL combines HSL with CTGAN-based generative augmentation of low-hatchability training samples. Model performance is assessed with outer five-fold cross-validation; augmentation and preprocessing use only the training data in each fold. SHAP provides explainable AI analysis of CHSL predictions.

## Repository layout

```text
.
|-- data/
|   `-- Hatchability.csv
|-- notebooks/
|   |-- 01_model_evaluation.ipynb
|   `-- 02_shap_analysis.ipynb
|-- outputs/                 # Generated results (not tracked)
|-- scripts/
|   |-- check_project.py
|   |-- run_evaluation.py
|   `-- run_shap.py
|-- src/                     # Data, augmentation, models, evaluation
|-- tests/                   # Data-contract checks
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

The dependency ranges in `requirements.txt` describe a compatible environment; an exact historical environment lock file is not available.

## Running the project

Run commands from the repository root.

1. Check the project and dataset:

   ```bash
   python scripts/check_project.py
   ```

2. Run the outer five-fold model evaluation:

   ```bash
   python scripts/run_evaluation.py
   ```

   Results are written to `outputs/`.

3. Run the CHSL SHAP analysis:

   ```bash
   python scripts/run_shap.py
   ```

   Explainability results are written under `outputs/shap/`. This step can take considerably longer than the project check.

The notebooks in `notebooks/` provide interactive entry points for the same workflows.

## Data source

The data originate from Hendrix Genetics and comprise 1,737 batches of 100-166 eggs incubated between 2010 and 2018. The dataset was published with the Supplementary Materials of:

> Bouba, I., Visser, B., Kemp, B., Rodenburg, T. B., & van den Brand, H. (2021). Predicting hatchability of layer breeders and identifying effects of animal related and environmental factors. *Poultry Science, 100*(10), 101394. https://doi.org/10.1016/j.psj.2021.101394

The article and supplementary-material links are available through [NCBI PubMed Central, PMCID: PMC8385447](https://pmc.ncbi.nlm.nih.gov/articles/PMC8385447/).

The included `data/Hatchability.csv` is an unchanged copy of the supplied cleaned dataset. Users should check the supplementary material's terms before redistributing the data.

## Citation

When using the dataset, cite Bouba et al. (2021) using the reference above. When using the CHSL implementation, also cite the associated CHSL paper; its final bibliographic details and DOI are pending.
