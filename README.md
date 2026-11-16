```
# P⁶Former: A Constraint Routing Framework for Physically Consistent Extreme Climate Prediction over the Tibetan Plateau

This repository contains the official PyTorch implementation of the paper:
**"P⁶Former: A Constraint Routing Framework for Physically Consistent Extreme Climate Prediction over the Tibetan Plateau"**

## Overview
P⁶Former is a physics-guided deep learning framework designed for spatiotemporal extreme climate prediction over the Tibetan Plateau. It addresses the "black-box" problem of traditional deep learning models by introducing the **Constraint Routing** computational paradigm, which decouples physical regularization from the main statistical prediction task to avoid gradient conflicts and target mismatch.

Key features include:
- **Constraint Routing Mechanism**: Routes physical soft constraints to an auxiliary meteorological field reconstruction head.
- **High Physical Consistency**: Drastically reduces thermodynamic and energy residuals evaluated in real atmospheric physical units compared to purely data-driven models.

## Environment Setup
We recommend using Python >= 3.9. You can set up the environment using `conda` or `venv`:
```bash
# Create a virtual environment (Optional)
python -m venv venv_py311

# Activate
# On Linux/Mac:
source venv_py311/bin/activate
# On Windows:
venv_py311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> 
> Note: cartopy might require manual installation of its binary dependencies depending on your OS. Please refer to the official Cartopy documentation if pip fails.

## Quick Start with Dummy Data (Reproducibility Check)

To ensure full reproducibility without downloading the full CMFD/ERA5 datasets, we provide a fully executable synthetic dummy dataset in the `data/dummy/` directory.

> 
> ⚠️ The dummy data is only for environment verification and pipeline debugging, **cannot reproduce the quantitative results in our paper**.

You can verify the end-to-end training and evaluation pipeline by running:

```
python -m model.train --dummy
```

This will execute a minimal 1-epoch training run on the synthetic data and generate a basic checkpoint, confirming that your environment is correctly set up and the code is fully executable.

## 1. Downloading CMFD Data

The China Meteorological Forcing Dataset (CMFD) used in this study is Version 2.0 (1951–2024). It can be downloaded from the National Tibetan Plateau Data Center (TPDC) at [https://doi.org/10.11888/Atmos.tpdc.302088](https://doi.org/10.11888/Atmos.tpdc.302088).

## 2. Preprocessing Scripts

We provide the data processing scripts in the `preprocessing/` directory. The pipeline is as follows:

- `merge_and_resample.py`: Aligns the spatial resolution of temperature and other variables to the 80×140 grid using average pooling.
- `calculate_extreme_indices.py`: Computes the 6 ETCCDI extreme indices (TXx, TNn, CDD, CWD, R95p, SDII) using a 7-day sliding window.
- `samples.py`: Constructs the input-output pairs (30 days input, 7 days output) with a 7-day step size.
- `split_dataset.py`: Splits the data chronologically into Train (2000–2015), Val (2016–2018), and Test (2019–2024) sets.

(Optional) If you need to download the raw ERA5-Land hourly 2m temperature data for verification, use `preprocessing/ERA5.py`.
Prerequisite: You must have a CDS API key configured (`~/.cdsapirc`).

After running the scripts, organize your `.npy` files exactly as follows:

```
your_project_root/
├── data/
│   └── split_7d/
│       ├── X_train.npy, y_train.npy
│       ├── X_val.npy, y_val.npy
│       └── X_test.npy, y_test.npy
```

## Usage

### 1. Training

To train the P⁶Former model with the default configuration (Physics + Data Augmentation):

```
python -m model.train
```

Input: `.npy` files from `data/split_7d/`.
Output: Best model checkpoints will be saved in the `checkpoints_7d/` directory. You can modify hyperparameters, physics loss weight (`lambda`), and random seeds in `model/config.py`.

To automatically train all core models and ablation variants sequentially:

```
python run_all_33.py --stage core
```

### 2. Evaluation

To evaluate the trained models and generate the R² comparison CSV:

```
# Evaluate main models and ablation models
python eval_r2_all.py

# Evaluate spatial correlation
python eval_spatial_corr.py

# Evaluate R95p stratified error
python eval_r95p_stratify.py
```

Input: Trained checkpoints and test data.
Output: Evaluation metrics (CSV files) and physical residual tables.

### 3. Reproducing Figures and Tables

To generate all figures (architecture, scatter plots, spatial distribution, residuals) and tables presented in the paper:

```
python total.py
```

Output: All generated figures and tables will be saved in the `paper_assets/` directory.

## Citation

If you find this code useful for your research, please consider citing our paper:

> 
> Citation details will be updated upon publication

## Contact

For any questions or issues, please open an issue in this repository or contact the authors.