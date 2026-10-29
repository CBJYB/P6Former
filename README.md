# P⁶Former: A Physics-Guided Transformer with Constraint Routing for Physically Consistent Extreme Climate Prediction

This repository contains the official PyTorch implementation of the paper:
**"P⁶Former: A Physics-Guided Transformer with Constraint Routing for Physically Consistent Extreme Climate Prediction"**

##  Overview
P⁶Former is a novel physics-guided deep learning framework designed for spatiotemporal extreme climate prediction over the Tibetan Plateau. It addresses the "black-box" problem of traditional deep learning models by embedding six-dimensional atmospheric physical constraints. 

Key features include:
- **Constraint Routing Mechanism**: Decouples physical regularization from the main statistical prediction task via an auxiliary output head to avoid gradient conflicts.
- **Deconvolutional Decoder**: Reconstructs local spatial details from the global attention bottleneck, mitigating the spatial smoothing effect.
- **High Physical Consistency**: Drastically reduces thermodynamic and energy residuals compared to purely data-driven models.

## ️ Environment Setup
We recommend using Python >= 3.9. You can set up the environment using `conda` or `venv`:

```bash
# Create a virtual environment (Optional)
python -m venv venv_py311
source venv_py311/bin/activate  # On Windows use: venv_py311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
(Note: cartopy might require manual installation of its binary dependencies depending on your OS. Please refer to the official Cartopy documentation if pip fails.)

 Data Preparation
Due to the large size of the datasets, we do not provide the raw data in this repository. Please download the CMFD and ETCCDI indices independently.

1. Downloading CMFD Data
The China Meteorological Forcing Dataset (CMFD) used in this study is Version 2.0 (1951–2024). It can be downloaded from the National Tibetan Plateau Data Center (TPDC):

Download Link: https://www.tpdc.ac.cn/zh-hans/data/e60dfd96-5fd8-493f-beae-e8e5d24dece4

2. Preprocessing Scripts
We provide the data processing scripts in the preprocessing/ directory. The pipeline is as follows:

Preprocessing CMFD & ETCCDI Data:

1merge_and_resample.py: Aligns the spatial resolution of temperature and other variables to the 80×140 grid using average pooling.

2.calculate_extreme_indices.py: Computes the 6 ETCCDI extreme indices (TXx, TNn, CDD, CWD, R95p, SDII) using a 7-day sliding window.

3.samples.py: Constructs the input-output pairs (30 days input, 7 days output) with a 7-day step size.

4.split_dataset.py: Splits the data chronologically into Train (2000-2015), Val (2016-2018), and Test (2019-2024) sets.

Downloading ERA5-Land Data (Optional):
If you need to download the raw ERA5-Land hourly 2m temperature data for verification, use preprocessing/ERA5.py.
Prerequisite: You must have a CDS API key configured (~/.cdsapirc).

After running the scripts, organize your .npy files exactly as follows:

text
your_project_root/
├── data/
│   └── split_7d/
│       ├── X_train.npy, y_train.npy
│       ├── X_val.npy, y_val.npy
│       └── X_test.npy, y_test.npy
 
  Usage
1. Training
To train the P⁶Former model with the default configuration (Physics + Data Augmentation):

bash
python -m model.train
You can modify hyperparameters, physics loss weight (λ), and random seeds in model/config.py.

To automatically train all core models and ablation variants sequentially:

bash
python run_all_33.py --stage core
2. Evaluation
To evaluate the trained models and generate the R² comparison CSV:

bash
# Evaluate main models and ablation models
python eval_r2_all.py

# Evaluate spatial correlation
python eval_spatial_corr.py

# Evaluate R95p stratified error
python eval_r95p_stratify.py
3. Reproducing Figures and Tables
To generate all figures (architecture, scatter plots, spatial distribution, residuals) and tables presented in the paper:

bash
python total.py
All generated figures and tables will be saved in the paper_assets/ directory.

  Citation
If you find this code useful for your research, please consider citing our paper:
(Citation details will be updated upon publication)

Data Citation
CMFD v2.0:

He, J., Yang, K., Li, X., Tang, W., Shao, C., Jiang, Y., & Ding, B. (2024). China meteorological forcing dataset v2.0 (1951-2024). National Tibetan Plateau / Third Pole Environment Data Center. https://doi.org/10.11888/Atmos.tpdc.302088

  Contact
For any questions or issues, please open an issue in this repository or contact the authors.