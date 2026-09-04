# SAFL Work Trial - Dataset Cartography

This repository contains the code for reproducing and extending the paper ["Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics" (Swayamdipta et al., 2020)](https://arxiv.org/abs/2009.10795).

## Directory Structure
- `Reproduction/`: Contains the script to reproduce the core cartography map on the SST-2 dataset using `roberta-base`.
- `requirements.txt`: Python dependencies.

## Setup Instructions

1. **Create a virtual environment** (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   ```
2. **Activate the environment**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you are using a GPU, you may need to install the CUDA-specific version of PyTorch first. See [pytorch.org](https://pytorch.org/).)*

## Running the Reproduction
To run the reproduction script and generate the cartography map:
```bash
python Reproduction/reproduction.py
```
This will train `roberta-base` for 5 epochs on SST-2, record the training dynamics, and output:
- `cartography_metrics.csv`: The confidence and variability metrics for every training example.
- `reproduction_map.png`: The visual scatter plot of the dataset.
