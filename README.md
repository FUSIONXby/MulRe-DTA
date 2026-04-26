# Drug-Target Affinity (DTA) Prediction with Graph Neural Networks

This project implements a Drug-Target Affinity (DTA) prediction system leveraging Graph Neural Networks (GNNs) and advanced graph augmentation techniques. It is designed to predict the binding affinity between drug molecules and target proteins, a crucial step in drug discovery.

## Features

-   **Graph Neural Network (GNN) Model:** Utilizes Graph Convolutional Networks (GCNs) for learning representations of drug molecules and target proteins.
-   **Cheminformatics-Guided Graph Augmentation:** Employs sophisticated augmentation strategies based on RDKit and NetworkX to enhance molecular graph representations by considering topological centrality and chemical properties (e.g., QED).
-   **Configurable Training:** Training parameters and model architecture can be easily configured via a `config.yaml` file.
-   **Early Stopping:** Integrates early stopping to prevent overfitting and optimize model performance.
-   **Evaluation Metrics:** Evaluates model performance using standard metrics such as Mean Squared Error (MSE) and R-squared.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Aload2
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n dta_gnn python=3.8
    conda activate dta_gnn
    ```
    *(Note: Python 3.7 or 3.8 is typically compatible with older PyTorch Geometric versions if issues arise.)*

3.  **Install PyTorch and PyTorch Geometric:**
    Install PyTorch first, then PyTorch Geometric according to your CUDA version. Refer to the official PyTorch and PyTorch Geometric documentation for the exact commands.

    Example for CUDA 11.3:
    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
    ```
    *(Adjust versions as needed based on your system and `config.yaml` requirements)*

4.  **Install other dependencies:**
    ```bash
    pip install pyyaml numpy pandas scikit-learn rdkit networkx tqdm
    ```
    *(Note: RDKit installation might be easier via Conda: `conda install -c conda-forge rdkit`)*

## Requirements

-   Python 3.8+
-   `torch` (e.g., `torch==1.10.0+cu113`)
-   `torch_geometric` (e.g., `torch-geometric`)
-   `pyyaml==6.0.1`
-   `numpy==1.23.5`
-   `pandas==1.5.3`
-   `scikit-learn==1.2.2`
-   `rdkit==2023.09.6`
-   `networkx==3.0`
-   `tqdm==4.66.1`

## Usage

### 1. Data Preparation

The project expects data in a specific structure, typically under a `data/` directory. The `davis` dataset is configured by default. Ensure your dataset files (e.g., `S1_test_set.txt`, `S1_train_set.txt`, `affinities`, and preprocessed drug/target graphs in `.pt` format) are correctly placed.

### 2. Configuration

Adjust the training parameters and model architecture by editing the `config.yaml` file:

```yaml
# Environment settings
env:
  cuda: 0             # CUDA device ID
  seed: 42            # Random seed for reproducibility
  run_id: "davis_mix_v1" # Identifier for the current run

# Path configurations
paths:
  dataset: "davis"    # Name of the dataset (e.g., "davis")
  data_root: "../data/" # Root directory for data
  log_xlsx: "training_noval.xlsx" # Log file for training results
  save_dir: "models/" # Directory to save trained models

# Training hyperparameters
train:
  epochs: 2200
  batch_size: 512
  lr: 0.0002
  pos_threshold: 7.0  # Threshold for positive affinity
  patience: 15        # Early stopping patience
  val_interval: 25    # Validation interval (epochs)

# Model architecture parameters
model:
  tau: 0.8
  lam: 0.5
  edge_dropout: 0.2
  embedding_dim: 128
  ns_dims: [null, 512, 256]        # Node feature dimensions (first element dynamic)
  drug_ms_dims: [78, 78, 156, 256] # Drug multi-scale feature dimensions
  target_ms_dims: [54, 54, 108, 256] # Target multi-scale feature dimensions
  cuda: 0 # CUDA device ID for model
```

### 3. Training

```bash
python Amain.py
```

### 4. Prediction (or Validation)

```bash
python Amain_val.py
```
*(Note: Based on file names, `Amain.py` seems to handle both training and testing. `Amain_val.py` might be for specific validation scenarios or a slightly different workflow.)*

## Project Structure

-   `Amain.py`: Main script for training and evaluating the DTA prediction model.
-   `Amain_val.py`: (Potentially) an alternative main script for validation or a slightly different workflow.
-   `config.yaml`: Configuration file for environment settings, paths, training hyperparameters, and model architecture.
-   `agumentation/`: Contains modules for graph augmentation.
    -   `graph_augmentation.py`: Implements cheminformatics-guided graph augmentation techniques (e.g., topological centrality, QED-based constraints).
-   `data/`: Directory for datasets.
    -   `davis/`: Contains specific dataset files, including affinity data, test/train splits, and `.aln` files.
    -   `davis/pt/`: Likely contains preprocessed PyTorch graph data.
-   `data_processing/`: Modules for data loading and preprocessing.
-   `models/`: Defines the GNN model architecture and prediction heads.
-   `training/`: Contains the `Trainer` class and early stopping logic.
-   `utils/`: Utility functions (metrics, helpers, datasets).

## Contributing

(Add your contribution guidelines here if applicable.)

## License

(Add your license information here if applicable.)
