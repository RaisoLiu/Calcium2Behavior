# Calcium2Behavior

## Overview
This project uses calcium imaging data to predict behavioral outcomes with PyTorch. The model supports both regression and classification tasks based on GRU neural network architecture.

## Setup

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
## Configuration
Edit `config.yaml` to set parameters. For example:

```yaml
data:
  npy_path: "./data/dataset.npy"
  label_path: "./data/labels.npy"
  left_window_size: 10      # Number of past time steps to use as input
  right_window_size: 1      # Number of future time steps to predict

training:
  total_epochs: 100
  learning_rate: 0.001
  hidden_dim: 64
  batch_size: 32
  task_type: "regression"   # or "classification"
  num_folds: 5              # Number of splits for time series cross-validation

output:
  dir_path: "./results"
  
device: "cuda"              # or "cpu"
```

## Usage
Run the training and evaluation:
```bash
python main.py --config config.yaml
```

## Output
Results including models, predictions, metrics, and plots will be saved under the specified output directory, structured by timestamp.
