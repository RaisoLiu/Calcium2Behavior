import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class CalciumDataset(Dataset):
    def __init__(self, data, labels, left_window_size, right_window_size, task_type):
        """
        Args:
            data (np.array): The calcium imaging data with shape (T, F).
            labels (np.array): The corresponding labels with shape (T,) or (T, ...).
            left_window_size (int): The number of past time steps to include as input.
            right_window_size (int): The number of future time steps to include as target.
            task_type (str): 'regression' or 'classification', determines label type.
        """
        self.data = data
        self.labels = labels
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size
        self.task_type = task_type

        # Only include samples that have a complete left and right window.
        self.length = len(data) - left_window_size - right_window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Adjust index to account for the left window size.
        index = idx + self.left_window_size
        # Input: previous left_window_size data points.
        window = self.data[index - self.left_window_size: index + self.right_window_size + 1]
        # Target: next right_window_size labels.
        label = self.labels[index]

        # Convert to torch tensors.
        window_tensor = torch.tensor(window, dtype=torch.float)
        if self.task_type == 'classification':
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label, dtype=torch.float)
        return window_tensor, label_tensor


def prepare_dataloaders(config):
    """
    Load data and labels from numpy files specified in the config, and create dataloaders for cross-validation.
    
    Returns:
        dataloaders: A list of tuples, each containing (train_loader, test_loader, all_loader) for a fold.
        data_specs: A dictionary containing 'input_dim' and 'output_dim'.
    """
    # Load numpy data and labels.
    data_path = config['data']['npy_path']
    label_path = config['data']['label_path']
    data = np.load(data_path)
    labels = np.load(label_path)
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    left_window_size = config['data']['left_window_size']
    right_window_size = config['data']['right_window_size']

    # Determine the task type; default to regression if not specified.
    task_type = config['training'].get('task_type', config['training'].get('prediction_type', 'regression'))

    # Create the dataset.
    dataset = CalciumDataset(data, labels, left_window_size, right_window_size, task_type)
    # Determine data specifications: input_dim is the number of features per time step.
    if len(data.shape) == 2:
        input_dim = data.shape[1]
    else:
        raise ValueError("Data array shape not supported")

    # For regression, output_dim is the length of the right window.
    # For classification, assume right_window_size is 1 and output_dim is the number of classes.
    if task_type == 'classification':
        output_dim = len(np.unique(labels))
    else:
        output_dim = labels.shape[1]

    data_specs = {'input_dim': input_dim, 'output_dim': output_dim}

    # Use TimeSeriesSplit for cross-validation to respect temporal order.
    batch_size = config['training'].get('batch_size', 32)
    num_folds = config['training'].get('num_folds', 5)

    # Valid indices from 0 to len(dataset)-1.
    indices = list(range(len(dataset)))
    
    # Split indices into contiguous folds.
    folds = np.array_split(indices, num_folds)
    # Convert each split to a list.
    folds = [list(f) for f in folds]
    

    dataloaders = []
    for test_indices in folds:
        # Training indices: all indices not in test_indices.
        train_indices = [idx for idx in indices if idx not in test_indices]
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        # all_loader includes the entire dataset (useful for evaluation/visualization).
        all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        dataloaders.append((train_loader, test_loader, all_loader))
    
    return dataloaders, data_specs