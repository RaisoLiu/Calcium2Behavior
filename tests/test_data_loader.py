import numpy as np
import os
import torch
import pytest
from Calcium2Behavior.data_loader import prepare_dataloaders

def create_dummy_files(tmp_path, T, F):
    """
    Create dummy numpy data files for testing.
    
    Args:
        tmp_path (Path): Temporary directory provided by pytest's tmp_path fixture.
        T (int): Number of time steps.
        F (int): Number of features.
        
    Returns:
        tuple: paths to the data file and label file as strings, along with the created data and labels arrays.
    """
    # Create dummy data: a (T, F) array and labels: a (T,) array.
    data = np.arange(T * F).reshape(T, F)
    labels = np.arange(T)
    data_file = tmp_path / "data.npy"
    label_file = tmp_path / "labels.npy"
    np.save(data_file, data)
    np.save(label_file, labels)
    return str(data_file), str(label_file), data, labels

def test_window_boundaries_and_folds(tmp_path):
    # Parameters for dummy data and window sizes.
    T = 20
    F = 2
    left_window_size = 5
    right_window_size = 2
    # Expected valid samples = T - left_window_size - right_window_size + 1
    expected_length = T - left_window_size - right_window_size + 1  # 20 - 5 - 2 + 1 = 14
    
    data_file, label_file, data, labels = create_dummy_files(tmp_path, T, F)
    
    # Create a dummy config dictionary.
    config = {
        'data': {
            'npy_path': data_file,
            'label_path': label_file,
            'left_window_size': left_window_size,
            'right_window_size': right_window_size
        },
        'training': {
            'batch_size': 4,
            'task_type': 'regression',
            'num_folds': 5,
            # Dummy training parameters
            'total_epochs': 1,
            'hidden_dim': 10,
            'learning_rate': 0.001
        },
        'output': {
            'dir_path': str(tmp_path / "results")
        },
        'device': 'cpu'
    }
    
    # Get dataloaders and data specifications.
    dataloaders, data_specs = prepare_dataloaders(config)
    
    # Verify that each fold respects the temporal ordering.
    for fold, (train_loader, test_loader, all_loader) in enumerate(dataloaders):
        # train_loader and test_loader are created from Subset objects.
        train_indices = train_loader.dataset.indices
        test_indices = test_loader.dataset.indices
        
        # Ensure that all indices are within the valid range [0, expected_length - 1].
        assert all(0 <= idx < expected_length for idx in train_indices), f"Fold {fold}: train indices out of range."
        assert all(0 <= idx < expected_length for idx in test_indices), f"Fold {fold}: test indices out of range."
        
        # For time series split, the max index in training should be less than the min index in testing.
        if len(train_indices) > 0 and len(test_indices) > 0:
            assert max(train_indices) < min(test_indices), f"Fold {fold}: training indices are not strictly before test indices."
    
    # Use the "all_loader" from the last fold to verify the overall dataset length.
    dataset = all_loader.dataset  # This is the original CalciumDataset instance.
    assert len(dataset) == expected_length, "Dataset length does not match expected length, boundary data may be included."
    
    # Check one sample boundary to verify window extraction is correct.
    # For dataset index 0, the sample should use data[0:left_window_size] as input
    # and labels[left_window_size : left_window_size + right_window_size] as the target.
    sample_window, sample_target = dataset[0]
    expected_window = torch.tensor(data[0:left_window_size], dtype=torch.float)
    expected_target = torch.tensor(labels[left_window_size:left_window_size + right_window_size], dtype=torch.float)
    
    assert torch.allclose(sample_window, expected_window), "Input window for index 0 is incorrect."
    assert torch.allclose(sample_target, expected_target), "Target window for index 0 is incorrect."
