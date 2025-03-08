#!/usr/bin/env python3
import argparse
import yaml
import torch
import os
import datetime
from Calcium2Behavior.data_loader import prepare_dataloaders
from Calcium2Behavior.trainer import train_model, test_model
from Calcium2Behavior.utils import save_results, plot_predictions

def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavior prediction using calcium imaging data"
    )
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device(config.get('device', 'cpu'))

    # Get dataset base name from config (assumes a key "dataset" exists; if not, defaults to "dataset")
    dataset_path = config.get("experiment", "Unknown")
    dataset_base = os.path.basename(dataset_path)
    # dataset_base = os.path.splitext(dataset_base)[0]

    # Create result directory with dataset base name and timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(config['output']['dir_path'], f"{dataset_base}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Create dataloaders
    dataloaders, data_specs = prepare_dataloaders(config)

    fold_results = {}
    for fold, (train_loader, test_loader, all_loader) in enumerate(dataloaders):
        print(f"Training Fold {fold}")

        # Train the model
        model, train_loss = train_model(train_loader, data_specs, config, device)

        # Test the model
        _, _, metric = test_model(test_loader, model, data_specs, device, config)
        fold_results[f'fold_{fold}_metric'] = metric

        predictions, ground_truth, _ = test_model(all_loader, model, data_specs, device, config)

        # Save results in the newly created result directory
        save_results(result_dir, fold, predictions, ground_truth, train_loss)

        # Plot results and save them in the result directory
        plot_predictions(result_dir, fold, predictions, ground_truth)

    # Save overall results and the used config into the result directory
    with open(os.path.join(result_dir, 'results.yaml'), 'w') as f:
        yaml.dump(fold_results, f)

    with open(os.path.join(result_dir, 'used_config.yaml'), 'w') as file:
        yaml.dump(config, file)

if __name__ == "__main__":
    main()
