# generate_config.py
import os
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate YAML config for behavior prediction.')
    parser.add_argument('--npy_folder', required=True, help='Path to folder containing npy data files.')
    parser.add_argument('--label_folder', required=True, help='Path to folder containing label npy files.')
    parser.add_argument('--output_folder', required=True, help='Directory to save config YAML files.')
    parser.add_argument('--left_window_size', default=50, type=int, help='Number of past time steps as input.')
    parser.add_argument('--right_window_size', default=49, type=int, help='Number of future time steps to predict.')
    parser.add_argument('--total_epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--task_type', default='regression', choices=['regression', 'classification'])
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()

    npy_files = sorted([f for f in os.listdir(args.npy_folder) if f.endswith('.neuron.npy')])

    for npy_file in npy_files:
        experiment_name = os.path.splitext(npy_file)[0]
        experiment_name = experiment_name.split('.neuron')[0]
        label_file = experiment_name + '_dir_event.npy'

        npy_path = os.path.join(args.npy_folder, npy_file)
        label_path = os.path.join(args.label_folder, label_file)

        if not os.path.exists(npy_path):
            print(f"Warning: Data file {npy_path} does not exist. Skipping.")
            continue

        if not os.path.exists(label_path):
            print(f"Warning: Label file {label_path} does not exist. Skipping.")
            continue

        config = {
            'experiment' : experiment_name,
            'data': {
                'npy_path': npy_path,
                'label_path': label_path,
                'left_window_size': args.left_window_size,
                'right_window_size': args.right_window_size
            },
            'training': {
                'total_epochs': args.total_epochs,
                'learning_rate': args.learning_rate,
                'hidden_dim': args.hidden_dim,
                'batch_size': args.batch_size,
                'task_type': args.task_type,
                'num_folds': 5
            },
            'output': {
                'dir_path': args.output_dir
            },
            'device': args.device
        }

        output_config_path = os.path.join(args.output_folder, f'{experiment_name}_config.yaml')
        with open(output_config_path, 'w') as file:
            yaml.dump(config, file)
        print(f"Generated config: {output_config_path}")


if __name__ == '__main__':
    main()
