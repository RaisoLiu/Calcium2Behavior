#!/bin/bash

CONFIG_FOLDER=$1

if [ -z "$CONFIG_FOLDER" ]; then
    echo "Usage: $0 <config_folder>"
    exit 1
fi

for config_file in "$CONFIG_FOLDER"/*.yaml; do
    echo "Running experiment with config $config_file"
    python main.py --config "$config_file"
done
