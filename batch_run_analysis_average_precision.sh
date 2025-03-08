#!/bin/bash

# Check if the base folder is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <base_folder>"
  exit 1
fi

base_folder="$1"

# Loop over subdirectories that match the pattern "no.*"
for folder in "$base_folder"/no.*; do
  if [ -d "$folder" ]; then
    echo "Processing folder: $folder"
    # Run the Python analysis script on the folder
    python analysis_average_precision.py "$folder" --reverse True
  fi
done
