#!/bin/bash

# Usage: ./batch_convert.sh <input_directory> <output_directory>
# This script processes all *.position.npy files in the input directory,
# calls the convert_to_state_class.py script for each, and saves the output
# in the output directory with a modified filename.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Create output directory if it does not exist
mkdir -p "$output_dir"

# Process each file matching *.position.npy in the input directory
for file in "$input_dir"/*.position.npy; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Remove the .position.npy suffix to get the base name
        base="${filename%.position.npy}"
        # Set the output file name; you can change the suffix as desired
        output_file="$output_dir/${base}_state.npy"
        echo "Processing $file -> $output_file"
        python convert_to_state_class.py -i "$file" -o "$output_file"
    fi
done
