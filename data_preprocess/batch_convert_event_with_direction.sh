#!/bin/bash

# Usage: ./batch_convert_event.sh <input_directory> <output_directory>
# This script processes all *.state.npy files in the input directory,
# calls the convert_to_event_probility.py script for each, and saves the output
# event probability file and combined plot into the output directory.
#
# Example:
#   ./batch_convert_event.sh /path/to/state_files /path/to/event_outputs

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_dir="$1"
output_dir="$2"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Process each file matching *.state.npy in the input directory
for file in "$input_dir"/*_state.npy; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Remove the .state.npy suffix to get the base name
        base="${filename%_state.npy}"
        # Define the output event file (and the plot will be saved alongside it)
        output_file="$output_dir/${base}_dir_event.npy"
        echo "Processing $file -> $output_file"
        python convert_to_event_probility_with_direction.py -i "$file" -o "$output_file"  --fps 1 --cutoff 0.08
    fi
done
