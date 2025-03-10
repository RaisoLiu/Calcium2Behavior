#!/bin/bash

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <0-raw-xlsx_directory> <1-neuron> <1-neuron_shuffle> <1-location> <2-class> <3-transition> <4-transition_direction>"
    exit 1
fi

input_dir="$1"
output_neuron_dir="$2"
output_neuron_shuffle_dir="$3"
output_location_dir="$4"
output_class_dir="$5"
output_transition_dir="$6"
output_transition_direction="$7"

echo "input_dir $input_dir"
echo "output_neuron_dir $output_neuron_dir"
echo "output_neuron_shuffle_dir $output_neuron_shuffle_dir"
echo "output_location_dir $output_location_dir"
echo "output_class_dir $output_class_dir"
echo "output_transition_dir $output_transition_dir"
echo "output_transition_direction $output_transition_direction"





# Create output directory if it doesn't exist
mkdir -p "$output_neuron_dir"
mkdir -p "$output_neuron_shuffle_dir"
mkdir -p "$output_location_dir"
mkdir -p "$output_class_dir"
mkdir -p "$output_transition_dir"
mkdir -p "$output_transition_direction"

# Process each file matching *.state.npy in the input directory
for file in "$input_dir"/*.xlsx; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        python scripts/convert_xlsx_to_neuron_and_location_npy.py -i "$file" --neuron_dir "$output_neuron_dir" --location_dir "$output_location_dir"
    fi
done

for file in "$output_location_dir"/*.npy; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        python scripts/convert_to_state_class.py -i "$file" -o "$output_class_dir"
    fi
done

for file in "$output_neuron_dir"/*.npy; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        python scripts/generate_shuffle_neuron.py -i "$file" -o "$output_neuron_shuffle_dir"
    fi
done


for file in "$output_class_dir"/*.npy; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        python scripts/convert_to_event_probility.py -i "$file" -o "$output_transition_dir"  --fps 1 --cutoff 0.08
    fi
done



for file in "$output_class_dir"/*.npy; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        python scripts/convert_to_event_probility_with_direction.py -i "$file" -o "$output_transition_direction"  --fps 1 --cutoff 0.08
    fi
done