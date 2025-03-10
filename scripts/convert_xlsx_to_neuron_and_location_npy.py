#!/usr/bin/env python3
import numpy as np
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Convert npy file to state classification")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input xlsx file")
    parser.add_argument("--neuron_dir", type=str,
                        help="Path to the output npy file")
    parser.add_argument("--location_dir", type=str,
                        help="Path to the output npy file")
    args = parser.parse_args()
    
    input_file = args.input
    neuron_dir = args.neuron_dir
    location_dir = args.location_dir

    try:
        df = pd.read_excel(input_file, sheet_name=None, header=1)
    except Exception as e:
        raise RuntimeError(f"Unable to load file {input_file}: {e}")

    for day in df.keys():
        neuron_name = set(df[day].columns)
        neuron_name.remove('time')
        neuron_name.remove('position')
        neuron_name = sorted(list(neuron_name))

        neuron = df[day][neuron_name].to_numpy().astype(np.float32)
        position = df[day]['position'].astype(np.int32)
        

        day = day.replace(' ', '')
        basename = os.path.basename(input_file)
        basename = os.path.splitext(basename)[0] + f'_{day}.npy'
        out_path = os.path.join(neuron_dir, basename)
        np.save(out_path, neuron)
        print(f"Conversion complete. Output file saved at: {out_path}, shape {neuron.shape}")

        out_path = os.path.join(location_dir, basename)
        np.save(out_path, position)
        print(f"Conversion complete. Output file saved at: {out_path}")
        print(basename, np.sum(np.isnan(neuron)), np.sum(np.isnan(position)))

if __name__ == '__main__':
    main()
