#!/usr/bin/env python3
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert npy file to state classification")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input npy file")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to the output npy file")
    args = parser.parse_args()
    
    input_file = args.input
    if args.output is None:
        input_dir = os.path.dirname(os.path.abspath(input_file))
        output_file = os.path.join(input_dir, "converted.npy")
    else:
        output_file = args.output

    try:
        arr = np.load(input_file)
    except Exception as e:
        raise RuntimeError(f"Unable to load file {input_file}: {e}")

    if arr.ndim != 1:
        raise ValueError("Input npy file must be a 1D array")

    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input npy file must contain only integers")
    if not np.all((arr >= 1) & (arr <= 32)):
        raise ValueError("Input npy file contains values outside the range 1-32")

    state_arr = np.zeros_like(arr)
    state1_mask = ((arr >= 9) & (arr <= 16)) | ((arr >= 25) & (arr <= 32))
    state_arr[state1_mask] = 1

    np.save(output_file, state_arr)
    print(f"Conversion complete. Output file saved at: {output_file}")

if __name__ == '__main__':
    main()
