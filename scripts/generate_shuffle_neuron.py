#!/usr/bin/env python3
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description="Shuffle the rows (T dimension) of a 2D numpy array (T x F)"
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input npy file containing a 2D array")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to the output npy file with shuffled rows")
    args = parser.parse_args()
    
    input_file = args.input
    output_dir = args.output

    try:
        arr = np.load(input_file)
    except Exception as e:
        raise RuntimeError(f"Unable to load file {input_file}: {e}")

    if arr.ndim != 2:
        raise ValueError("Input npy file must be a 2D array (T x F)")
    
    # Shuffle the rows (T dimension)
    shuffled_arr = arr.copy()
    np.random.shuffle(shuffled_arr)

    basename = os.path.basename(input_file)
    basename = os.path.splitext(basename)[0] + '_shuffle.npy'
    out_path = os.path.join(output_dir, basename)
    np.save(out_path, shuffled_arr)
    print(f"Conversion complete. Output file saved at: {out_path}, shape {shuffled_arr.shape}")

if __name__ == '__main__':
    main()
