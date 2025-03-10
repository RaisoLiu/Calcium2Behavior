#!/usr/bin/env python3
import numpy as np
import argparse
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def noncausal_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def main():
    parser = argparse.ArgumentParser(
        description="Convert state file to event probability signal and plot combined figure"
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input state npy file")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to the output event probability npy file")
    parser.add_argument("--fps", type=float, default=1,
                        help="Frames per second (default: 1)")
    parser.add_argument("--cutoff", type=float, default=0.1,
                        help="Low pass filter cutoff (default: 0.1)")
    args = parser.parse_args()
    
    input_file = args.input
    output_dir = args.output


    # Load the state file
    state = np.load(input_file)
    if state.ndim != 1:
        raise ValueError("Input state file must be a 1D array")
    
    # Create an event signal array (initialize with zeros)
    event_signal = np.zeros_like(state, dtype=np.float32)
    n = len(state)
    
    # Detect transitions: when state changes between consecutive frames,
    # mark both frames as events (set to 1)
    for i in range(n - 1):
        if state[i] < state[i + 1]:
            event_signal[i] = 1
            event_signal[i + 1] = 1
        if state[i] > state[i + 1]:
            event_signal[i] = -1
            event_signal[i + 1] = -1

    # Use the provided fps and cutoff values
    fs = args.fps
    cutoff = args.cutoff
    order = 1
    filtered_signal = noncausal_lowpass_filter(event_signal, cutoff, fs, order)

    # Scale the filtered signal so that its maximum value becomes 0.99
    max_val = np.max(filtered_signal)
    if max_val > 0:
        scaled_signal = filtered_signal / max_val * 0.99
    else:
        scaled_signal = filtered_signal

    # Save the output event probability signal
    basename = os.path.basename(input_file)
    out_path = os.path.join(output_dir, basename)
    np.save(out_path, scaled_signal)
    print(f"Conversion complete. Output file saved at: {out_path}")
    

    
    # Create a combined plot with two subplots
    plt.figure(figsize=(10, 8))
    
    # Top subplot: state signal
    plt.subplot(2, 1, 1)
    plt.plot(state, label="State")
    plt.title("State Signal")
    plt.xlabel("Frame")
    plt.ylabel("State")
    plt.legend()
    
    # Bottom subplot: event probability signal
    plt.subplot(2, 1, 2)
    plt.plot(scaled_signal, label="Event Probability")
    plt.title("Event Probability Signal")
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.legend()
    
    plt.tight_layout()
    
    # Generate the plot file name based on the input file name
    basename = os.path.basename(input_file)
    basename = os.path.splitext(basename)[0] + '_plot.png'
    out_path = os.path.join(output_dir, basename)
    plt.savefig(out_path)
    print(f"Combined plot saved at: {out_path}")

    # Generate the plot file name based on the input file name
    basename = os.path.basename(input_file)
    basename = os.path.splitext(basename)[0] + '_plot.eps'
    out_path = os.path.join(output_dir, basename)
    plt.savefig(out_path)
    print(f"Combined plot saved at: {out_path}")

    plt.close()

if __name__ == "__main__":
    main()
