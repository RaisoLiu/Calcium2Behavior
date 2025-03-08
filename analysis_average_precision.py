#!/usr/bin/env python3
import os
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_peak_metrics(gt_peaks, pred_peaks, tolerance=10):
    """
    Calculate metrics comparing two sets of peaks.
    Returns a dictionary with:
        - TP: True Positives
        - FP: False Positives
        - FN: False Negatives
        - precision: Precision value
        - recall: Recall value
        - f1: F1 score
    """
    TP = 0
    FP = 0
    FN = 0
    matched_pred_indices = set()
    
    # For each ground truth peak, try to find a matching predicted peak within the tolerance
    for gt in gt_peaks:
        matched = False
        for idx, pred in enumerate(pred_peaks):
            if idx not in matched_pred_indices and abs(gt - pred) <= tolerance:
                TP += 1
                matched_pred_indices.add(idx)
                matched = True
                break
        if not matched:
            FN += 1

    FP = len(pred_peaks) - len(matched_pred_indices)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def compute_ap(precision_array, recall_array):
    """
    Compute Average Precision (AP).

    Assumes precision_array and recall_array are numpy arrays of the same length,
    with recall_array in increasing order.

    AP is defined as:
        AP = sum((recall[i] - recall[i-1]) * precision[i])
    
    If the first value of recall_array is not 0, a 0 is prepended.
    """
    if recall_array[0] != 0:
        recall_array = np.concatenate(([0], recall_array))
        precision_array = np.concatenate(([precision_array[0]], precision_array))
    dR = np.diff(recall_array)
    ap = np.sum(dR * precision_array[1:])
    return ap

def plot_precision_recall(recall_list, precision_list, ap_value, fold, output_path):
    """
    Plot the Precision-Recall curve based on the provided recall and precision lists,
    and save the plot to output_path.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Fold {fold}), AP: {ap_value:.4f}')
    plt.grid(True)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.savefig(output_path + '.png')
    plt.savefig(output_path + '.eps')
    plt.close()

def process_fold(folder, fold, total_folds, reverse='False', tolerance=10):
    """
    Process a single fold:
    1. Load the corresponding .npy files and slice the data for this fold.
    2. Compute precision and recall at different thresholds.
    3. Calculate Average Precision and save the Precision-Recall curve.
    """
    pred_file = os.path.join(folder, f'predictions_fold_{fold}.npy')
    gt_file = os.path.join(folder, f'ground_truth_fold_{fold}.npy')
    
    if not os.path.exists(pred_file) or not os.path.exists(gt_file):
        print(f"File {pred_file} or {gt_file} does not exist, skipping fold {fold}")
        return None

    # Load the .npy files; assume the first column contains the primary values
    prediction = np.load(pred_file)[:, 0]
    ground_truth = np.load(gt_file)[:, 0]

    if reverse == 'True':
        prediction = -prediction
        ground_truth = -ground_truth

    n_samples = len(ground_truth)
    start_idx = int(n_samples * fold / total_folds)
    end_idx = int(n_samples * (fold + 1) / total_folds)
    prediction = prediction[start_idx:end_idx]
    ground_truth = ground_truth[start_idx:end_idx]

    # Find peaks in the ground truth data (using tolerance as the minimum distance)
    gt_peaks = find_peaks(ground_truth, distance=tolerance)[0]

    precision_list = []
    recall_list = []

    # Evaluate at different thresholds
    for th in np.arange(0.01, 1, 0.01):
        pred_peaks = find_peaks(prediction, distance=tolerance, height=(1 - th))[0]
        if len(pred_peaks) == 0:
            continue
        metrics = calculate_peak_metrics(gt_peaks, pred_peaks, tolerance=tolerance)
        precision_list.append(metrics['precision'])
        recall_list.append(metrics['recall'])

    if len(precision_list) == 0 or len(recall_list) == 0:
        print(f"No valid precision/recall data for fold {fold}, skipping.")
        return {
            'average_precision': 0,
        }

    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)
    ap_value = compute_ap(precision_array, recall_array)

    # Plot and save the Precision-Recall curve
    isinv = 'inv_' if reverse == 'True' else ''
    pr_curve_path = os.path.join(folder, f'{isinv}precision_recall_fold_{fold}')
    plot_precision_recall(recall_array, precision_array, ap_value, fold, pr_curve_path)

    return {
        'average_precision': float(ap_value),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Average Precision over 5-Folds and plot Precision-Recall Curves"
    )
    parser.add_argument("folder", type=str, help="Folder containing result.yaml and .npy files")
    parser.add_argument("--reverse", default='False', type=str, help="Inverse Score to calcualte the other side ap.")
    args = parser.parse_args()
    folder = args.folder

    result_yaml_path = os.path.join(folder, "results.yaml")
    if not os.path.exists(result_yaml_path):
        print(f"results.yaml not found in {folder}")
        return

    # Load result.yaml
    with open(result_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    total_folds = 5  # Fixed to 5 folds

    ap_results = {}

    # Process each fold
    isinv = 'inv_' if args.reverse == 'True' else ''
    for fold in range(total_folds):
        print(f"Processing fold {fold} ...")
        fold_result = process_fold(folder, fold, total_folds, args.reverse)
        if fold_result is not None:
            ap_results[f"{isinv}fold_{fold}"] = fold_result
        else:
            ap_results[f"{isinv}fold_{fold}"] = "Error or no data"

    # Update the YAML with AP results under the key "AP_results"
    data[f'{isinv}AP_results'] = ap_results
    with open(result_yaml_path, 'w') as f:
        yaml.safe_dump(data, f)

    print("All folds processed. AP results updated in result.yaml.")

if __name__ == "__main__":
    main()
