# utils.py
import numpy as np
import matplotlib.pyplot as plt
import os


def save_results(result_dir, fold, predictions, ground_truth, train_loss):
    np.save(f"{result_dir}/predictions_fold_{fold}.npy", predictions)
    np.save(os.path.join(result_dir, f'ground_truth_fold_{fold}.npy'), ground_truth)
    np.save(os.path.join(result_dir, f'train_loss_fold_{fold}.npy'), train_loss)


def plot_predictions(result_dir, fold, predictions, ground_truth):
    import os
    import matplotlib.pyplot as plt

    n = len(ground_truth)
    # 計算 test 區間 (fold 對應的 20% 區間)
    test_start = int(n * fold * 0.2)
    test_end = int(n * (fold + 1) * 0.2)
    
    plt.figure(figsize=(10, 5))
    
    # 在 test 區間加上淺紅色 (alpha=0.2)
    plt.axvspan(test_start, test_end, facecolor='red', alpha=0.2)
    
    # 在剩餘區間加上青色 (alpha=0.2)
    if test_start > 0:
        plt.axvspan(0, test_start, facecolor='cyan', alpha=0.2)
    if test_end < n:
        plt.axvspan(test_end, n, facecolor='cyan', alpha=0.2)
    
    # 繪製實際的預測與真實值線圖
    plt.plot(ground_truth, label='Ground Truth', alpha=0.5)
    plt.plot(predictions, label='Predictions')
    
    plt.title(f'Fold {fold} Predictions vs Ground Truth')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(result_dir, f'prediction_fold_{fold}.png')
    plt.savefig(plot_path)
    plot_path = os.path.join(result_dir, f'prediction_fold_{fold}.eps')
    plt.savefig(plot_path)
    plt.close()

