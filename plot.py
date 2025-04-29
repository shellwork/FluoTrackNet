import matplotlib.pyplot as plt
import os

def visualization_plot(y, y_pred, sampler, num_samples_to_plot=300):
    os.makedirs("./plot", exist_ok=True)
    
    # ========== Visualization 1: Time Series Comparison ==========
    n_plot = min(num_samples_to_plot, y.shape[0], y_pred.shape[0])
    if y.ndim > 1:
        true_vals = y[:n_plot].mean(axis=1)  # 取每个样本所有网格点的均值
        pred_vals = y_pred[:n_plot].mean(axis=1)
    else:
        true_vals = y[:n_plot]
        pred_vals = y_pred[:n_plot]

    plt.figure(figsize=(10, 5))
    plt.plot(true_vals, label="Ground Truth")
    plt.plot(pred_vals, label="Prediction", linestyle="--")
    plt.title(f"Prediction vs. Ground Truth (first {n_plot} samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    time_series_plot_path = "./plot/time_series_compare.png"
    plt.savefig(time_series_plot_path)
    plt.close()
    print(f"Time-series comparison plot saved to: {time_series_plot_path}")

    # ========== Visualization 2: Heatmap of the second-to-last frame ==========
    # We assume each frame has H*W samples, so total_frames = N // (H*W).
    T_test = sampler.volume_test.shape[0]
    H = sampler.volume_test.shape[1]
    W = sampler.volume_test.shape[2]

    N = y_pred.shape[0]
    total_frames = N // (H * W)  # integer division
    print(f"\nWe have total_frames = {total_frames}, from y_pred.shape={y_pred.shape} and H*W={H*W}.")

    # second-to-last frame index
    target_frame = total_frames - 2
    if target_frame < 0:
        print("Warning: No second-to-last frame (total_frames < 2). Abort heatmap.")
        return

    start_idx = target_frame * (H * W)
    end_idx = (target_frame + 1) * (H * W)
    if end_idx > N:
        print("Warning: frame index out of range, cannot plot heatmap.")
        return

    # reshape to (H, W)
    grid_pred = y_pred[start_idx:end_idx, 0].reshape(H, W)

    plt.figure(figsize=(6, 5))
    plt.imshow(grid_pred, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title(f"Predicted Heatmap of Frame #{target_frame} (2nd to last)")
    heatmap_path = "./plot/second_to_last_frame_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap of the second-to-last frame saved to: {heatmap_path}")
