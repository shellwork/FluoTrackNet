import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from file_loader import file_loader
from attention import Attention, SimpleAttention

def inference_and_visualize_npy(
    model_path,
    config_path="config.yaml",
    att_lstm_num=3,
    long_term_lstm_seq_len=3,
    short_term_lstm_seq_len=7,
    nbhd_size=1,
    cnn_nbhd_size=3,
    num_samples_to_plot=300
):
    """
    Using the saved STDN model + .npz data to run inference and visualize results.
      - Plot (and save) a time-series comparison chart (ground truth vs. prediction).
      - Plot (and save) a heatmap of the second-to-last frame's prediction.
    """

    # Ensure the directory for plots exists
    os.makedirs("./plot", exist_ok=True)

    # 1) Instantiate file_loader and retrieve the test set
    sampler = file_loader(config_path=config_path)

    # Get the 7 outputs from sample_stdn(...):
    #   (att_cnnx, att_flow, att_x, cnnx, flow, x, y)
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
        datatype="test",
        att_lstm_num=att_lstm_num,
        long_term_lstm_seq_len=long_term_lstm_seq_len,
        short_term_lstm_seq_len=short_term_lstm_seq_len,
        nbhd_size=nbhd_size,
        cnn_nbhd_size=cnn_nbhd_size
    )

    print("Sampling done. Shapes info:")
    print(f"  - att_cnnx length = {len(att_cnnx)}; shape example = {att_cnnx[0].shape}")
    print(f"  - att_flow length = {len(att_flow)}; shape example = {att_flow[0].shape}")
    print(f"  - att_x length    = {len(att_x)};   shape example = {att_x[0].shape}")
    print(f"  - cnnx length     = {len(cnnx)};    shape example = {cnnx[0].shape}")
    print(f"  - flow length     = {len(flow)};    shape example = {flow[0].shape}")
    print("  - x.shape =", x.shape)
    print("  - y.shape =", y.shape)

    # 2) Load model with custom_objects
    model = load_model(
        model_path,
        custom_objects={"Attention": Attention, "SimpleAttention": SimpleAttention}
    )
    print("\nModel loaded successfully:", model_path)

    # 3) Run inference
    model_inputs = att_cnnx + att_flow + att_x + cnnx + flow + [x]
    y_pred = model.predict(model_inputs)
    print("Inference done. y_pred.shape =", y_pred.shape)

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


if __name__ == '__main__':
    model_path = "model_output/bioCV_STDN20250304184041.keras"
    config_path = "config.yaml"

    # Must match training settings
    att_lstm_num = 2
    long_term_lstm_seq_len = 3
    short_term_lstm_seq_len = 4
    nbhd_size = 2
    cnn_nbhd_size = 3

    inference_and_visualize_npy(
        model_path=model_path,
        config_path=config_path,
        att_lstm_num=att_lstm_num,
        long_term_lstm_seq_len=long_term_lstm_seq_len,
        short_term_lstm_seq_len=short_term_lstm_seq_len,
        nbhd_size=nbhd_size,
        cnn_nbhd_size=cnn_nbhd_size,
        num_samples_to_plot=100
    )
