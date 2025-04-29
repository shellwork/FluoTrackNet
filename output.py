import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # 显式导入 tensorflow
from tensorflow.keras.models import load_model
import yaml
import argparse # 用于接收命令行参数

# 确保可以导入同目录下的模块
import file_loader # 导入 file_loader.py
import models      # 导入 models.py
import pre_data    # 导入 pre_data.py
from attention import Attention, SimpleAttention # 导入 attention.py
from plot import visualization_plot            # 导入 plot.py

if __name__ == '__main__':
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Run STDN model inference on a custom image folder.")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained .keras model file.')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='Path to the configuration file (config.yaml).')
    parser.add_argument('--test_image_folder', type=str, required=True,
                        help='Path to the folder containing the new test images (.jpg).')
    # 添加 grid_size 参数，如果 config.yaml 中没有的话
    parser.add_argument('--grid_size', type=int, nargs=2, default=None,
                        help='Grid size (rows cols) for preprocessing. Overrides config if provided.')

    args = parser.parse_args()

    # --- 2. 加载配置和模型参数 ---
    print(f"Loading configuration from: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # 从 config 或 args 获取 grid_size
    if args.grid_size:
        grid_size_preproc = tuple(args.grid_size)
        print(f"Using grid size from command line: {grid_size_preproc}")
    elif "grid_size" in config and isinstance(config["grid_size"], list) and len(config["grid_size"]) == 2:
        grid_size_preproc = tuple(config["grid_size"])
        print(f"Using grid size from config.yaml: {grid_size_preproc}")
    else:
        # 如果 config 和命令行都没有，设置一个默认值或报错
        grid_size_preproc = (16, 16) # 或者你可以选择报错
        print(f"Warning: Grid size not found in config or command line. Using default: {grid_size_preproc}")
        # raise ValueError("Grid size must be specified either in config.yaml or via --grid_size argument.")

    # 模型相关参数 (必须与训练时一致)
    att_lstm_num = config["att_lstm_num"]
    long_term_lstm_seq_len = config["long_term_lstm_seq_len"]
    short_term_lstm_seq_len = config["short_term_lstm_seq_len"]
    nbhd_size = config["nbhd_size"]
    cnn_nbhd_size = config["cnn_nbhd_size"]
    # 从 config 获取其他 sample_stdn 可能需要的参数
    hist_feature_daynum = config.get("hist_feature_daynum", 7) # 提供默认值以防万一
    last_feature_num = config.get("last_feature_num", 48)    # 提供默认值以防万一

    # --- 3. 预处理新的测试图片 ---
    print(f"\nPreprocessing images from folder: {args.test_image_folder}")
    if not os.path.isdir(args.test_image_folder):
        raise FileNotFoundError(f"Test image folder not found: {args.test_image_folder}")

    # 调用 pre_data 中的函数处理单个文件夹
    # 注意：如果你的新测试数据包含多个逻辑上的实验序列，应该用 merge_experiments_with_zeros
    # 这里假设 test_image_folder 只包含一个连续的时间序列
    new_volume_data, new_flow_data = pre_data.process_image_sequence(
        args.test_image_folder,
        grid_size=grid_size_preproc,
        save_demo=False # 推理时通常不需要保存光流演示图
    )

    if new_volume_data.size == 0:
         raise ValueError(f"Preprocessing failed: No volume data generated from {args.test_image_folder}")
    if new_flow_data.size == 0 and new_volume_data.shape[0] > 1:
        print(f"Warning: No flow data generated. This might happen if there's only one image.")
        # 如果需要flow数据但没有生成（比如图片少于2张），可能需要填充零值或调整策略
        if new_volume_data.shape[0] > 1:
             h_flow, w_flow = new_volume_data.shape[1:3]
             num_flow_expected = new_volume_data.shape[0] - 1
             new_flow_data = np.zeros((num_flow_expected, h_flow, w_flow, 4), dtype=np.float32)
             print(f"Filled flow data with zeros, shape: {new_flow_data.shape}")


    print("Applying min-max scaling to the new preprocessed data...")
    # 使用 pre_data 中的归一化函数
    new_volume_data = pre_data.min_max_scale(new_volume_data)
    # 只有当 flow 数据存在时才进行归一化
    if new_flow_data.size > 0:
         new_flow_data = pre_data.min_max_scale(new_flow_data)
    else:
         print("Skipping flow data scaling as it's empty or non-existent.")


    print(f"Preprocessed new volume shape: {new_volume_data.shape}")
    print(f"Preprocessed new flow shape: {new_flow_data.shape if new_flow_data.size > 0 else 'N/A'}")

    # --- 4. 创建 file_loader 实例并注入数据 ---
    print("\nCreating file_loader instance...")
    # 传递 config_path 以便 sampler 能读取 timeslot_sec, threshold 等参数
    sampler = file_loader.file_loader(config_path=args.config_path)

    print("Injecting preprocessed data into file_loader...")
    # 直接将预处理好的 numpy 数组赋值给 sampler 的 test 属性
    sampler.volume_test = new_volume_data
    sampler.flow_test = new_flow_data
    sampler.isVolumeLoaded = True # 手动标记为已加载，跳过内部的 load_volume
    sampler.isFlowLoaded = True   # 手动标记为已加载，跳过内部的 load_flow

    # --- 5. 加载模型 ---
    print(f"\nLoading model from: {args.model_path}")
    # 确保 TensorFlow 操作在 GPU 上执行（如果可用）
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # 可根据需要设置内存增长或其他 GPU 选项
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, using CPU.")

    model = load_model(
        args.model_path,
        custom_objects={"Attention": Attention, "SimpleAttention": SimpleAttention}
    )
    print("Model loaded successfully.")
    model.summary() # 打印模型结构

    # --- 6. 使用注入的数据进行采样 ---
    print("\nSampling data using file_loader with the injected test set...")
    # 调用 sample_stdn， datatype="test" 会让它使用我们刚刚注入的 sampler.volume_test 和 sampler.flow_test
    try:
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
            datatype="test", # 关键：指定使用 test 数据集
            att_lstm_num=att_lstm_num,
            long_term_lstm_seq_len=long_term_lstm_seq_len,
            short_term_lstm_seq_len=short_term_lstm_seq_len,
            nbhd_size=nbhd_size,
            cnn_nbhd_size=cnn_nbhd_size,
            hist_feature_daynum=hist_feature_daynum, # 确保传递这些参数
            last_feature_num=last_feature_num      # 确保传递这些参数
        )
    except ValueError as e:
        print(f"\nError during sampling: {e}")
        print("This might be due to insufficient data length after preprocessing.")
        print(f"Required minimum time steps depends on sequence lengths and history settings.")
        print(f"Volume data time steps: {sampler.volume_test.shape[0]}")
        # 根据 sample_stdn 的 time_start 计算所需的最少时间步
        min_required_steps = (hist_feature_daynum + att_lstm_num) * sampler.timeslot_daynum + long_term_lstm_seq_len + 1
        print(f"Estimated minimum required time steps: {min_required_steps} (may vary based on exact logic)")
        exit(1) # 采样失败则退出

    print("Sampling complete.")
    print("Sampled data shapes for inference:")
    print(f"  att_cnnx: list of {len(att_cnnx)} tensors, first tensor shape: {att_cnnx[0].shape if att_cnnx else 'N/A'}")
    print(f"  att_flow: list of {len(att_flow)} tensors, first tensor shape: {att_flow[0].shape if att_flow else 'N/A'}")
    print(f"  att_x (lstm_att_features): list of {len(att_x)} arrays, first array shape: {att_x[0].shape if att_x else 'N/A'}")
    print(f"  cnnx: list of {len(cnnx)} tensors, first tensor shape: {cnnx[0].shape if cnnx else 'N/A'}")
    print(f"  flow (flow_features): list of {len(flow)} tensors, first tensor shape: {flow[0].shape if flow else 'N/A'}")
    print(f"  x (short_term_lstm_features): shape: {x.shape if isinstance(x, np.ndarray) else 'N/A'}")
    print(f"  y (labels): shape: {y.shape if isinstance(y, np.ndarray) else 'N/A'}")

    if y.shape[0] == 0:
        print("\nError: No samples were generated. Check data length and sampling parameters.")
        exit(1)

    # --- 7. 执行模型推理 ---
    print("\nStarting model inference...")
    # 确保输入列表顺序和模型期望的一致
    model_input = att_cnnx + att_flow + att_x + cnnx + flow + [x, ]
    y_pred = model.predict(model_input)
    print("Inference done.")
    print(f"Prediction output (y_pred) shape: {y_pred.shape}")

    # --- 8. 评估和可视化 ---
    threshold = float(sampler.threshold) # 从 sampler 获取阈值
    print(f"\nEvaluating with threshold: {threshold}")
    print(f"Ground truth (y) shape before evaluation: {y.shape}")
    print(f"Predictions (y_pred) shape before evaluation: {y_pred.shape}")

    # 调用可视化函数 (确保 plot.py 中的 visualization_plot 函数可用)
    print("Generating visualization plot...")
    # 你可能需要根据 y 和 y_pred 的实际样本数量调整 num_samples_to_plot
    num_samples = y.shape[0]
    visualization_plot(y, y_pred, sampler, num_samples_to_plot=min(300, num_samples))
    print("Visualization plot generated (if applicable).")
    print("\nScript finished.")