import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import yaml

# 确保可以导入同目录下的模块
import file_loader
import models
import pre_data
from attention import Attention, SimpleAttention
from plot import visualization_plot

if __name__ == '__main__':
    # --- 1. 加载配置文件 ---
    config_path = 'config.yaml'
    print(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 2. 从配置中获取所有参数 ---
    # 必需参数
    model_path = config["model_path"]
    test_image_folder = config["test_image_folder"]
    
    # 预处理参数（带默认值）
    grid_size_preproc = tuple(config.get("grid_size", [16, 16]))
    print(f"Using grid size from config.yaml: {grid_size_preproc}")

    # 模型参数
    att_lstm_num = config["att_lstm_num"]
    long_term_lstm_seq_len = config["long_term_lstm_seq_len"]
    short_term_lstm_seq_len = config["short_term_lstm_seq_len"]
    nbhd_size = config["nbhd_size"]
    cnn_nbhd_size = config["cnn_nbhd_size"]
    hist_feature_daynum = config.get("hist_feature_daynum", 7)
    last_feature_num = config.get("last_feature_num", 48)

    # --- 3. 预处理测试图片 ---
    print(f"\nPreprocessing images from folder: {test_image_folder}")
    if not os.path.isdir(test_image_folder):
        raise FileNotFoundError(f"Test image folder not found: {test_image_folder}")

    new_volume_data, new_flow_data = pre_data.process_image_sequence(
        test_image_folder,
        grid_size=grid_size_preproc,
        save_demo=False
    )

    if new_volume_data.size == 0:
        raise ValueError(f"Preprocessing failed: No volume data generated from {test_image_folder}")

    # 处理空光流数据的情况
    if new_flow_data.size == 0 and new_volume_data.shape[0] > 1:
        print("Warning: No flow data generated. Filling with zeros...")
        h_flow, w_flow = new_volume_data.shape[1:3]
        num_flow_expected = new_volume_data.shape[0] - 1
        new_flow_data = np.zeros((num_flow_expected, h_flow, w_flow, 4), dtype=np.float32)

    # 数据归一化
    print("Applying min-max scaling...")
    new_volume_data = pre_data.min_max_scale(new_volume_data)
    if new_flow_data.size > 0:
        new_flow_data = pre_data.min_max_scale(new_flow_data)

    print(f"Preprocessed volume shape: {new_volume_data.shape}")
    print(f"Preprocessed flow shape: {new_flow_data.shape if new_flow_data.size > 0 else 'N/A'}")

    # --- 4. 初始化数据加载器 ---
    print("\nCreating file_loader instance...")
    sampler = file_loader.file_loader(config_path=config_path)
    
    # 注入预处理数据
    sampler.volume_test = new_volume_data
    sampler.flow_test = new_flow_data
    sampler.isVolumeLoaded = True
    sampler.isFlowLoaded = True

    # --- 5. 加载模型 ---
    print(f"\nLoading model from: {model_path}")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")

    model = load_model(
        model_path,
        custom_objects={"Attention": Attention, "SimpleAttention": SimpleAttention},
        compile=False
    )

    model.summary()

    # --- 6. 数据采样 ---
    print("\nSampling data for inference...")
    try:
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
            datatype="test",
            att_lstm_num=att_lstm_num,
            long_term_lstm_seq_len=long_term_lstm_seq_len,
            short_term_lstm_seq_len=short_term_lstm_seq_len,
            nbhd_size=nbhd_size,
            cnn_nbhd_size=cnn_nbhd_size,
            hist_feature_daynum=hist_feature_daynum,
            last_feature_num=last_feature_num
        )
    except ValueError as e:
        print(f"Sampling Error: {e}")
        min_steps = (hist_feature_daynum + att_lstm_num) * sampler.timeslot_daynum + long_term_lstm_seq_len + 1
        print(f"Volume data steps: {sampler.volume_test.shape[0]}, Required minimum: {min_steps}")
        exit(1)

    # --- 7. 执行推理 ---
    print("\nStarting inference...")
    model_input = att_cnnx + att_flow + att_x + cnnx + flow + [x, ]
    y_pred = model.predict(model_input)
    print(f"Prediction shape: {y_pred.shape}")

    # --- 8. 结果可视化 ---
    print("\nGenerating visualizations...")
    visualization_plot(y, y_pred, sampler, num_samples_to_plot=min(300, y.shape[0]))
    print("Script completed successfully.")