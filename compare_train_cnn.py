#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后的 CNN 模型，用于利用光流数据预测下一时刻每个网格的荧光强度。
在原始结构上增加了网络深度、BatchNormalization 和 Dropout 层，有助于提升模型泛化性能。
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import datetime

# -----------------------
# 参数设置
# -----------------------
parser = argparse.ArgumentParser(description="优化后的CNN光流模型训练")
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
parser.add_argument('--patience', type=int, default=10, help='EarlyStopping等待轮数')
parser.add_argument('--data_folder', type=str, default="data", help='预处理数据保存的文件夹')
parser.add_argument('--model_save_dir', type=str, default="model_output", help='模型保存目录')
args = parser.parse_args()

# -----------------------
# 数据加载
# -----------------------
def load_data(data_folder):
    # 加载预处理后的数据
    volume_train = np.load(os.path.join(data_folder, "volume_train.npz"))["volume"]  # (T_train, H, W)
    flow_train   = np.load(os.path.join(data_folder, "flow_train.npz"))["flow"]      # (T_train-1, H, W, 4)
    volume_test  = np.load(os.path.join(data_folder, "volume_test.npz"))["volume"]    # (T_test, H, W)
    flow_test    = np.load(os.path.join(data_folder, "flow_test.npz"))["flow"]        # (T_test-1, H, W, 4)
    
    # 训练数据：用 flow_train[i] 预测 volume_train[i+1]
    X_train = flow_train
    Y_train = volume_train[1:]
    
    # 测试数据：确保标签数与流数据数一致
    n_test = flow_test.shape[0]
    X_test = flow_test
    Y_test = volume_test[1:n_test+1]
    
    # 扩展标签维度为 (H, W, 1)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_test  = np.expand_dims(Y_test, axis=-1)
    
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    return X_train, Y_train, X_test, Y_test

# -----------------------
# 优化后的模型构建
# -----------------------
def build_optimized_cnn_model(input_shape):
    inputs = Input(shape=input_shape)  # input_shape: (H, W, 4)
    
    # Block 1
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 融合层，逐步减少通道数
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    # 输出层：1通道输出，使用 sigmoid 使得输出在 [0, 1] 范围内
    outputs = Conv2D(1, (3, 3), padding="same", activation="sigmoid")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# -----------------------
# 评估函数
# -----------------------
def eval_metrics(y_true, y_pred, threshold=0.0):
    """
    计算预测结果的 RMSE 和 MAPE，排除小于 threshold 的像素（防止除零问题）
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true > threshold
    if np.sum(mask) == 0:
        return -1, -1
    rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return rmse, mape

# -----------------------
# 主训练流程
# -----------------------
def main():
    # 加载数据
    X_train, Y_train, X_test, Y_test = load_data(args.data_folder)
    print("训练数据：X_train shape: {}, Y_train shape: {}".format(X_train.shape, Y_train.shape))
    print("测试数据：X_test shape: {}, Y_test shape: {}".format(X_test.shape, Y_test.shape))
    
    # 获取输入尺寸
    input_shape = X_train.shape[1:]  # (H, W, 4)
    
    # 构建优化后的模型
    model = build_optimized_cnn_model(input_shape)
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    
    # 设置 EarlyStopping
    stopper = EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    
    # 训练模型
    history = model.fit(X_train, Y_train, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        validation_split=0.2, 
                        callbacks=[stopper])
    
    # 在测试集上预测
    Y_pred = model.predict(X_test)
    print("预测结果 shape:", Y_pred.shape)
    
    # 评估
    rmse, mape = eval_metrics(Y_test, Y_pred, threshold=0.0)
    print("测试集评估结果：RMSE = {:.4f}, MAPE = {:.2f}%".format(rmse, mape * 100))
    
    # 保存模型
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = os.path.join(args.model_save_dir, "cnn_opticalflow_optimized_" + curr_time + ".keras")
    model.save(model_path)
    print("模型已保存至", model_path)

if __name__ == "__main__":
    main()
