#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 LSTM 对时序体数据进行建模，利用前面若干帧预测下一帧各网格的荧光强度。
输入：一段序列的体数据（每帧形状为 (H, W)，在输入前将每帧展平成向量）
标签：序列后续一帧（同样展平后再reshape回 (H, W)）
数据从预处理后保存的 npz 文件中加载
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import datetime

# -----------------------
# 参数设置
# -----------------------
parser = argparse.ArgumentParser(description="LSTM时序模型训练")
parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
parser.add_argument('--patience', type=int, default=10, help='EarlyStopping等待轮数')
parser.add_argument('--seq_len', type=int, default=5, help='LSTM输入序列长度')
parser.add_argument('--data_folder', type=str, default="data", help='预处理数据保存的文件夹')
parser.add_argument('--model_save_dir', type=str, default="model_output", help='模型保存目录')
args = parser.parse_args()

# -----------------------
# 数据加载与构造序列
# -----------------------
def load_volume_data(data_folder, seq_len):
    # 这里只用体数据进行时序建模
    volume_train = np.load(os.path.join(data_folder, "volume_train.npz"))["volume"]  # shape: (T_train, H, W)
    volume_test  = np.load(os.path.join(data_folder, "volume_test.npz"))["volume"]    # shape: (T_test, H, W)
    
    # 根据序列长度构造样本（滑动窗口法）：X 为连续 seq_len 帧，Y 为紧接着的一帧
    def create_sequences(volumes, seq_len):
        X, Y = [], []
        T = volumes.shape[0]
        for t in range(T - seq_len):
            seq = volumes[t: t+seq_len]       # shape: (seq_len, H, W)
            target = volumes[t+seq_len]         # shape: (H, W)
            X.append(seq)
            Y.append(target)
        return np.array(X), np.array(Y)
    
    X_train, Y_train = create_sequences(volume_train, seq_len)
    X_test, Y_test   = create_sequences(volume_test, seq_len)
    
    # 展平每一帧为向量（例如：H x W --> H*W）
    H, W = volume_train.shape[1:3]
    X_train = X_train.reshape((-1, args.seq_len, H * W))
    Y_train = Y_train.reshape((-1, H * W))
    X_test  = X_test.reshape((-1, args.seq_len, H * W))
    Y_test  = Y_test.reshape((-1, H * W))
    
    return X_train, Y_train, X_test, Y_test, (H, W)

# -----------------------
# 模型构建
# -----------------------
def build_lstm_model(seq_len, input_dim):
    inputs = Input(shape=(seq_len, input_dim))
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(64)(x)
    # 直接输出展平向量，与标签形状一致
    outputs = Dense(input_dim, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# -----------------------
# 评估函数
# -----------------------
def eval_metrics(y_true, y_pred, threshold=0.0):
    """
    计算预测结果的 RMSE 和 MAPE，排除小于 threshold 的像素
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
    # 加载数据并构造序列
    X_train, Y_train, X_test, Y_test, (H, W) = load_volume_data(args.data_folder, args.seq_len)
    print("训练数据：X_train shape: {}, Y_train shape: {}".format(X_train.shape, Y_train.shape))
    print("测试数据：X_test shape: {}, Y_test shape: {}".format(X_test.shape, Y_test.shape))
    
    input_dim = H * W
    # 构建 LSTM 模型
    model = build_lstm_model(args.seq_len, input_dim)
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
    model_path = os.path.join(args.model_save_dir, "lstm_model_" + curr_time + ".keras")
    model.save(model_path)
    print("模型已保存至", model_path)

if __name__ == "__main__":
    main()
