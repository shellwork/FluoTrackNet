import sys
import datetime
import argparse
import logging
import os
import numpy as np
import yaml  # 用于加载 YAML 配置文件

import keras
import tensorflow as tf

import file_loader
import models

# 设置 GPU 仅在需要时动态分配内存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import swanlab

# ----------------------
# 命令行参数设置
# ----------------------
parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
parser.add_argument('--max_epochs', type=int, default=1000, help='maximum epochs')
parser.add_argument('--att_lstm_num', type=int, default=3, help='the number of time for attention (i.e., value of Q in the paper)')
parser.add_argument('--long_term_lstm_seq_len', type=int, default=3, help='the number of days for attention mechanism (i.e., value of P in the paper)')
parser.add_argument('--short_term_lstm_seq_len', type=int, default=7, help='the length of short term value')
parser.add_argument('--cnn_nbhd_size', type=int, default=3, help='neighbors for local cnn (2*cnn_nbhd_size+1) for area size')
parser.add_argument('--nbhd_size', type=int, default=2, help='for feature extraction')
parser.add_argument('--cnn_flat_size', type=int, default=128, help='dimension of local conv output')
parser.add_argument('--model_name', type=str, default='', help='model_name')

# SwanLab 相关参数
parser.add_argument('--swanlab', action='store_true', help='Enable SwanLab training tracking')
parser.add_argument('--workspace', type=str, default='', help='SwanLab workspace name')
parser.add_argument('--project', type=str, default='', help='SwanLab project name')
parser.add_argument('--experiment_name', type=str, default='', help='SwanLab experiment name')

# 增加 --config 参数，用于指定 YAML 配置文件路径
parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')

args = parser.parse_args()

# 如果提供了配置文件，则加载并更新参数
if args.config:
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # 更新 args 中的属性（配置文件中的设置将覆盖默认值）
        for key, value in config.items():
            setattr(args, key, value)
        print("Loaded configuration from", args.config)
    except Exception as e:
        print("Failed to load config file:", e)

print(args)


class CustomStopper(keras.callbacks.EarlyStopping):
    # 增加参数 start_epoch 用于控制从何时开始 EarlyStopping 检查
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class SwanLabCallback(keras.callbacks.Callback):
    """
    自定义 Keras 回调，用于在每个 epoch 结束时将训练指标记录到 SwanLab
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            swanlab.log(logs, step=epoch)


def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask) == 0:
        return -1
    mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))
    return rmse, mape


def eval_lstm(y, pred_y, threshold):
    # 打印评估前的形状信息
    print("eval_lstm: Original y shape:", y.shape)
    print("eval_lstm: Original pred_y shape:", pred_y.shape)
    
    # 将标签展平成一维数组
    y = y.flatten()
    pred_y = pred_y.flatten()
    
    # 检查两者长度是否一致，否则截断到较小的长度
    if y.shape[0] != pred_y.shape[0]:
        min_len = min(y.shape[0], pred_y.shape[0])
        print("Warning: y and pred_y lengths differ: y_len = {}, pred_y_len = {}. Truncating to {} samples".format(y.shape[0], pred_y.shape[0], min_len))
        y = y[:min_len]
        pred_y = pred_y[:min_len]
    
    print("eval_lstm: Flattened y length:", y.shape[0])
    print("eval_lstm: Flattened pred_y length:", pred_y.shape[0])
    
    # 构造阈值掩码
    mask = y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(y[mask] - pred_y[mask]) / y[mask])
        rmse = np.sqrt(np.mean(np.square(y[mask] - pred_y[mask])))
    else:
        mape, rmse = None, None

    return rmse, mape


def main(batch_size=64, max_epochs=100, validation_split=0.2, callbacks=None):
    model_hdf5_path = "./model_output/"
    
    # 读入数据
    sampler = file_loader.file_loader()
    modeler = models.models()

    # 采样训练数据
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
        datatype="train",
        att_lstm_num=args.att_lstm_num,
        long_term_lstm_seq_len=args.long_term_lstm_seq_len,
        short_term_lstm_seq_len=args.short_term_lstm_seq_len,
        nbhd_size=args.nbhd_size,
        cnn_nbhd_size=args.cnn_nbhd_size
    )
    
    print("Training data shapes:")
    print("  att_cnnx: list of {} tensors, first tensor shape: {}".format(len(att_cnnx), att_cnnx[0].shape))
    print("  att_flow: list of {} tensors, first tensor shape: {}".format(len(att_flow), att_flow[0].shape))
    print("  att_x: list of {} tensors, first tensor shape: {}".format(len(att_x), att_x[0].shape))
    print("  cnnx: list of {} tensors, first tensor shape: {}".format(len(cnnx), cnnx[0].shape))
    print("  flow: list of {} tensors, first tensor shape: {}".format(len(flow), flow[0].shape))
    print("  x shape: ", x.shape)
    print("  y shape: ", y.shape)
    
    print("Start training {0} with input shape {2} / {1}".format(args.model_name, x.shape, cnnx[0].shape))

    model = modeler.stdn(
        att_lstm_num=args.att_lstm_num,
        att_lstm_seq_len=args.long_term_lstm_seq_len,
        lstm_seq_len=len(cnnx),
        feature_vec_len=x.shape[-1],
        cnn_flat_size=args.cnn_flat_size,
        nbhd_size=cnnx[0].shape[1],
        nbhd_type=cnnx[0].shape[-1]
    )

    # 训练时增加 EarlyStopping 和 SwanLab 跟踪（如果启用）
    model.fit(
        x=att_cnnx + att_flow + att_x + cnnx + flow + [x, ],
        y=y,
        batch_size=batch_size,
        validation_split=validation_split,
        epochs=max_epochs,
        callbacks=callbacks
    )

    # 测试数据采样与预测
    # 注意：测试时必须传入和训练时一致的参数
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
        datatype="test",
        att_lstm_num=args.att_lstm_num,
        long_term_lstm_seq_len=args.long_term_lstm_seq_len,
        short_term_lstm_seq_len=args.short_term_lstm_seq_len,
        nbhd_size=args.nbhd_size,
        cnn_nbhd_size=args.cnn_nbhd_size
    )
    
    print("Test data shapes:")
    print("  att_cnnx: list of {} tensors, first tensor shape: {}".format(len(att_cnnx), att_cnnx[0].shape))
    print("  att_flow: list of {} tensors, first tensor shape: {}".format(len(att_flow), att_flow[0].shape))
    print("  att_x: list of {} tensors, first tensor shape: {}".format(len(att_x), att_x[0].shape))
    print("  cnnx: list of {} tensors, first tensor shape: {}".format(len(cnnx), cnnx[0].shape))
    print("  flow: list of {} tensors, first tensor shape: {}".format(len(flow), flow[0].shape))
    print("  x shape: ", x.shape)
    print("  y shape: ", y.shape)
    
    y_pred = model.predict(att_cnnx + att_flow + att_x + cnnx + flow + [x, ])
    print("y_pred shape:", y_pred.shape)
    
    threshold = float(sampler.threshold)
    print("Evaluating threshold: {0}.".format(threshold))
    
    # 调用评估函数前打印预测与标签的形状
    # 调用评估函数前打印预测与标签的形状
    print("Before evaluation, y shape:", y.shape)
    print("Before evaluation, y_pred shape:", y_pred.shape)

    rmse, mape = eval_lstm(y, y_pred, threshold)
    print("Test on model {0}:".format(args.model_name))
    print("  rmse = {0}, mape = {1}%".format(rmse, mape * 100 if mape is not None else None))

    
    # SwanLab记录测试指标
    if args.swanlab:
        swanlab.log({
            "test/rmse": prmse,
            "test/mape": pmape * 100 if pmape is not None else None
        })

    # 保存模型
    currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.path.join(model_hdf5_path, args.model_name + currTime + ".keras")
    if os.path.exists(file_path):
        os.remove(file_path)
    model.save(file_path, save_format='keras')


if __name__ == "__main__":
    # 构造回调列表
    callbacks = []
    # 初始化 EarlyStopping
    stopper = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=20)
    callbacks.append(stopper)
    
    # 初始化 SwanLab实验跟踪
    if args.swanlab:
        swanlab.init(
            project=args.project if args.project else "DefaultProject",
            experiment_name=args.experiment_name if args.experiment_name else "DefaultExperiment",
            description="Training {} model".format(args.model_name),
            config=vars(args)
        )
        callbacks.append(SwanLabCallback())
    
    main(batch_size=args.batch_size, max_epochs=args.max_epochs, callbacks=callbacks)
