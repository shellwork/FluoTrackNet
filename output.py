import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import file_loader  # 请确保 file_loader.py 在你的 PYTHONPATH 中

def inference_and_visualize_keras(model_path, config_path="data.json", dataset_type="taxi", nbhd_size=2, cnn_nbhd_size=3):
    # 直接加载 .keras 模型
    model = load_model(model_path)
    print("模型加载成功：", model_path)
    
    # 根据数据集类型加载配置
    if dataset_type == "taxi":
        sampler = file_loader.file_loader(config_path=config_path)
    elif dataset_type == "bike":
        sampler = file_loader.file_loader(config_path="data_bike.json")
    else:
        raise Exception("不支持的数据集类型，请选择 'taxi' 或 'bike'")
    
    # 使用采样器生成评估数据
    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
        datatype="test",
        nbhd_size=nbhd_size,
        cnn_nbhd_size=cnn_nbhd_size
    )
    print("评估数据采样完成，样本数：", y.shape[0])
    
    # 执行推理，输入顺序需与训练时一致
    y_pred = model.predict(att_cnnx + att_flow + att_x + cnnx + flow + [x, ])
    print("推理完成，预测结果形状：", y_pred.shape)
    
    # 可视化结果：这里只绘制第一个通道的对比曲线
    num_samples_to_plot = 100  # 可以根据需要调整显示样本数量
    plt.figure(figsize=(12, 6))
    true_values = y[:num_samples_to_plot, 0] if y.ndim > 1 else y[:num_samples_to_plot]
    pred_values = y_pred[:num_samples_to_plot, 0] if y_pred.ndim > 1 else y_pred[:num_samples_to_plot]
    plt.plot(true_values, 'b-', label="真实值")
    plt.plot(pred_values, 'r--', label="预测值")
    plt.xlabel("样本索引")
    plt.ylabel("归一化体量")
    plt.title("评估数据集上的推理结果对比")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # 修改下面的路径为实际保存的 .keras 模型路径
    model_path = "./model_output/stdn_model.keras"
    # 根据你的数据集选择对应的配置文件和数据类型
    inference_and_visualize_keras(model_path, config_path="data_bike.json", dataset_type="bike", nbhd_size=2, cnn_nbhd_size=3)
