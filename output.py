import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from file_loader import file_loader
from attention import Attention, SimpleAttention
import models


if __name__ == '__main__':
    # load parameters
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_path = "model_output/bioCV_STDN20250426151531.keras"
    config_path = "config.yaml"
    # Must match training settings
    att_lstm_num = config["att_lstm_num"]
    long_term_lstm_seq_len = config["long_term_lstm_seq_len"]
    short_term_lstm_seq_len = config["short_term_lstm_seq_len"]
    nbhd_size = config["nbhd_size"]
    cnn_nbhd_size = config["cnn_nbhd_size"]
    
    # load model files
    model = load_model(
        model_path,
        custom_objects={"Attention": Attention, "SimpleAttention": SimpleAttention}
    )
    print("\nModel loaded successfully:", model_path)
    
    # inference
    sampler = file_loader.file_loader()
    modeler = models.models()

    att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(
        datatype="test",
        att_lstm_num=att_lstm_num,
        long_term_lstm_seq_len=long_term_lstm_seq_len,
        short_term_lstm_seq_len=short_term_lstm_seq_len,
        nbhd_size=nbhd_size,
        cnn_nbhd_size=cnn_nbhd_size
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
    print("Inference done. y_pred shape:", y_pred.shape)
    
    threshold = float(sampler.threshold)
    print("Evaluating threshold: {0}.".format(threshold))
    print("Before evaluation, y shape:", y.shape)
    print("Before evaluation, y_pred shape:", y_pred.shape)
    
    from plot import visualization_plot
    visualization_plot(y, y_pred, sampler, num_samples_to_plot=300)