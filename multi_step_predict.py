import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from file_loader import file_loader
from attention import Attention, SimpleAttention

def multi_step_predict_entire_grid_by_stdn(
    model_path,
    config_path,
    att_lstm_num=3,
    long_term_lstm_seq_len=3,
    short_term_lstm_seq_len=7,
    hist_feature_daynum=7,
    last_feature_num=48,
    nbhd_size=1,
    cnn_nbhd_size=3,
    predict_steps=3,
    n=1
):
    """
    复用 file_loader.sample_stdn 的逻辑，对测试集最后时刻开始，
    进行整图多步滚动预测 (predict_steps 步)，并在每一步完成后输出热图。

    当 volume_test 与 flow_test 的时间维度之差大于 n 帧时，
    直接截掉 volume_test 多余的帧以保证 flow_test.shape[0] == volume_test.shape[0] - 1。
    """

    # 确保保存预测可视化图的目录存在
    os.makedirs("./plot_predict", exist_ok=True)

    # ========== 1) 加载原测试集 & 对齐 =============
    base_loader = file_loader(config_path=config_path)
    base_loader.load_volume()
    base_loader.load_flow()

    volume_test = base_loader.volume_test  # shape (T_test, H, W, ...) or (T_test, H, W)
    flow_test   = base_loader.flow_test    # shape (T_test-1, H, W, 4)
    T_test = volume_test.shape[0]

    # 计算差距
    diff = volume_test.shape[0] - flow_test.shape[0]  # 理论上 diff 应该 = 1
    if diff > n:
        print(f"volume_test 与 flow_test 之差 = {diff} 帧 (大于 n={n})，截去 volume_test 多余的帧数至对齐。")
        volume_test = volume_test[: (flow_test.shape[0] + 1)]
        T_test = volume_test.shape[0]
    else:
        print(f"volume_test 与 flow_test 时间差 = {diff} (在允许范围 n={n} 内或更少)，不做截断。")

    if volume_test.shape[0] - flow_test.shape[0] != 1:
        raise ValueError(f"数据长度不匹配: volume_test.shape[0]={volume_test.shape[0]}, "
                         f"flow_test.shape[0]={flow_test.shape[0]}, 理应相差1帧才可对齐光流。")

    # ========== 2) 扩展 volume_test / flow_test 以容纳 predict_steps =============
    T_test = volume_test.shape[0]
    if volume_test.ndim == 3:
        # (T, H, W)
        new_vol_shape = (T_test + predict_steps, volume_test.shape[1], volume_test.shape[2])
    else:
        # (T, H, W, C_v)
        new_vol_shape = (T_test + predict_steps,
                         volume_test.shape[1],
                         volume_test.shape[2],
                         volume_test.shape[3])

    volume_data = np.zeros(new_vol_shape, dtype=volume_test.dtype)
    volume_data[:T_test] = volume_test

    new_flow_shape = (flow_test.shape[0] + predict_steps,
                      flow_test.shape[1],
                      flow_test.shape[2],
                      flow_test.shape[3])  # channel=4
    flow_data = np.zeros(new_flow_shape, dtype=flow_test.dtype)
    flow_data[: (T_test - 1)] = flow_test

    H = volume_data.shape[1]
    W = volume_data.shape[2]

    # ========== 3) 加载模型 (自定义层) =============
    model = load_model(
        model_path,
        custom_objects={"Attention": Attention, "SimpleAttention": SimpleAttention}
    )
    print("模型加载成功:", model_path)

    # ========== 4) 多步滚动预测，每步都可视化并保存图 ===========
    for step in range(1, predict_steps + 1):
        t = (T_test - 1) + step
        # mock loader, 只到 volume_data[:t+1], flow_data[:t+1]
        mock_loader = file_loader(config_path=config_path)
        mock_loader.isVolumeLoaded = True
        mock_loader.isFlowLoaded = True

        mock_loader.volume_test = volume_data[:(t+1)]
        mock_loader.flow_test   = flow_data[:(t+1)]

        att_cnnx, att_flow, att_x, cnnx, flow_, x_, y_ = mock_loader.sample_stdn(
            datatype="test",
            att_lstm_num=att_lstm_num,
            long_term_lstm_seq_len=long_term_lstm_seq_len,
            short_term_lstm_seq_len=short_term_lstm_seq_len,
            hist_feature_daynum=hist_feature_daynum,
            last_feature_num=last_feature_num,
            nbhd_size=nbhd_size,
            cnn_nbhd_size=cnn_nbhd_size
        )

        model_inputs = att_cnnx + att_flow + att_x + cnnx + flow_ + [x_]
        y_pred = model.predict(model_inputs)  # (N, C_v)

        # 取最后一帧 (H*W) 的预测
        N = y_pred.shape[0]
        hw = H * W
        last_frame_pred = y_pred[N - hw : N]
        if volume_data.ndim == 3:
            last_frame_pred = last_frame_pred.reshape((H, W))  # 单通道
        else:
            C_v = volume_data.shape[3]
            last_frame_pred = last_frame_pred.reshape((H, W, C_v))

        volume_data[t] = last_frame_pred

        # 若要更新 flow_data[t]，可替换
        if t < flow_data.shape[0]:
            flow_data[t] = 0.0

        print(f"[{step}/{predict_steps}] => Predicted time={t}, shape={last_frame_pred.shape}")

        # ----------- (A) 每步保存热图 -----------
        plt.figure()
        if volume_data.ndim == 3:
            # 单通道
            plt.imshow(last_frame_pred, cmap='jet')
            plt.title(f"Predicted frame at time={t} [step={step}] (single-channel)")
        else:
            # 多通道 => 仅可视化第0通道
            plt.imshow(last_frame_pred[..., 0], cmap='jet')
            plt.title(f"Predicted frame at time={t}, ch=0 [step={step}]")
        plt.colorbar()

        # 保存到 ./plot_predict/pred_step_{step}_time_{t}.png
        save_path = f"./plot_predict/pred_step_{step}_time_{t}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"热图已保存: {save_path}")

    # ========== 5) 返回最终滚动预测的 volume_data ===========
    return volume_data


if __name__ == "__main__":
    model_path = "model_output/bioCV_STDN20250304184041.keras"
    config_path = "config.yaml"

    # 和训练时的参数对齐
    att_lstm_num = 2
    long_term_lstm_seq_len = 3
    short_term_lstm_seq_len = 4
    # hist_feature_daynum = 7
    # last_feature_num = 48
    nbhd_size = 2
    cnn_nbhd_size = 3
    predict_steps = 20
    n = 1

    final_data = multi_step_predict_entire_grid_by_stdn(
        model_path=model_path,
        config_path=config_path,
        att_lstm_num=att_lstm_num,
        long_term_lstm_seq_len=long_term_lstm_seq_len,
        short_term_lstm_seq_len=short_term_lstm_seq_len,
        # hist_feature_daynum=hist_feature_daynum,
        # last_feature_num=last_feature_num,
        nbhd_size=nbhd_size,
        cnn_nbhd_size=cnn_nbhd_size,
        predict_steps=predict_steps,
        n=n
    )
    print("Done. final_data.shape =", final_data.shape)
