import numpy as np
import yaml

class file_loader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # 每天的时间槽数（如48个时间槽，依据 timeslot_sec 设置）
        self.timeslot_daynum = int(72000 / self.config["timeslot_sec"])
        self.threshold = int(self.config["threshold"])
        self.isVolumeLoaded = False
        self.isFlowLoaded = False

    def load_flow(self):
        # 原文尝试用 flow_train_max 做归一化，这里在数据预处理时已经归一化到0-1.
        self.flow_train = np.load(self.config["flow_train"], mmap_mode='r')["flow"] 
        self.flow_test = np.load(self.config["flow_test"], mmap_mode='r')["flow"]
        self.isFlowLoaded = True

    def load_volume(self):
        self.volume_train = np.load(self.config["volume_train"], mmap_mode='r')["volume"]
        self.volume_test = np.load(self.config["volume_test"], mmap_mode='r')["volume"]
        self.isVolumeLoaded = True

    def _extract_window(self, data, t, x, y, nbhd_size):
        """
        在时间 t 的 data[t], 取 (x, y) 为中心的 (2*nbhd_size+1, 2*nbhd_size+1) 邻域窗口，
        data[t] 的形状是 (H, W, C)，或 (H, W) 也行，下面默认存在通道 C。
        """
        # 如果 data[t] 是 (H, W) 没有通道维度，可先在外面加一维
        if data.ndim == 3:  # (T, H, W)
            channels = 1
        else:
            channels = data.shape[-1]  # (T, H, W, C)

        # 创建零填充窗口
        window = np.zeros((2 * nbhd_size + 1, 2 * nbhd_size + 1, channels), dtype=data.dtype)

        # 边界索引
        x_start = max(x - nbhd_size, 0)
        x_end = min(x + nbhd_size + 1, data.shape[1])
        y_start = max(y - nbhd_size, 0)
        y_end = min(y + nbhd_size + 1, data.shape[2])

        # 计算贴到 window 的起始位置
        wx_start = x_start - (x - nbhd_size)
        wy_start = y_start - (y - nbhd_size)

        # 取 data[t, x_start:x_end, y_start:y_end, :] 填到 window
        if data.ndim == 3:
            # data[t] -> shape (H, W)
            # 这里给它加一维通道
            patch = data[t, x_start:x_end, y_start:y_end]
            patch = patch[..., np.newaxis]  # (h_sub, w_sub, 1)
        else:
            # data[t] -> shape (H, W, C)
            patch = data[t, x_start:x_end, y_start:y_end, :]

        window[wx_start:wx_start + (x_end - x_start),
               wy_start:wy_start + (y_end - y_start), :] = patch
        return window

    def _extract_window_from_array(self, array, x, y, nbhd_size):
        """
        针对 flow[t] 这样的 (H, W, C) 数组，取 (x, y) 为中心的邻域，零填充边界。
        """
        channels = array.shape[-1]
        window = np.zeros((2 * nbhd_size + 1, 2 * nbhd_size + 1, channels), dtype=array.dtype)

        x_start = max(x - nbhd_size, 0)
        x_end = min(x + nbhd_size + 1, array.shape[0])
        y_start = max(y - nbhd_size, 0)
        y_end = min(y + nbhd_size + 1, array.shape[1])

        wx_start = x_start - (x - nbhd_size)
        wy_start = y_start - (y - nbhd_size)

        window[wx_start:wx_start + (x_end - x_start),
               wy_start:wy_start + (y_end - y_start), :] = array[x_start:x_end, y_start:y_end, :]
        return window

    def sample_stdn(self, datatype, att_lstm_num=3, long_term_lstm_seq_len=3,
                    short_term_lstm_seq_len=7, hist_feature_daynum=7, last_feature_num=48,
                    nbhd_size=1, cnn_nbhd_size=3):
        """
        将原文的逻辑简化，改为:
          - volume: shape (T, H, W) or (T, H, W, C_v)
          - flow:   shape (T, H, W, 4)，每个时刻( H, W, 4 )表示上下左右4通道
        """
        if not self.isVolumeLoaded:
            self.load_volume()
        if not self.isFlowLoaded:
            self.load_flow()

        if long_term_lstm_seq_len % 2 != 1:
            raise ValueError("Att-lstm seq_len must be odd!")

        if datatype == "train":
            data = self.volume_train   # shape (T, H, W, C_v) or (T, H, W)
            flow_data = self.flow_train  # shape (T, H, W, 4)
        elif datatype == "test":
            data = self.volume_test
            flow_data = self.flow_test
        else:
            raise ValueError("datatype must be 'train' or 'test'")

        # 准备输出容器：与原代码同名，但内部含义会略有差异
        # CNN注意力和FLOW注意力特征
        cnn_att_features = []
        flow_att_features = []
        lstm_att_features = []
        for i in range(att_lstm_num):
            cnn_att_features.append([[] for _ in range(long_term_lstm_seq_len)])
            flow_att_features.append([[] for _ in range(long_term_lstm_seq_len)])
            lstm_att_features.append([])

        # 短期CNN & FLOW特征
        cnn_features = [[] for _ in range(short_term_lstm_seq_len)]
        flow_features = [[] for _ in range(short_term_lstm_seq_len)]
        short_term_lstm_features = []
        labels = []

        # 计算可以开始采样的时间起点
        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        time_end = data.shape[0]
        # data.shape[-1] 如果 data是 (T, H, W), 那 data.shape[-1] = W, 可能不是通道了
        # 这里仅在做最后 label 时，会 flatten -> 作为回归目标
        # 你可以根据需要只预测单通道，也可一次预测全部通道
        # 下面假设 data.shape 是 (T,H,W,C_v)
        if data.ndim == 3:
            volume_type = 1
        else:
            volume_type = data.shape[-1]

        for t in range(time_start, time_end):
            if t % 100 == 0:
                print(f"Now sampling at time slot {t} ...")

            H = data.shape[1]
            W = data.shape[2]
            for x in range(H):
                for y in range(W):
                    # ----------------------
                    # 1) 短期 LSTM 特征
                    # ----------------------
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t 范围: [t - short_term_lstm_seq_len, ..., t-1]
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        # (a) CNN特征: 从 data[real_t] 中提取 (cnn_nbhd_size) 邻域
                        cnn_feature = self._extract_window(data, real_t, x, y, cnn_nbhd_size)
                        cnn_features[seqn].append(cnn_feature)

                        # (b) FLOW特征: 直接取 flow_data[real_t], 再做邻域提取
                        if real_t < 0 or real_t >= flow_data.shape[0]:
                            # 边界情况，可用零填充或直接跳过
                            local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4), dtype=flow_data.dtype)
                        else:
                            flow_map = flow_data[real_t]     # shape (H, W, 4)
                            local_flow_feature = self._extract_window_from_array(flow_map, x, y, cnn_nbhd_size)
                        flow_features[seqn].append(local_flow_feature)

                        # (c) LSTM 特征向量:
                        #    - 邻域特征 (nbhd_size)
                        #    - 最近时刻 features (last_feature_num)
                        #    - 历史天 features (hist_feature_daynum * timeslot_daynum)
                        nbhd_feature = self._extract_window(data, real_t, x, y, nbhd_size).flatten()
                        last_feature_start = max(real_t - last_feature_num, 0)
                        last_feature_array = data[last_feature_start: real_t, x, y]
                        last_feature_vec = last_feature_array.flatten()
                        hist_feature_list = []
                        for past_t in range(real_t - hist_feature_daynum*self.timeslot_daynum, real_t, self.timeslot_daynum):
                            if past_t >= 0:
                                hist_feature_list.append(data[past_t, x, y])
                            else:
                                hist_feature_list.append(np.zeros_like(data[0, x, y]))
                        hist_feature = np.array(hist_feature_list).flatten()

                        feature_vec = np.concatenate((hist_feature, last_feature_vec, nbhd_feature))
                        short_term_lstm_samples.append(feature_vec)

                    short_term_lstm_features.append(np.array(short_term_lstm_samples))

                    # ----------------------
                    # 2) Attention-LSTM 特征
                    # ----------------------
                    for att_lstm_cnt in range(att_lstm_num):
                        long_term_lstm_samples = []
                        # 原文：att_t = t - (att_lstm_num - att_lstm_cnt)*timeslot_daynum + (long_term_lstm_seq_len -1)/2 +1
                        att_t = t - (att_lstm_num - att_lstm_cnt)*self.timeslot_daynum \
                                  + (long_term_lstm_seq_len -1)//2 + 1
                        att_t = int(att_t)

                        for seqn in range(long_term_lstm_seq_len):
                            real_t = att_t - (long_term_lstm_seq_len - seqn)
                            # (a) CNN特征
                            cnn_feature = self._extract_window(data, real_t, x, y, cnn_nbhd_size)
                            cnn_att_features[att_lstm_cnt][seqn].append(cnn_feature)

                            # (b) FLOW特征
                            if real_t < 0:
                                local_flow_feature = np.zeros((2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4), dtype=flow_data.dtype)
                            else:
                                flow_map = flow_data[real_t]
                                local_flow_feature = self._extract_window_from_array(flow_map, x, y, cnn_nbhd_size)
                            flow_att_features[att_lstm_cnt][seqn].append(local_flow_feature)

                            # (c) LSTM特征（邻域 + 最近 + 历史天）
                            nbhd_feature = self._extract_window(data, real_t, x, y, nbhd_size).flatten()
                            last_feature_start = max(real_t - last_feature_num, 0)
                            last_feature_array = data[last_feature_start: real_t, x, y]
                            last_feature_vec = last_feature_array.flatten()
                            hist_feature_list = []
                            for past_t in range(real_t - hist_feature_daynum*self.timeslot_daynum, real_t, self.timeslot_daynum):
                                if past_t >= 0:
                                    hist_feature_list.append(data[past_t, x, y])
                                else:
                                    hist_feature_list.append(np.zeros_like(data[0, x, y]))
                            hist_feature = np.array(hist_feature_list).flatten()

                            feature_vec = np.concatenate((hist_feature, last_feature_vec, nbhd_feature))
                            long_term_lstm_samples.append(feature_vec)

                        lstm_att_features[att_lstm_cnt].append(np.array(long_term_lstm_samples))

                    # ----------------------
                    # 3) 预测标签
                    # ----------------------
                    # 这里假设我们要预测 data[t, x, y, :] (或 data[t, x, y] 若单通道)
                    label_val = data[t, x, y]
                    if label_val.ndim == 0:
                        # 说明是标量
                        label_val = np.array([label_val])
                    labels.append(label_val.flatten())

        # 整理输出到 numpy array
        output_cnn_att_features = []
        output_flow_att_features = []
        for i in range(att_lstm_num):
            lstm_att_features[i] = np.array(lstm_att_features[i])  # [N, long_term_lstm_seq_len, feature_dim]
            for j in range(long_term_lstm_seq_len):
                cnn_att_features[i][j] = np.array(cnn_att_features[i][j])    # [N, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, C]
                flow_att_features[i][j] = np.array(flow_att_features[i][j])  # [N, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4]
                output_cnn_att_features.append(cnn_att_features[i][j])
                output_flow_att_features.append(flow_att_features[i][j])

        for i in range(short_term_lstm_seq_len):
            cnn_features[i] = np.array(cnn_features[i])    # [N, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, C]
            flow_features[i] = np.array(flow_features[i])  # [N, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1, 4]

        short_term_lstm_features = np.array(short_term_lstm_features)  # [N, short_term_lstm_seq_len, feature_dim]
        labels = np.array(labels)  # [N, C_v]

        return (output_cnn_att_features, output_flow_att_features,
                lstm_att_features, cnn_features, flow_features,
                short_term_lstm_features, labels)