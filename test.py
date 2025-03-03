import numpy as np
import os

def check_data_shapes(
    volume_train_path="data/volume_train.npz",
    volume_test_path="data/volume_test.npz",
    flow_train_path="data/flow_train.npz",
    flow_test_path="data/flow_test.npz"
):
    # 1) 加载数据
    if not (os.path.exists(volume_train_path) and 
            os.path.exists(volume_test_path) and 
            os.path.exists(flow_train_path) and 
            os.path.exists(flow_test_path)):
        print("找不到指定的 npz 文件，请检查路径是否正确。")
        return

    vtrain = np.load(volume_train_path)["volume"]
    vtest  = np.load(volume_test_path )["volume"]
    ftrain = np.load(flow_train_path  )["flow"]
    ftest  = np.load(flow_test_path   )["flow"]

    # 2) 打印形状
    print("==== 形状检查 ====")
    print(f"volume_train.shape = {vtrain.shape}")
    print(f"volume_test.shape  = {vtest.shape}")
    print(f"flow_train.shape   = {ftrain.shape}")
    print(f"flow_test.shape    = {ftest.shape}")

    # 3) 进一步的逻辑检查
    # -----------------------------------------------------------
    # 一般情况下： 
    #   volume_train.shape[0] + volume_test.shape[0] = T_all
    #   flow_train.shape[0] + flow_test.shape[0] = T_all - 1
    # 并且理想情况下:
    #   flow_train.shape[0] = volume_train.shape[0] - 1
    #   flow_test.shape[0]  = volume_test.shape[0]  - 1
    # 如果不满足，可以在此打印错误信息或警告。
    # -----------------------------------------------------------

    T_train_vol = vtrain.shape[0]
    T_test_vol  = vtest.shape[0]
    T_train_flow = ftrain.shape[0]
    T_test_flow  = ftest.shape[0]

    all_good = True  # 用于标记所有检查项是否通过

    # a) 检查 train 数据对齐
    if T_train_vol - 1 != T_train_flow:
        print(f"[警告] volume_train.shape[0] - 1 = {T_train_vol - 1}，"
              f"但 flow_train.shape[0] = {T_train_flow}，不相等！")
        all_good = False
    else:
        print(f"[OK] 训练集帧数匹配: (volume={T_train_vol}, flow={T_train_flow} => flow=volume-1)")

    # b) 检查 test 数据对齐
    if T_test_vol - 1 != T_test_flow:
        print(f"[警告] volume_test.shape[0] - 1 = {T_test_vol - 1}，"
              f"但 flow_test.shape[0] = {T_test_flow}，不相等！")
        all_good = False
    else:
        print(f"[OK] 测试集帧数匹配: (volume={T_test_vol}, flow={T_test_flow} => flow=volume-1)")

    # c) 合计数量是否衔接
    T_all_vol = T_train_vol + T_test_vol
    T_all_flow = T_train_flow + T_test_flow
    if T_all_flow != T_all_vol - 1:
        print(f"[警告] 总帧数 volume={T_all_vol}, flow={T_all_flow}。"
              f"理论上 flow 应该比 volume 少 1，但 {T_all_flow} != {T_all_vol - 1}。")
        all_good = False
    else:
        print(f"[OK] 合并后总帧数匹配: volume={T_all_vol}, flow={T_all_flow} => flow=volume-1")

    print("==== 检查结果 ====")
    if all_good:
        print("所有数据形状都符合预期！")
    else:
        print("部分检查不匹配，请留意上方警告信息。")


if __name__ == "__main__":
    # 你可以根据实际文件路径来改参数
    check_data_shapes(
        volume_train_path="data/volume_train.npz",
        volume_test_path="data/volume_test.npz",
        flow_train_path="data/flow_train.npz",
        flow_test_path="data/flow_test.npz"
    )
