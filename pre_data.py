import os
import cv2
import numpy as np
from glob import glob


# 单个实验处理函数

def preprocess_image(image):
    # image = cv2.GaussianBlur(image, (3,3), 0)  # 如需滤波，可用：cv2.GaussianBlur(image, (3,3), 0)
    return image

def grid_intensity_statistics(image, grid_size=(8, 8)):
    green_channel = image[:, :, 1].astype(np.float32)
    h, w = green_channel.shape
    grid_h, grid_w = grid_size
    num_rows = h // grid_h
    num_cols = w // grid_w
    
    intensity_map = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    for r in range(num_rows):
        for c in range(num_cols):
            row_start = r * grid_h
            row_end   = (r + 1) * grid_h
            col_start = c * grid_w
            col_end   = (c + 1) * grid_w
            patch = green_channel[row_start:row_end, col_start:col_end]
            val = np.mean(patch)
            intensity_map[r, c] = val
    
    return intensity_map

def detect_fluorescent_points(image, max_corners=300, quality_level=0.05, min_distance=2, blockSize=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                     qualityLevel=quality_level,
                                     minDistance=min_distance,
                                     blockSize=blockSize)
    return points

def compute_flow_optical(prev_img, curr_img, grid_size, draw_flow=False):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    prev_points = detect_fluorescent_points(prev_img)
    if prev_points is None:
        if draw_flow:
            return None, None
        return None
    
    num_prev = prev_points.shape[0]
    print(f"光流跟踪：前一帧检测到 {num_prev} 个荧光点")
    
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                      prev_points, None, **lk_params)
    
    good_prev = prev_points[status.flatten() == 1]
    good_curr = curr_points[status.flatten() == 1]
    
    h, w = prev_gray.shape
    grid_h, grid_w = grid_size
    num_rows = h // grid_h
    num_cols = w // grid_w
    num_cells = num_rows * num_cols
    flow_matrix = np.zeros((num_cells, num_cells), dtype=np.int32)
    
    def get_cell_index(pt):
        x, y = pt.ravel()
        row = int(y) // grid_h
        col = int(x) // grid_w
        row = min(row, num_rows - 1)
        col = min(col, num_cols - 1)
        return row * num_cols + col

    vis_img = prev_img.copy() if draw_flow else None

    for pt_prev, pt_curr in zip(good_prev, good_curr):
        cell_from = get_cell_index(pt_prev)
        cell_to = get_cell_index(pt_curr)
        flow_matrix[cell_from, cell_to] += 1

        if draw_flow:
            x0, y0 = pt_prev.ravel().astype(int)
            x1, y1 = pt_curr.ravel().astype(int)
            cv2.circle(vis_img, (x0, y0), radius=2, color=(255, 0, 0), thickness=-1)
            cv2.arrowedLine(vis_img, (x0, y0), (x1, y1),
                            color=(0, 255, 0), thickness=1, tipLength=0.3)
    
    if draw_flow:
        return flow_matrix, vis_img
    return flow_matrix

def flow_matrix_to_map(flow_matrix, num_rows, num_cols):
    flow_map = np.zeros((num_rows, num_cols, 4), dtype=np.float32)
    
    def cell_index(r, c):
        return r * num_cols + c
    
    for i in range(num_rows):
        for j in range(num_cols):
            from_idx = cell_index(i, j)
            # 上
            if i > 0:
                to_idx = cell_index(i - 1, j)
                flow_map[i, j, 0] = flow_matrix[from_idx, to_idx]
            # 右
            if j < num_cols - 1:
                to_idx = cell_index(i, j + 1)
                flow_map[i, j, 1] = flow_matrix[from_idx, to_idx]
            # 下
            if i < num_rows - 1:
                to_idx = cell_index(i + 1, j)
                flow_map[i, j, 2] = flow_matrix[from_idx, to_idx]
            # 左
            if j > 0:
                to_idx = cell_index(i, j - 1)
                flow_map[i, j, 3] = flow_matrix[from_idx, to_idx]
    
    return flow_map

def min_max_scale(data):
    dmin = data.min()
    dmax = data.max()
    if dmax == dmin:
        return data
    return (data - dmin) / (dmax - dmin)

def process_image_sequence(image_folder, grid_size=(8, 8), save_demo=True):
    image_files = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not image_files:
        raise ValueError(f"未在指定文件夹 {image_folder} 内找到 .jpg 图片。")
    
    volumes = []
    flows = []
    prev_img = None
    demo_saved = False
    
    for idx, f in enumerate(image_files):
        img = cv2.imread(f)
        if img is None:
            continue
        img_filtered = preprocess_image(img)
        
        intensity_map = grid_intensity_statistics(img_filtered, grid_size)
        volumes.append(intensity_map)
        
        if prev_img is not None:
            if save_demo and not demo_saved:
                flow_matrix, vis_img = compute_flow_optical(prev_img, img_filtered,
                                                            grid_size, draw_flow=True)
                demo_path = os.path.join("data", "optical_flow_demo.jpg")
                if flow_matrix is not None and vis_img is not None:
                    cv2.imwrite(demo_path, vis_img)
                    print(f"光流演示图已保存至 {demo_path}")
                demo_saved = True
            else:
                flow_matrix = compute_flow_optical(prev_img, img_filtered,
                                                   grid_size, draw_flow=False)
            
            if flow_matrix is not None:
                h, w = intensity_map.shape
                flow_map = flow_matrix_to_map(flow_matrix, h, w)
            else:
                h, w = intensity_map.shape
                flow_map = np.zeros((h, w, 4), dtype=np.float32)
            
            flows.append(flow_map)
        
        prev_img = img_filtered
    
    volumes = np.array(volumes)  # (T, H, W)
    flows = np.array(flows)      # (T-1, H, W, 4)

    # min_max归一化
    # volumes = min_max_scale(volumes)
    # flows = min_max_scale(flows)
    
    return volumes, flows


# 多组实验合并
def merge_experiments_with_zeros(experiment_folders, grid_size=(8,8)):
    """
    对于多个实验文件夹 experiment_folders，依次调用 process_image_sequence，
    在相邻实验之间插入 (1, H, W) 的零帧和 (1, H, W, 4) 的零flow。
    返回合并后的 volumes, flows。
    """
    all_volumes = []
    all_flows = []
    
    for i, folder in enumerate(experiment_folders):
        print(f"\n>>> 处理第 {i+1} 个实验：{folder}")
        v, f = process_image_sequence(folder, grid_size=grid_size)
        
        if i == 0:
            # 第一组实验，直接放进去
            all_volumes.append(v)
            all_flows.append(f)
        else:
            # 在前一组实验之后插入一个0帧
            # 先确定当前 H, W
            H, W = v.shape[1], v.shape[2]
            
            # zero volume shape: (1, H, W)
            zero_vol = np.zeros((1, H, W), dtype=v.dtype)
            # zero flow shape: (1, H, W, 4)
            zero_flow = np.zeros((1, H, W, 4), dtype=f.dtype)
            
            # 插入 0 帧后再衔接新的实验
            all_volumes.append(zero_vol)
            all_flows.append(zero_flow)
            
            all_volumes.append(v)
            all_flows.append(f)
    
    # 拼接
    merged_volumes = np.concatenate(all_volumes, axis=0)
    merged_flows   = np.concatenate(all_flows, axis=0)
    return merged_volumes, merged_flows

def save_preprocessed_data(volumes, flows, save_folder):
    """
    按时间顺序把 volumes & flows 划分为 train/test 并保存。
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    T = volumes.shape[0]
    train_idx = int(0.8 * T)
    
    volume_train = volumes[:train_idx]   
    volume_test  = volumes[train_idx:]   
    
    # flows: (T-1, ...)
    if train_idx - 1 > 0:
        flow_train = flows[:train_idx - 1]
    else:
        flow_train = flows
    flow_test = flows[train_idx - 1:] if (train_idx - 1) < flows.shape[0] else []
    
    np.savez_compressed(os.path.join(save_folder, "volume_train.npz"), volume=volume_train)
    np.savez_compressed(os.path.join(save_folder, "volume_test.npz"),  volume=volume_test)
    np.savez_compressed(os.path.join(save_folder, "flow_train.npz"),   flow=flow_train)
    np.savez_compressed(os.path.join(save_folder, "flow_test.npz"),    flow=flow_test)
    
    print("预处理数据保存完成：")
    print(f"  volume_train.npz: {volume_train.shape}")
    print(f"  volume_test.npz: {volume_test.shape}")
    print(f"  flow_train.npz: {flow_train.shape if len(flow_train) else 0}")
    print(f"  flow_test.npz: {flow_test.shape if len(flow_test) else 0}")


if __name__ == "__main__":
    # 1) 定义多组实验文件夹，每个文件夹下都有 .jpg 图像
    experiment_folders = [
        "data_pre/1BMSC-20 h/20h BMSC8_多通道缩时_20240723_02",
        "data_pre/1BMSC-20 h/20h BMSC5_多通道缩时_20240723_02",
        "data_pre/1BMSC-20 h/20h BMSC98_缩时_20240723_01",
    ]
    # 网格大小
    grid_size = (16, 16)
    
    # 2) 合并多组实验 + 插入0帧
    volumes_merged, flows_merged = merge_experiments_with_zeros(experiment_folders, grid_size)
    print("\n合并后的 volume 形状:", volumes_merged.shape)  # (T_all, H, W)
    print("合并后的 flow   形状:", flows_merged.shape)     # (T_all-1, H, W, 4)
    
    # 数据归一化处理
    volumes_merged = min_max_scale(volumes_merged)
    flows_merged = min_max_scale(flows_merged)
    
    # 3) 分割 & 保存
    save_folder = "data"
    save_preprocessed_data(volumes_merged, flows_merged, save_folder)
