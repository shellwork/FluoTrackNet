import os
import cv2
import numpy as np
from glob import glob

def preprocess_image(image):
    """
    对图像做高斯滤波预处理
    """
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def detect_features(image, max_corners=300, quality_level=0.05, min_distance=2, blockSize=5):
    """
    检测图像中的角点（荧光点）
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                     qualityLevel=quality_level,
                                     minDistance=min_distance,
                                     blockSize=blockSize)
    return points

def visualize_optical_flow(prev_img, curr_img):
    """
    计算两帧图像之间的光流，并在 prev_img 上绘制出光流箭头与检测到的角点
    """
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # 检测前一帧角点
    features = detect_features(prev_img)
    if features is None:
        print("未检测到足够角点，跳过该帧光流可视化")
        return None
    
    # Lucas-Kanade 光流参数
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    
    # 计算当前帧中对应的角点位置
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None, **lk_params)
    if next_points is None:
        print("光流计算失败")
        return None

    # 选择跟踪成功的点
    good_prev = features[status.flatten() == 1]
    good_curr = next_points[status.flatten() == 1]

    # 创建一个副本用于绘制光流
    vis_img = prev_img.copy()
    for pt_prev, pt_curr in zip(good_prev, good_curr):
        x0, y0 = pt_prev.ravel().astype(int)
        x1, y1 = pt_curr.ravel().astype(int)
        # 在原点画个小圆点
        cv2.circle(vis_img, (x0, y0), 2, (255, 0, 0), -1)
        # 画箭头指示光流方向
        cv2.arrowedLine(vis_img, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=1, tipLength=0.3)
    return vis_img

def process_experiment(folder_path, output_dir):
    """
    针对单个实验文件夹：
      1. 读取所有 .jpg 图像（按文件名排序）
      2. 依次计算连续帧之间的光流，并保存可视化结果
    """
    image_files = sorted(glob(os.path.join(folder_path, "*.jpg")))
    if len(image_files) < 2:
        print(f"文件夹 {folder_path} 中图片不足两张，无法计算光流")
        return

    # 创建输出目录（以实验文件夹名作为子目录）
    exp_name = os.path.basename(os.path.normpath(folder_path))
    exp_output_dir = os.path.join(output_dir, exp_name)
    if not os.path.exists(exp_output_dir):
        os.makedirs(exp_output_dir)
    
    prev_img = None
    for idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = preprocess_image(img)
        if prev_img is None:
            prev_img = img
            continue

        vis = visualize_optical_flow(prev_img, img)
        if vis is not None:
            # 文件名中包含实验文件夹名和帧编号
            output_path = os.path.join(exp_output_dir, f"{exp_name}_flow_{idx:04d}.jpg")
            cv2.imwrite(output_path, vis)
            print(f"保存可视化图：{output_path}")
        prev_img = img

if __name__ == "__main__":
    # 三组数据的实验文件夹路径（请根据实际路径进行修改）
    experiment_folders = [
        "data_pre/frames_output/cbd_缩时_20250410_03",
        "data_pre/frames_output/cbd_缩时_20250410_07",
        "data_pre/frames_output/cbd_缩时_20250410_05",
        "data_pre/frames_output/cbd_缩时_20250410_06",
        "data_pre/frames_output/cbd_缩时_20250410_04",
    ]
    
    # 输出目录，用于保存光流可视化结果
    output_directory = "optical_flow_visualization"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for folder in experiment_folders:
        print(f"\n>>> 正在处理实验文件夹：{folder}")
        process_experiment(folder, output_directory)
