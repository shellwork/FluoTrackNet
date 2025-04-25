import cv2
import os
from pathlib import Path

def process_videos(video_paths, output_root='./frames'):
    """处理多个视频文件，每个视频的帧保存到独立子目录
    
    Args:
        video_paths (list): 视频文件路径列表
        output_root (str): 输出根目录，默认为'./frames'
    """
    # 遍历所有视频文件
    for video_path in video_paths:
        # 验证视频文件存在
        if not os.path.exists(video_path):
            print(f"视频文件 {video_path} 不存在，跳过处理")
            continue

        # 创建视频专属输出目录（使用视频文件名）
        video_name = Path(video_path).stem  # 去除扩展名的文件名
        output_dir = os.path.join(output_root, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件 {video_path}，跳过处理")
            continue

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n开始处理: {video_name}")
        print(f"视频信息: {width}x{height} @ {fps:.2f} fps，共 {total_frames} 帧")

        # 逐帧处理
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 生成带编号的文件名
            frame_filename = os.path.join(
                output_dir, 
                f"{video_name}_frame{frame_count:06d}.jpg"
            )
            
            # 保存帧（可修改为png等其他格式）
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

            # 每处理100帧打印进度
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧...")

        # 释放资源
        cap.release()
        print(f"处理完成：{video_name}，共保存 {frame_count} 帧到 {output_dir}\n")

if __name__ == "__main__":
    # 配置部分（按需修改）
    video_directory = "./data_pre/videos"  # 视频存放目录
    output_root = "./data_pre/frames_output"  # 统一输出根目录

    # 自动获取目录下所有avi和mp4文件（可扩展其他格式）
    video_extensions = ['*.avi', '*.mp4', '*.mov']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(Path(video_directory).glob(ext))

    # 如果没有找到视频文件
    if not video_paths:
        print(f"在 {video_directory} 目录下未找到视频文件")
    else:
        # 开始批量处理
        process_videos(video_paths, output_root)
        print("所有视频处理完成！")