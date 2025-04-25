import numpy as np

def inspect_npz_file(file_path, num_samples=2):
    try:
        print(f"\n=== 检查文件: {file_path} ===")
        # 加载 .npz 文件
        data = np.load(file_path)
        
        # 列出所有保存的数组名称
        files_in_npz = data.files
        print(f"文件包含的数组: {files_in_npz}")
        
        # 遍历每个数组并输出信息
        for arr_name in files_in_npz:
            array = data[arr_name]
            print(f"\n数组名称: {arr_name}")
            print(f"形状: {array.shape}")
            print(f"数据类型: {array.dtype}")
            print(f"矩阵求和：{array.sum()}")
            
            # 输出前 num_samples 个样本（如果维度允许）
            if array.ndim >= 1:
                print("前几行数据示例:")
                print(array[:num_samples] if array.shape[0] > 0 else "空数组")
            else:
                print("数据是标量或0维数组")
            
                
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
    finally:
        data.close() if 'data' in locals() else None

# 调用函数检查文件（替换为你的实际路径）
inspect_npz_file("volume_test.npz")
inspect_npz_file("flow_test.npz")