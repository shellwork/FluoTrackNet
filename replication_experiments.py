import subprocess
import sys
import yaml
import os
import shutil # 用于复制文件

# 假设 main.py 与此脚本在同一目录下
MAIN_SCRIPT_PATH = "main.py"
ORIGINAL_CONFIG_PATH = "config.yaml" # 您的原始配置文件名
TEMP_CONFIG_BASENAME = "temp_config_replicate" # 临时配置文件的前缀

# 重复实验的次数
NUM_REPLICATES = 3

def run_experiment_with_modified_config(replicate_num):
    """
    修改配置文件并运行单次实验。

    Args:
        replicate_num (int): 当前重复实验的编号。
    """
    try:
        # 1. 读取原始 YAML 配置文件内容
        with open(ORIGINAL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 原始配置文件 '{ORIGINAL_CONFIG_PATH}' 未找到。", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析原始配置文件 '{ORIGINAL_CONFIG_PATH}' 失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 修改配置
    # 获取原始实验名称，如果不存在则使用一个默认值
    base_experiment_name = config_data.get("experiment_name", "DefaultExperiment")
    modified_experiment_name = f"{base_experiment_name}_replicate{replicate_num}"
    
    config_data["experiment_name"] = modified_experiment_name
    config_data["swanlab"] = True # 确保启用 SwanLab

    # 如果 main.py 中的 swanlab.init 也依赖于 config_data 中的 project 和 workspace,
    # 确保它们也存在或者在这里设置
    if "project" not in config_data:
        config_data["project"] = "DefaultProject" # 或者从脚本参数获取
        print(f"警告: 'project' 未在 {ORIGINAL_CONFIG_PATH} 中找到，已设置为 'DefaultProject'。")
    # if "workspace" not in config_data:
    #     config_data["workspace"] = "DefaultWorkspace" # 如果需要的话

    # 3. 将修改后的配置写入临时 YAML 文件
    temp_config_path = f"{TEMP_CONFIG_BASENAME}_{replicate_num}.yaml"
    try:
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True) # sort_keys=False 保持原始顺序
        print(f"--- 为 Replicate {replicate_num} 创建临时配置文件: {temp_config_path} ---")
        print(f"--- 实验名称设置为: {modified_experiment_name} ---")
        print(f"--- SwanLab 设置为: True ---")
    except IOError as e:
        print(f"错误: 写入临时配置文件 '{temp_config_path}' 失败: {e}", file=sys.stderr)
        return # 无法继续，则跳过此复制

    # 4. 构建并执行 main.py 的命令
    command = [
        sys.executable, # 使用当前 Python 解释器
        MAIN_SCRIPT_PATH,
        "--config", temp_config_path,
    ]

    print(f"--- 开始执行 Replicate {replicate_num} (实验名: {modified_experiment_name}) ---")
    print(f"命令: {' '.join(command)}")

    try:
        # 执行命令，实时打印输出
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        
        # 实时打印标准输出
        for line in process.stdout:
            print(line, end='')
        
        # 实时打印标准错误
        for line in process.stderr:
            print(line, end='', file=sys.stderr)
            
        process.wait() # 等待子进程结束

        if process.returncode == 0:
            print(f"--- Replicate {replicate_num} ({modified_experiment_name}) 成功完成。 ---")
        else:
            print(f"--- Replicate {replicate_num} ({modified_experiment_name}) 失败，退出码: {process.returncode}。 ---", file=sys.stderr)

    except FileNotFoundError:
        print(f"错误: 脚本 '{MAIN_SCRIPT_PATH}' 未找到。", file=sys.stderr)
        print(f"请确保 main.py 在同一目录或更新 MAIN_SCRIPT_PATH。", file=sys.stderr)
    except Exception as e:
        print(f"在 Replicate {replicate_num} ({modified_experiment_name}) 执行期间发生意外错误。", file=sys.stderr)
        print(str(e), file=sys.stderr)
    finally:
        # 5. 清理：删除临时配置文件
        try:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
                print(f"--- 已删除临时配置文件: {temp_config_path} ---")
        except OSError as e:
            print(f"警告: 删除临时配置文件 '{temp_config_path}' 失败: {e}", file=sys.stderr)

if __name__ == "__main__":
    # 检查原始配置文件是否存在
    if not os.path.exists(ORIGINAL_CONFIG_PATH):
        print(f"错误: 核心配置文件 '{ORIGINAL_CONFIG_PATH}' 未找到。脚本无法继续。", file=sys.stderr)
        print(f"请确保 '{ORIGINAL_CONFIG_PATH}' 文件存在于脚本运行的目录中。", file=sys.stderr)
        sys.exit(1)

    for i in range(1, NUM_REPLICATES + 1):
        run_experiment_with_modified_config(i)
        print("\n" + "="*50 + "\n") # 在两次实验之间添加分隔符

    print(f"--- 所有 {NUM_REPLICATES} 次重复实验均已尝试执行。 ---")