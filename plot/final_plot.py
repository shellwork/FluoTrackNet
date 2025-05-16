import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn for potentially better aesthetics if desired, though Matplotlib is primary here
import numpy as np
import os

# --- 1. 绘图风格设置 (Publication Quality) ---
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Arial', 'DejaVu Sans'], # Ensure SimHei is available
    'axes.unicode_minus': False,
    'font.size': 11, # Slightly smaller base font for a compact look
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6), # Default, will be overridden in functions
    'lines.linewidth': 1.8, # Slightly thinner lines
    'lines.markersize': 6,
    'errorbar.capsize': 3, # Smaller capsize
    'grid.linestyle': '--', # Dashed grid lines, common in publication styles
    'grid.alpha': 0.7
})

# --- 2. 数据输入区 ---
# 您需要修改以下部分以匹配您的数据和文件路径

# 2.1 损失曲线数据 (从CSV文件读取)
# 格式: {'模型名称1': {'train': '路径/到/训练损失1.csv', 'val': '路径/到/验证损失1.csv'},
#        '模型名称2': {'train': '路径/到/训练损失2.csv', 'val': '路径/到/验证损失2.csv'}, ...}
# CSV文件应有 'step' 列和代表重复实验的后续列。
# 例如:
# step,loss_repA,loss_repB,loss_repC
# 0,0.1,0.12,0.11
# 1,0.09,0.08,0.085
# ...

# 创建示例文件 (如果它们不存在)
dummy_data_dir = "./"
if not os.path.exists(dummy_data_dir):
    os.makedirs(dummy_data_dir)

dummy_train_loss_modelA_path = os.path.join(dummy_data_dir, "loss.csv")
dummy_val_loss_modelA_path = os.path.join(dummy_data_dir, "val_loss.csv")
dummy_train_loss_modelB_path = os.path.join(dummy_data_dir, "modelB_train_loss.csv")
dummy_val_loss_modelB_path = os.path.join(dummy_data_dir, "modelB_val_loss.csv")

# ---- 示例CSV创建结束 ----

loss_csv_files = {
    'FluoTrackNet': { # 您可以为您的模型命名
        'train': dummy_train_loss_modelA_path, # <-- 修改为您训练损失CSV的实际路径
        'val': dummy_val_loss_modelA_path      # <-- 修改为您验证损失CSV的实际路径
    },
    # '模型B': {
    #     'train': dummy_train_loss_modelB_path,
    #     'val': dummy_val_loss_modelB_path
    # }
    # 如果有更多模型，按此格式添加
}
step_column_name_in_csv = 'step' # CSV中代表训练步数的列名

# 2.2 MAPE 和 RMSE 数据 (手动输入)
# 格式: {'模型/配置名称': [重复1值, 重复2值, 重复3值]}
# mape_data_comparison = {
#     'FluoTrackNet': [10.80, 11.58, 11.15],
#     'CNN': [62.57, 55.11, 55.01],
#     'LSTM': [67.99, 67.82, 67.51]
# }

# rmse_data_comparison = {
#     'FluoTrackNet': [0.015466, 0.01648, 0.01617],
#     'CNN': [0.0384, 0.0356, 0.0369],
#     'LSTM': [0.0472, 0.0469, 0.0471]
# }

rmse_data_comparison = {
    'ver_2_3': [0.0154666332527995, 0.016481015831232, 0.0161775816231966],
    'Ver_2_1': [0.0152189433574676, 0.0104870507493615],
    'Ver_1_1': [0.0293043032288551, 0.0293961521238088, 0.0320063158869743]
}

mape_data_comparison = {
    'ver_2_3': [10.806119441986, 11.5845218300819, 11.1535042524337],
    'Ver_2_1': [9.38047766685485, 8.28179270029068],
    'Ver_1_1': [15.1275441050529, 15.3703674674034, 15.5423238873481]
}

# 2.3 消融实验数据 (手动输入)
# 三组: 原模型, 原模型+模块1, 原模型+模块2
# 每组重复三次，比较MAPE和RMSE
ablation_configurations = ['原模型', '原模型+模块A', '原模型+模块B'] # 确保顺序与下面数据一致

ablation_mape_data = {
    ablation_configurations[0]: [25.0, 25.5, 24.5], # MAPE值通常较小，这里假设是百分比或其他调整后的值
    ablation_configurations[1]: [20.0, 20.5, 19.5],
    ablation_configurations[2]: [18.0, 18.5, 17.5]
}

ablation_rmse_data = {
    ablation_configurations[0]: [0.50, 0.52, 0.48],
    ablation_configurations[1]: [0.40, 0.42, 0.38],
    ablation_configurations[2]: [0.35, 0.36, 0.34]
}


# --- 3. 绘图函数 ---

def plot_loss_curves_from_csv(loss_files_dict, step_col='step',
                              title='Loss Curve', xlabel='Train Steps', ylabel='Loss',
                              output_filename='combined_loss_curves.png'):
    """
    从CSV文件读取并绘制训练和验证损失曲线，带有重复实验的标准差填充区域。
    loss_files_dict: 字典，键是模型名称，值是包含 'train' 和 'val' CSV文件路径的字典。
    step_col: CSV中包含step/epoch信息的列名。
    """
    plt.figure(figsize=(12, 7)) # 稍大图像以便容纳多条线
    color_idx = 0
    # 使用Seaborn的颜色调色板，以便有足够的区分度
    # num_total_lines = sum(len(paths) for paths in loss_files_dict.values())
    # colors = sns.color_palette("husl", num_total_lines) # hue, saturation, lightness
    colors = plt.cm.get_cmap('Blues')(np.linspace(0.4, 0.9, len(loss_files_dict) * 2))
    for model_name, paths in loss_files_dict.items():
        for loss_type, csv_path in paths.items():
            if not csv_path or not os.path.exists(csv_path):
                print(f"警告: 模型 '{model_name}' 的 {loss_type} 损失CSV文件路径 '{csv_path}' 无效或未提供，跳过。")
                color_idx +=1 # 即使跳过，也消耗一个颜色索引以保持后续颜色一致性
                continue
            
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"错误: 读取CSV文件 '{csv_path}' 失败: {e}")
                color_idx += 1
                continue

            if step_col not in df.columns:
                print(f"错误: Step列 '{step_col}' 在文件 '{csv_path}' 中未找到。可用的列: {df.columns.tolist()}")
                color_idx += 1
                continue

            # 自动识别重复实验列 (除了step_col之外的所有数值列)
            rep_cols = [col for col in df.columns if col != step_col and pd.api.types.is_numeric_dtype(df[col])]
            if not rep_cols:
                print(f"警告: 在文件 '{csv_path}' 中模型 '{model_name}' ({loss_type}) 未找到重复实验数据列。")
                # 尝试将第一列（非step_col）作为数据（如果只有一列数据）
                if len(df.columns) > 1 and df.columns[1] != step_col and pd.api.types.is_numeric_dtype(df[df.columns[1]]):
                    rep_cols = [df.columns[1]]
                    print(f"  将其处理为单次运行数据: {rep_cols[0]}")
                else:
                    color_idx += 1
                    continue


            steps = df[step_col]
            loss_values = df[rep_cols].values # (n_steps, n_reps)
            
            mean_loss = np.mean(loss_values, axis=1)
            std_loss = np.std(loss_values, axis=1) if loss_values.shape[1] > 1 else np.zeros_like(mean_loss)


            line_style = '--' if loss_type == 'val' else '-'
            label_name = f'{model_name} - {"Val" if loss_type == "val" else "Train"}'
            current_color = colors[color_idx % len(colors)] 


            plt.plot(steps, mean_loss, label=label_name, linestyle=line_style, color=current_color)
            if loss_values.shape[1] > 1: # 仅当有重复时才绘制误差带
                 plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss,
                                 color=current_color, alpha=0.2)
            color_idx += 1

    if plt.gca().has_data(): # 检查是否实际绘制了任何数据
        plt.title(title, fontsize=18, fontweight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        print(f"损失曲线图已保存为 {output_filename}")
        plt.show()
    else:
        print("没有数据被绘制到损失曲线图中。")


def plot_metric_barchart(data_dict, metric_name,
                         title='', xlabel='Model', ylabel_override=None,
                         output_filename_prefix='metric_comparison',
                         config_order=None): # 新增config_order用于指定顺序
    """
    为给定的指标数据（如MAPE, RMSE）绘制柱状图，带有误差棒。
    data_dict: 字典，键是模型/配置名称，值是包含多次重复实验结果的列表。
    metric_name: 指标的名称 (例如 'MAPE', 'RMSE')。
    config_order: 可选列表，用于指定柱状图中各项的顺序。
    """
    if config_order:
        labels = [label for label in config_order if label in data_dict] # 保证顺序且存在于数据中
        means = [np.mean(data_dict[label]) for label in labels]
        std_devs = [np.std(data_dict[label]) for label in labels]
    else:
        labels = list(data_dict.keys())
        means = [np.mean(values) for values in data_dict.values()]
        std_devs = [np.std(values) for values in data_dict.values()]

    if not labels:
        print(f"警告: 指标 '{metric_name}' 没有数据可供绘制。")
        return

    x = np.arange(len(labels))
    width = 0.7 # 柱子的宽度，可以根据柱子数量调整

    # 为每个指标图使用不同的颜色方案，或者固定一种
    # colors = sns.color_palette("viridis", len(labels))
    colors = plt.cm.get_cmap('Blues')(np.linspace(0.5, 1.0, len(labels)))


    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.6), 6)) # 动态调整图宽度
    rects = ax.bar(x, means, width, yerr=std_devs, capsize=5, color=colors, label=metric_name)

    ax.set_ylabel(ylabel_override if ylabel_override else metric_name, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title if title else f'{metric_name} Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11) # 旋转标签防止重叠
    # ax.legend(loc='upper right') # 如果只有一个指标，图例可能不是必须的

    # 在柱子上方显示数值
    for i, rect in enumerate(rects):
        height = rect.get_height()
        err = std_devs[i]
        # 调整文本位置，考虑误差棒
        text_y_position = height + err + (max(means) * 0.02) if height >= 0 else height - err - (max(means) * 0.05)
        if height < 0: # 对于负值，文本可能在柱子下方
             text_y_position = height - err - (abs(min(means))*0.05) # 简单调整


        ax.annotate(f'{height:.3f}\n(±{err:.3f})', # 显示均值和标准差
                    xy=(rect.get_x() + rect.get_width() / 2, height), # 锚点在柱顶
                    xytext=(0, 5 if height >=0 else -25),  # 向上或向下偏移
                    textcoords="offset points",
                    ha='center', va='bottom' if height >=0 else 'top', fontsize=9)

    fig.tight_layout()
    output_filename_base = output_filename_prefix.lower().replace(' ', '_')
    metric_name_safe = metric_name.lower().replace(' ', '_').replace('%','pct').replace('(','').replace(')','')
    final_output_filename = f"{output_filename_base}_{metric_name_safe}.png"
    plt.savefig(final_output_filename, dpi=300)
    print(f"{metric_name} 柱状图已保存为 {final_output_filename}")
    plt.show()


# --- 4. 主程序调用示例 ---
if __name__ == '__main__':
    print("开始生成图表...")

    # 4.1 绘制损失和验证损失曲线 (从CSV文件)
    plot_loss_curves_from_csv(
        loss_csv_files,
        step_col=step_column_name_in_csv,
        title='Training Loss & Evaluating Loss Curve', # 总标题
        output_filename='all_models_loss_curves.png'
    )

    # 4.2 绘制 MAPE 柱状图 (常规比较)
    plot_metric_barchart(mape_data_comparison,
                         metric_name='MAPE',
                         title='MAPE',
                         output_filename_prefix='comparison')

    # 4.3 绘制 RMSE 柱状图 (常规比较)
    plot_metric_barchart(rmse_data_comparison,
                         metric_name='RMSE',
                         title='RMSE',
                         output_filename_prefix='comparison')

    # 4.4 绘制消融实验结果柱状图
    # 为MAPE绘制一个图
#     plot_metric_barchart(ablation_mape_data,
#                          metric_name='MAPE',
#                          title='Ablation Study for MAPE',
#                          xlabel='Model',
#                          config_order=ablation_configurations, # 指定配置的顺序
#                          output_filename_prefix='ablation_study')

#     # 为RMSE绘制一个图
#     plot_metric_barchart(ablation_rmse_data,
#                          metric_name='RMSE',
#                          title='Ablation Study for RMSE',
#                          xlabel='Model',
#                          config_order=ablation_configurations, # 指定配置的顺序
#                          output_filename_prefix='ablation_study')

    print("所有图表生成完毕。")