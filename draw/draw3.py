import pandas as pd 
import matplotlib.pyplot as plt
import os

# 读取两个CSV文件
file1 = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/data_y.csv', header=None)
file2 = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/pred_y.csv', header=None)

# 确保输出目录存在
output_dir = '/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/comparison_plots/'
os.makedirs(output_dir, exist_ok=True)

# 定义步长
step_size = 96
chunk_size = 577

# 总共的步数
total_steps = 0

# 用于存储前一行的最后一步数据
last_value_file1 = None
last_value_file2 = None

# 遍历整个文件，按照每 577 步进行
for start in range(0, len(file1), chunk_size):
    plt.figure(figsize=(12, 6))

    # 计算当前区块的结束位置
    end = min(start + chunk_size, len(file1))

    # 对区块内每 96 步绘制一次
    for i in range(start, end, step_size):
        file1_values = file1.iloc[i].values
        file2_values = file2.iloc[i].values

        # 如果有上一行的最后一步，则用它连接当前行的第一步，确保连续
        if last_value_file1 is not None and last_value_file2 is not None:
            # 连接文件1和文件2的最后一步和当前行的第一步
            plt.plot([total_steps - 1, total_steps], [last_value_file1, file1_values[0]], color='blue')
            plt.plot([total_steps - 1, total_steps], [last_value_file2, file2_values[0]], color='orange')

        # 绘制每一行的96步
        plt.plot(range(total_steps, total_steps + step_size), file1_values, label=f'true' if i == start else "", color='blue')  # 文件1
        plt.plot(range(total_steps, total_steps + step_size), file2_values, label=f'prediction' if i == start else "", color='orange')  # 文件2

        # 更新累积步长，保证下一行接在这一行之后
        total_steps += step_size

        # 保存当前行的最后一步，以便连接下一行
        last_value_file1 = file1_values[-1]
        last_value_file2 = file2_values[-1]

    # 图形设置
    plt.title(f'Comparison for steps {start} to {end}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

    # 保存图片
    plt.savefig(f'{output_dir}/comparison_rows_{start}_to_{end}.png')
    plt.close()  # 关闭图形以释放内存

    # 重置累积步长以便每个图从零开始
    total_steps = 0
    last_value_file1 = None
    last_value_file2 = None
