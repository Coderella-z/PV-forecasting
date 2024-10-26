# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # 读取数据
# true_data = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/data_x.csv', header=None)
# pred_data = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/pred_y.csv', header=None)
# true_96_data = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/data_y.csv', header=None)

# # 确保输出目录存在
# output_dir = '/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/plots/'
# os.makedirs(output_dir, exist_ok=True)

# # 对每一行数据进行绘图
# for i in range(min(len(true_data), len(pred_data), len(true_96_data))):
#     true_values = true_data.iloc[i].values
#     pred_values = pred_data.iloc[i].values
#     true_96_values = true_96_data.iloc[i].values

#     # 创建图形
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(336), true_values, label='True Values', color='blue')
#     plt.plot(range(335, 432), [true_values[-1]] + pred_values.tolist(), label='Predicted Values', color='orange')
#     plt.plot(range(335, 432), [true_values[-1]] + true_96_values.tolist(), label='True 96 Steps', color='green')
#     plt.title(f'True vs Predicted Values - Row {i}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid()

#     # 保存图片
#     plt.savefig(f'{output_dir}/true_vs_predicted_row_{i}.png')
#     plt.close()  # 关闭图形以释放内存

import pandas as pd 
import matplotlib.pyplot as plt
import os

# 读取数据
true_data = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/data_x.csv', header=None)
pred_data = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/pred_y.csv', header=None)
true_96_data = pd.read_csv('/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/data_y.csv', header=None)

# 确保输出目录存在
output_dir = '/root/LLM_for_TS/Time-LLM-NWP/checkpoints/TimeLLMNWP2_336_96_TimeLLMNWP2_pv_336_168_96-TimeLLMNWP2/checkpoint-8/plots/'
os.makedirs(output_dir, exist_ok=True)

# 对每一行数据进行绘图
for i in range(min(len(true_data), len(pred_data), len(true_96_data))):
    true_values = true_data.iloc[i].values
    pred_values = pred_data.iloc[i].values
    true_96_values = true_96_data.iloc[i].values

    # 创建图形
    plt.figure(figsize=(12, 6))
    plt.plot(range(336), true_values, label='True Values', color='blue')
    plt.plot(range(336, 432), pred_values, label='Predicted Values', color='orange')  # 修改为直接绘制pred_values
    plt.plot(range(336, 432), true_96_values, label='True 96 Steps', color='green')  # 同上
    plt.title(f'True vs Predicted Values - Row {i}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()

    # 保存图片
    plt.savefig(f'{output_dir}/true_vs_predicted_row_{i}.png')
    plt.close()  # 关闭图形以释放内存
