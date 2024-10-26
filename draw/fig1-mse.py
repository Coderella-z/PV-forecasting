# import numpy as np
# import matplotlib.pyplot as plt

# # 设置模型名称和对应的 mse 值
# models = ['LSTM', 'BiLSTM', 'seq2seq', 'Transformer', 'LSTMformer']
# mse_hist = [0.716, 0.680, 0.711, 0.682, 0.664]  # 仅使用历史数据的 mse
# mse_weather = [0.582, 0.516, 0.407, 0.429, 0.306]  # 使用天气预测数据的 mse

# # 设置柱状图的宽度和 X 坐标
# bar_width = 0.35
# index = np.arange(len(models))

# # 绘制柱状图
# plt.figure(figsize=(8, 6))
# plt.bar(index, mse_hist, bar_width, label='Historical data only', color='orange')
# plt.bar(index + bar_width, mse_weather, bar_width, label='With weather forecast data', color='pink')

# # 添加标签、标题和坐标轴说明
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('mse (kW)', fontsize=12)
# plt.title('Comparison of mse for Different Models', fontsize=14)
# plt.xticks(index + bar_width / 2, models)
# plt.legend()

# # 显示mse值
# for i in range(len(models)):
#     plt.text(i, mse_hist[i] + 0.02, f'{mse_hist[i]:.3f}', ha='center', color='black')
#     plt.text(i + bar_width, mse_weather[i] + 0.02, f'{mse_weather[i]:.3f}', ha='center', color='black')

# # 显示图表
# plt.tight_layout()
# plt.show()
# # 保存图片
# plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig1.png')
# plt.close()  # 关闭图形以释放内存

import numpy as np
import matplotlib.pyplot as plt

# 设置模型名称和对应的 mse 值
models = ['our model', 'DLinear', 'iTransformer']
mse_hist = [0.244, 0.235, 0.229 ]  # 仅使用历史数据的 mse
mse_weather = [0.124, 0.132, 0.140 ]  # 使用天气预测数据的 mse

# 设置柱状图的宽度和 X 坐标
bar_width = 0.35
index = np.arange(len(models))

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(index, mse_hist, bar_width, label='Historical data only', color='orange')
plt.bar(index + bar_width, mse_weather, bar_width, label='With weather forecast data', color='pink')

# 添加标签、标题和坐标轴说明
plt.xlabel('Models', fontsize=12)
plt.ylabel('MSE (kW)', fontsize=12)
plt.title('Comparison of MSE for Different Models', fontsize=14)
plt.xticks(index + bar_width / 2, models)
plt.legend()

# 调整 y 轴范围，确保最高点不会超过边界
plt.ylim(0.0, 0.30)  # 设置 y 轴的上下限

# 显示mse值
for i in range(len(models)):
    plt.text(i, mse_hist[i] + 0.01, f'{mse_hist[i]:.3f}', ha='center', color='black')
    plt.text(i + bar_width, mse_weather[i] + 0.01, f'{mse_weather[i]:.3f}', ha='center', color='black')

# 显示图表
plt.tight_layout()
plt.show()
# 保存图片
plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig1-MSE.png')
plt.close()  # 关闭图形以释放内存
