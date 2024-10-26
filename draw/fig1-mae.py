# import numpy as np
# import matplotlib.pyplot as plt

# # 设置模型名称和对应的 MAE 值
# models = ['LSTM', 'BiLSTM', 'seq2seq', 'Transformer', 'LSTMformer']
# mae_hist = [0.716, 0.680, 0.711, 0.682, 0.664]  # 仅使用历史数据的 MAE
# mae_weather = [0.582, 0.516, 0.407, 0.429, 0.306]  # 使用天气预测数据的 MAE

# # 设置柱状图的宽度和 X 坐标
# bar_width = 0.35
# index = np.arange(len(models))

# # 绘制柱状图
# plt.figure(figsize=(8, 6))
# plt.bar(index, mae_hist, bar_width, label='Historical data only', color='orange')
# plt.bar(index + bar_width, mae_weather, bar_width, label='With weather forecast data', color='pink')

# # 添加标签、标题和坐标轴说明
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('MAE (kW)', fontsize=12)
# plt.title('Comparison of MAE for Different Models', fontsize=14)
# plt.xticks(index + bar_width / 2, models)
# plt.legend()

# # 显示MAE值
# for i in range(len(models)):
#     plt.text(i, mae_hist[i] + 0.02, f'{mae_hist[i]:.3f}', ha='center', color='black')
#     plt.text(i + bar_width, mae_weather[i] + 0.02, f'{mae_weather[i]:.3f}', ha='center', color='black')

# # 显示图表
# plt.tight_layout()
# plt.show()
# # 保存图片
# plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig1.png')
# plt.close()  # 关闭图形以释放内存

import numpy as np
import matplotlib.pyplot as plt

# 设置模型名称和对应的 MAE 值
models = ['our model', 'DLinear', 'iTransformer']
mae_hist = [0.288, 0.301, 0.277]  # 仅使用历史数据的 MAE
mae_weather = [0.237, 0.237,0.223 ]  # 使用天气预测数据的 MAE

# 设置柱状图的宽度和 X 坐标
bar_width = 0.35
index = np.arange(len(models))

# 绘制柱状图
plt.figure(figsize=(8, 6))
plt.bar(index, mae_hist, bar_width, label='Historical data only', color='orange')
plt.bar(index + bar_width, mae_weather, bar_width, label='With weather forecast data', color='pink')

# 添加标签、标题和坐标轴说明
plt.xlabel('Models', fontsize=12)
plt.ylabel('MAE (kW)', fontsize=12)
plt.title('Comparison of MAE for Different Models', fontsize=14)
plt.xticks(index + bar_width / 2, models)
plt.legend()

# 调整 y 轴范围，确保最高点不会超过边界
plt.ylim(0.0, 0.35)  # 设置 y 轴的上下限

# 显示MAE值
for i in range(len(models)):
    plt.text(i, mae_hist[i] + 0.01, f'{mae_hist[i]:.3f}', ha='center', color='black')
    plt.text(i + bar_width, mae_weather[i] + 0.01, f'{mae_weather[i]:.3f}', ha='center', color='black')

# 显示图表
plt.tight_layout()
plt.show()
# 保存图片
plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig1-pv.png')
plt.close()  # 关闭图形以释放内存
