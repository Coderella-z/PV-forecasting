import numpy as np
import matplotlib.pyplot as plt

# 模型名称和对应的 mse 值
models_1h = ['Time-LLM', 'DLinear', 'iTransformer']
models_24h = ['Time-LLM', 'DLinear', 'iTransformer']

# mse 值（96步历史数据输入）
mse_hist_1h = [0.281,  0.284, 0.266]  # 历史数据
mse_weather_1h = [0.147, 0.145, 0.147]  # 天气预测数据

# mse 值（336步历史数据输入）
mse_hist_24h = [0.244, 0.235, 0.229]
mse_weather_24h = [0.124, 0.132, 0.140]

# 设置柱状图的宽度和 X 坐标
bar_width = 0.35
index_1h = np.arange(len(models_1h))
index_24h = np.arange(len(models_24h)) + len(models_1h)  # 24小时部分的X坐标

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制1小时历史数据输入的柱状图
plt.bar(index_1h, mse_hist_1h, bar_width, label='Historical data only (96-step)', color='orange')
plt.bar(index_1h + bar_width, mse_weather_1h, bar_width, label='With weather forecast data (96-step)', color='lightgreen')

# 绘制24小时历史数据输入的柱状图
plt.bar(index_24h, mse_hist_24h, bar_width, label='Historical data only (336-step)', color='purple')
plt.bar(index_24h + bar_width, mse_weather_24h, bar_width, label='With weather forecast data (336-step)', color='yellow')

# 添加虚线分隔
plt.axvline(x=len(models_1h) - 0.5, color='black', linestyle='--')

# 添加标签、标题和坐标轴说明
plt.xlabel('Models', fontsize=12)
plt.ylabel('mse (kW)', fontsize=12)
plt.title('mse Comparison for Different Models with 96-step and 336-step Data Input', fontsize=14)

# 设置 X 轴的刻度和标签
plt.xticks(list(index_1h + bar_width / 2) + list(index_24h + bar_width / 2), models_1h + models_24h)

# 设置 X 轴的范围，去除右侧的空白区域
plt.xlim(-0.5, len(models_1h) + len(models_24h) - 0.5)

# 显示图例
plt.legend()

# 调整 y 轴范围，确保最高点不会超过边界
plt.ylim(0.0, 0.35)  # 设置 y 轴的上下限

# 显示mse值，动态调整标注位置，确保不超出边界
for i in range(len(models_1h)):
    plt.text(i, mse_hist_1h[i] + 0.01, f'{mse_hist_1h[i]}', ha='center', color='black')
    plt.text(i + bar_width, mse_weather_1h[i] + 0.01, f'{mse_weather_1h[i]}', ha='center', color='black')

for i in range(len(models_24h)):
    plt.text(i + len(models_1h), mse_hist_24h[i] + 0.01, f'{mse_hist_24h[i]}', ha='center', color='black')
    plt.text(i + len(models_1h) + bar_width, mse_weather_24h[i] + 0.01, f'{mse_weather_24h[i]}', ha='center', color='black')

# 显示图表
plt.tight_layout()
plt.show()

# 保存图片
plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig2-mse2.png')
plt.close()  # 关闭图形以释放内存
