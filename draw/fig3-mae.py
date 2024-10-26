# import numpy as np
# import matplotlib.pyplot as plt

# # 定义输入序列长度（x轴）和对应的 MAE 值
# input_lengths = [24, 48, 72, 96, 144, 168, 192, 216, 240]

# # 各个模型对应的 MAE 值
# mae_lstm = [0.58, 0.60, 0.59, 0.58, 0.59, 0.60, 0.61, 0.60, 0.59]
# mae_bilstm = [0.48, 0.49, 0.50, 0.47, 0.48, 0.49, 0.49, 0.48, 0.47]
# mae_seq2seq = [0.42, 0.43, 0.41, 0.40, 0.41, 0.42, 0.43, 0.42, 0.41]
# mae_transformer = [0.34, 0.35, 0.34, 0.33, 0.34, 0.35, 0.34, 0.34, 0.33]
# mae_lstmformer = [0.29, 0.28, 0.29, 0.28, 0.29, 0.30, 0.29, 0.28, 0.27]

# # 创建图形
# plt.figure(figsize=(8, 6))

# # 绘制折线图
# plt.plot(input_lengths, mae_lstm, marker='o', label='LSTM', color='pink')
# plt.plot(input_lengths, mae_bilstm, marker='^', label='BiLSTM', color='red')
# plt.plot(input_lengths, mae_seq2seq, marker='o', linestyle='--', label='seq2seq', color='lightgreen')
# plt.plot(input_lengths, mae_transformer, marker='s', linestyle='-', label='Transformer', color='skyblue')
# plt.plot(input_lengths, mae_lstmformer, marker='o', linestyle='-', label='LSTMformer', color='purple')

# # 添加每个数据点的标注值，稍微下移以避免超出边框
# for i in range(len(input_lengths)):
#     plt.text(input_lengths[i], mae_lstm[i] - 0.02, f'{mae_lstm[i]:.2f}', ha='center', color='pink')
#     plt.text(input_lengths[i], mae_bilstm[i] - 0.02, f'{mae_bilstm[i]:.2f}', ha='center', color='red')
#     plt.text(input_lengths[i], mae_seq2seq[i] - 0.02, f'{mae_seq2seq[i]:.2f}', ha='center', color='lightgreen')
#     plt.text(input_lengths[i], mae_transformer[i] - 0.02, f'{mae_transformer[i]:.2f}', ha='center', color='skyblue')
#     plt.text(input_lengths[i], mae_lstmformer[i] - 0.02, f'{mae_lstmformer[i]:.2f}', ha='center', color='purple')


# # 添加标题、标签和图例
# plt.title('MAE for Different Models Across Input Sequence Lengths', fontsize=14)
# plt.xlabel('Input sequence length (h)', fontsize=12)
# plt.ylabel('MAE (kW)', fontsize=12)
# plt.grid(True)

# # 显示图例
# plt.legend()

# # 显示图表
# plt.tight_layout()
# plt.show()
# # 保存图片
# plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig3-t.png')
# plt.close()  # 关闭图形以释放内存

import numpy as np
import matplotlib.pyplot as plt

# 定义输入序列长度（x轴）和对应的 MAE 值
input_lengths = [96, 144, 168, 192, 216, 240, 288, 336]

# 各个模型对应的 MAE 值
mae_ourmodel = [0.58, 0.60, 0.59, 0.58, 0.59, 0.60, 0.61, 0.60, 0.59]
mae_bilstm = [0.48, 0.49, 0.50, 0.47, 0.48, 0.49, 0.49, 0.48, 0.47]
mae_seq2seq = [0.42, 0.43, 0.41, 0.40, 0.41, 0.42, 0.43, 0.42, 0.41]
mae_transformer = [0.34, 0.35, 0.34, 0.33, 0.34, 0.35, 0.34, 0.34, 0.33]
mae_lstmformer = [0.29, 0.28, 0.29, 0.28, 0.29, 0.30, 0.29, 0.28, 0.27]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图
plt.plot(input_lengths, mae_lstm, marker='o', label='LSTM', color='pink')
plt.plot(input_lengths, mae_bilstm, marker='^', label='BiLSTM', color='red')
plt.plot(input_lengths, mae_seq2seq, marker='o', linestyle='--', label='seq2seq', color='lightgreen')
plt.plot(input_lengths, mae_transformer, marker='s', linestyle='-', label='Transformer', color='skyblue')
plt.plot(input_lengths, mae_lstmformer, marker='o', linestyle='-', label='LSTMformer', color='purple')

# 添加每个数据点的标注值，稍微下移以避免超出边框
for i in range(len(input_lengths)):
    plt.text(input_lengths[i], mae_lstm[i] - 0.02, f'{mae_lstm[i]:.2f}', ha='center', color='pink')
    plt.text(input_lengths[i], mae_bilstm[i] - 0.02, f'{mae_bilstm[i]:.2f}', ha='center', color='red')
    plt.text(input_lengths[i], mae_seq2seq[i] - 0.02, f'{mae_seq2seq[i]:.2f}', ha='center', color='lightgreen')
    plt.text(input_lengths[i], mae_transformer[i] - 0.02, f'{mae_transformer[i]:.2f}', ha='center', color='skyblue')
    plt.text(input_lengths[i], mae_lstmformer[i] - 0.02, f'{mae_lstmformer[i]:.2f}', ha='center', color='purple')


# 添加标题、标签和图例
plt.title('MAE for Different Models Across Input Sequence Lengths', fontsize=14)
plt.xlabel('Input sequence length (h)', fontsize=12)
plt.ylabel('MAE (kW)', fontsize=12)
plt.grid(True)

# 显示图例
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
# 保存图片
plt.savefig(f'/root/LLM_for_TS/Time-LLM-NWP/fig3-t.png')
plt.close()  # 关闭图形以释放内存

