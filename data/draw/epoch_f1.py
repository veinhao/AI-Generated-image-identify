import matplotlib.pyplot as plt
import torch

# 加载数据
data = torch.load('/home/nano/PycharmProjects/AI-Generated-image-identify/data/log/dataset_origin-best-f1_score-11-15-14:23.pth')

# 将数据从 GPU 移动到 CPU 并转换为 Python 列表
values = [x.cpu().item() for x in data]

# 创建 epochs 轴的值
epochs = range(len(values))

# 计算最大值及其位置
max_value = max(values)
max_index = values.index(max_value)

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(epochs, values, marker='o', linestyle='-', color='b')
plt.axhline(y=max_value, color='r', linestyle='-', label=f'Max Value: {max_value}')
plt.text(0, max_value, f'  {max_value}', color='r', verticalalignment='bottom')

# 设置图表标题和轴标签
plt.title('VITH(FREEZE) RESNET(FREEZE) Lion')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.xticks(epochs)  # 设置x轴刻度为 epochs 数组的下标
plt.grid(True)
plt.legend()
plt.show()
