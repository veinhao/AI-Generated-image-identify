import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_features, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        assert num_features % num_heads == 0, "num_features % num_heads must equal 0"

        self.num_heads = num_heads
        self.num_features = num_features
        self.query = nn.Linear(num_features, num_features)
        self.key = nn.Linear(num_features, num_features)
        self.value = nn.Linear(num_features, num_features)

        self.fc = nn.Linear(num_features, num_features)

    def forward(self, x):
        batch_size = x.size(0)

        # 查询、键、值
        query = self.query(x).view(batch_size, -1, self.num_heads, self.num_features // self.num_heads).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.num_features // self.num_heads).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.num_features // self.num_heads).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.num_features // self.num_heads)
        attention = F.softmax(scores, dim=-1)

        # 融合值
        x = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_features)
        x = self.fc(x)
        return x

# # 示例使用
# num_features = 512  # 特征维度，应与 ResNet 和 ViT 的输出维度相同
# num_heads = 8       # 注意力头的数量
#
# # 创建多头自注意力模块
# mhsa = MultiHeadSelfAttention(num_features, num_heads)
#
# # 假设 resnet_features 和 vit_features 是从 ResNet 和 ViT 中提取的特征
# # 需要保证它们的维度是一致的
# combined_features = torch.cat((resnet_features, vit_features), dim=1)
#
# # 融合特征
# fused_features = mhsa(combined_features)
