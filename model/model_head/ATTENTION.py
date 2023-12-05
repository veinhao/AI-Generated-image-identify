import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_head.FC4 import FCHead


class SimpleSelfAttention(nn.Module):
    """
    simple attention + fc4
    """
    def __init__(self, feature_dim, t):
        super(SimpleSelfAttention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim))

    def forward(self, x):
        # x 形状为 (batch_size, num_features, feature_dim)
        # 权重形状为 (feature_dim, )

        # 使用 softmax 来获取正规化的权重
        attn_weights = F.softmax(self.attention_weights, dim=0)

        # 计算加权平均
        weighted = torch.mul(x, attn_weights)  # 应用权重
        attn_output = torch.sum(weighted, dim=0)  # 按特征维度求和
        return torch.unsqueeze(attn_output, dim=0)
