from transformers import ViTFeatureExtractor, ViTModel
import torch
from torch import nn


# 定义一个新的头部为二分类任务的全连接层
class ViTForBinaryClassification(nn.Module):
    def __init__(self, config1):
        super(ViTForBinaryClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(config1.VIT_PRETRAINED_DIR)
        # self.model_head = config1.MODEL_HEAD(config1.HIDDEN_SIZE, 2).to(self.device)

        # 冻结预训练模型的所有参数
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        x = outputs.pooler_output
        # x = self.model_head(x)  # 调用独立的头部模块
        return x
