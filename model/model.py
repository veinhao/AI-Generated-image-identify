from transformers import ViTFeatureExtractor, ViTModel
from model.model_middle.msa import MultiHeadSelfAttention
import torch
from torch import nn
from  model.model_head.FC4 import FCHead


# 定义一个新的头部为二分类任务的全连接层

class ClassModel(nn.Module):
    def __init__(self, config):
        super(ClassModel, self).__init__()

        self.config = config
        self.device = self.config.DEVICE
        self.model = None
        self.model_a = config.MODELA(self.config).to(self.device)
        if self.config.MODEL_MIDDLE == 'two path msa':
            self.model_b = config.MODELB().to(self.device)
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.msa = MultiHeadSelfAttention(config.NUM_FEATURE, config.NUM_HEAD).to(self.device)
            for param in self.model_b.parameters():
                param.requires_grad = False

        for param in self.model_a.parameters():
            param.requires_grad = False

        self.model_head = self.config.MODEL_HEAD(self.config.HIDDEN_SIZE).to(self.device)

    def forward(self, x):
        if self.config.MODEL_MIDDLE == 'two path msa':
            modela_f = self.model_a(x)
            modelb_f = self.model_b(x)
            modelb_f = self.global_avg_pool(modelb_f)
            modelb_f = modelb_f.view(modelb_f.size(0), -1)
            feature = torch.cat((modela_f, modelb_f), dim=1)
            fused_feature = self.msa(feature)

            out = self.model_head(fused_feature)
            out = torch.squeeze(out, dim=1)
        else:
            modela_f = self.model_a(x)
            out = self.model_head(modela_f)
        return out

