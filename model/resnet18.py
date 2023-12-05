import torch
from torchvision.models import resnet18
from torch import nn


class RES18(nn.Module):
    def __init__(self, config):
        super(RES18, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.model = nn.Sequential(
            self.resnet,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # 添加展平操作
            nn.Linear(512, 256),
            nn.ReLU(),  # 添加ReLU激活函数
            nn.Linear(256, 64),
            nn.ReLU(),  # 添加ReLU激活函数
            nn.Linear(64, 2),
        )
        self.model.to(config.DEVICE)

    def forward(self, x):
        return self.model(x)
