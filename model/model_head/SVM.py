import torch
import torch.nn as nn


class LinearSVM(nn.Module):
    """简单的线性 SVM"""
    def __init__(self, in_features, out_features):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.fc(x)

