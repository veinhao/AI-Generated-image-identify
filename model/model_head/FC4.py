from torch import nn


class FCHead(nn.Module):
    def __init__(self, hidden_size: int):
        super(FCHead, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.ReLU()(x)
        x = self.classifier(x)
        return x
