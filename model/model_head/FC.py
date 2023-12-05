from torch import nn


class FC2Head(nn.Module):
    def __init__(self, hidden_size: int):
        super(FC2Head, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 2)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.classifier(x)
        return x
