import torch.nn as nn
import torch


class BasicBlock(nn.Module):   # 定义残差结构（18层、34层）
    expansion = 1        # 对应残差结构中主分支中所采用的的卷积核的个数有没有发生变化（18层、34层没有发生变化，50层、101层、152层的卷积核个数是发生了变化的）
    # 定义初始函数定义残差结构（18层、34层）

    def __init__(self, in_channel, out_channel, stride=1, downsample = None, **kwargs):  # downsample 指下采样参数默认设置为None，指虚线的残差结构（降维的作用）
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 不使用偏置bias（使用BN是不需要使用偏置的）
        self.bn1 = nn.BatchNorm2d(out_channel)   # 定义BN层1（括号中对应的就是conv1的输出）
        self.relu = nn.ReLU()  # 定义RELU激活函数
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # 下采样方法

    def forward(self, x):  # 定义正向传播过程
        identity = x       # 将x赋值给identity，也就是捷径路上的输出值
        if self.downsample is not None:   # 进行一个判断如果不使用下采样就是指对应残差图中实线的操作
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity   # 卷积层输出和捷径相加
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):   # 定义50层、101层、152层的残差结构
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4   # 第50、101、152层的resnet，卷积层3的卷积核是卷积层1.2的4倍，所以这里是expansion = 4
    # 定义初始化函数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,   # 第三层的卷积核是前两层的4倍，也就是要乘上self.expansion
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):               # 正向传播过程
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        # 定义主分支
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):   # 定义ResNet整个网络的框架部分

    def __init__(self,
                 block,   # 对应的残差结构，根据定义的不同的层结构传入不同的block(18、34层就是BasicBlock)
                 blocks_num,   # 所使用的残差结构的数目
                 num_classes=1000,  # 训练集分类个数
                 include_top=True,  # 方便在resnet基础上搭建更加复杂的网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top   # 将include_top传入类变量中
        self.in_channel = 64   # 输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group   # ！！！！这两行问一下，不明白！！！！
                         # 输入特征矩阵的深度3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # 就是霹导表格中conv2中对应的一系列残差结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):  # channel对应的是滴一层的卷积核个数
        downsample = None  # 定义一个下采样，赋值为None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,   # 虚线残差结构
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):  # 通过一个for循环把实线的残差结构压入
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):  # 根据要定义的网络 ，看参数
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth  # 官方下载的路径
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=False)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)