import torch.nn as nn
import torch


# 定义ResNet18/34的残差结构，为2个3x3的卷积
class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1

    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样  其实就是在残差支路上加了一次卷积
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out


# 定义ResNet50/101/152的残差结构，为1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # expansion是指在每个小残差块内，减小尺度增加维度的倍数，如64*4=256
    # Bottleneck层输出通道是输入的4倍
    expansion = 4

    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构，专门用来改变x的通道数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out


# 定义ResNet类
class ResNetForSegmentation(nn.Module):
    def __init__(self, block, layers, num_classes=1000, include_top=False):
        super(ResNetForSegmentation, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 添加适用于分割的卷积层
        self.conv_final = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=1)
        # 添加上采样层以匹配输入尺寸
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
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
        x = self.conv_final(x)  # 输出每个类别的预测
        x = self.upsample(x)  # 上采样回原始尺寸
        return x

# ResNet()中block参数对应的位置是BasicBlock或Bottleneck
# ResNet()中blocks_num[0-3]对应[3, 4, 6, 3]，表示残差模块中的残差数
# 34层的resnet
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNetForSegmentation(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 50层的resnet
#num_classes是需要分类的个数
def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNetForSegmentation(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 101层的resnet
def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNetForSegmentation(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)