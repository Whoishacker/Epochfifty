import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from d2l import torch as d2l


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use1_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.res_conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.res_conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

        if use1_1conv:
            self.res_conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.res_conv3 = None

        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.res_conv1(x)))
        y = self.bn2(self.res_conv2(y))
        if self.res_conv3:
            x = self.res_conv3(x)
        y += x
        return F.relu(y)


def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, output_channels, use1_1conv=True, strides=2))
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk


if __name__=='__main__':
    """ResNet18"""
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(
        b1,
        b2,
        b3,
        b4,
        b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 10)
    )

    # X = torch.randn(32, 1, 256, 256)
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output shape:\t', X.shape)

    # 展示网络架构
    summary(net, (32, 1, 256, 256))

    # 训练网络
    lr, num_epochs, batch_size = 0.001, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())









