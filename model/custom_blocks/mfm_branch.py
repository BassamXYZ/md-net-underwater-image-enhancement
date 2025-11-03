import torch.nn as nn


class Branch(nn.Module):
    '''......MFM Branch......'''

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Branch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, padding=(kernel_size // 2), bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)
        x4 = self.IN(x3)
        x5 = self.conv2(x4)

        return x1, x5
