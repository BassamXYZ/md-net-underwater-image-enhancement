import torch.nn as nn


class CWLayer(nn.Module):
    '''........Channel Weight (CW).......'''

    def __init__(self, channels):
        super(CWLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels // 8, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.CA(y)
        return x * y
