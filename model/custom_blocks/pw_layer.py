import torch.nn as nn


class PWLayer(nn.Module):
    '''........Pixel Weight (PW).......'''

    def __init__(self, channels):
        super(PWLayer, self).__init__()

        self.PA = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(channels // 8, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.PA(x)
        return x * y
