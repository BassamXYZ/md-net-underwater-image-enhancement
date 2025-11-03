import torch.nn as nn
from model.custom_blocks import PWLayer, CWLayer


class DWEU(nn.Module):
    '''........Dual-layer Weight Estimation Unit.......'''

    def __init__(self, channels, kernel_size):
        super(DWEU, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding=(kernel_size // 2), bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding=(kernel_size // 2), bias=True)
        self.palayer = PWLayer(channels)
        self.calayer = CWLayer(channels)

    def forward(self, x):
        y = self.conv1(x)
        y1 = self.calayer(y)
        y2 = self.palayer(y)
        y = y1 + y2
        y = self.conv2(y)
        y = y + x
        return y


class DWEUGroupStructure(nn.Module):
    '''........Group structure for DWEU.......'''

    def __init__(self, channels, kernel_size, blocks):
        super(DWEUGroupStructure, self).__init__()
        modules = [DWEU(channels, kernel_size) for _ in range(blocks)]
        self.gs = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gs(x)
        return res
