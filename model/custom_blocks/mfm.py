import torch
import torch.nn as nn
from model.custom_blocks import Branch


class MFM(nn.Module):
    '''......Multi-layer Fusion......'''

    def __init__(self):
        super(MFM, self).__init__()
        # make MFM channels number dynamic
        self.brabch_3 = Branch(3, 64, 3)
        self.brabch_5 = Branch(3, 64, 5)
        self.brabch_7 = Branch(3, 64, 7)

        self.fusion = nn.Sequential(*[
            nn.Conv2d(64 * 3, 64 // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64 // 8, 3, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x11, x1 = self.brabch_3(x)
        x22, x2 = self.brabch_5(x)
        x33, x3 = self.brabch_7(x)

        w = self.fusion(torch.cat([x1, x2, x3], dim=1))
        w = torch.split(w, 1, dim=1)
        y = w[0] * x1 + w[1] * x2 + w[2] * x3

        return x33, y
