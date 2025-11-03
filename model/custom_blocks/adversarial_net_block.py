import torch.nn as nn
from model.custom_blocks import DWEUGroupStructure, MFM


class AdversarialNetBlock(nn.Module):
    '''Architecture of:
       .........TMD (Transmission Map Disentanglement)
       .........SRD (Scene Radiance Disentanglement)
    '''

    def __init__(self):
        super(AdversarialNetBlock, self).__init__()
        self.multilayer_fusion = MFM()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dweu_group = DWEUGroupStructure(64, 3, 4)
        self.final = nn.Conv2d(64, 3, 1, padding=0, bias=True)

    def forward(self, x):
        x33, x1 = self.multilayer_fusion(x)

        x2 = self.avg_pool(x1)
        x3 = self.dweu_group(x1)

        x4 = x2 * x3 + x33
        y = self.Final(x4)

        return y
