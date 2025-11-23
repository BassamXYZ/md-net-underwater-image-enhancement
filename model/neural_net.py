import os
import torch.nn as nn
from model.custom_blocks import AdversarialNetBlock, gbl

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.srd = AdversarialNetBlock()
        self.tmd = AdversarialNetBlock()

    def forward(self, data):
        j_out = self.srd(data)
        t_out = self.tmd(data)
        if data.device == "cuda":
            a_out = gbl(data).cuda()
        else:
            a_out = gbl(data)

        i_rec = j_out * t_out + (1 - t_out) * a_out

        return j_out, i_rec
