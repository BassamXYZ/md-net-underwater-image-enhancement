import numpy as np
from PIL import ImageFilter
from torchvision.transforms import ToTensor
from model.util import torch_to_np, np_to_pil


def gbl(x):
    """.....Global Background Light....."""
    np_img = np.clip(torch_to_np(x), 0, 1)
    pil_img = np_to_pil(np_img)
    h, w = pil_img.size
    windows = (h + w) / 2
    gbl = pil_img.filter(ImageFilter.GaussianBlur(windows))
    gbl = ToTensor()(gbl)
    return gbl.unsqueeze(0)
