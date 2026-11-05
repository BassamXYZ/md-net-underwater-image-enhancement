import numpy as np
import torch
from PIL import Image


def np_to_pil(np_img):
    """
        Converts image in np.array format to PIL image.
        From C x W x H [0..1] to  W x H x C [0...255]
        :param np_img:
        :return:
    """

    ar = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    if np_img.shape[0] == 1:
        ar = ar[0]
    else:
        assert np_img.shape[0] == 3, np_img.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def torch_to_np(ten_img):
    """
        Converts an image in torch.Tensor format to np.array.
        From 1 x C x W x H [0..1] to  C x W x H [0..1]
        :param ten_img:
        :return:
    """
    return ten_img.detach().cpu().numpy()[0]

def to_pil(t):
    """
    Converts a (3, H, W) tensor to a PIL image, clamping to [0, 1] first.

    The network output is an unbounded linear convolution. Casting out-of-range values to uint8
    without clamping makes them wrap around, which shows up as red dots and bright patches.
    """
    t = t.detach().float().clamp(0, 1).cpu()
    arr = np.round(t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def load_trained_model(path, device):
    """Builds MD-Net, loads the checkpoint stored at `path` and returns the model in eval mode."""
    # Imported here to avoid a circular import (neural_net -> custom_blocks -> gbl -> util)
    from model.neural_net import NeuralNetwork

    m = NeuralNetwork().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Expected output: <All keys matched successfully>
    print("load_state_dict:", m.load_state_dict(ckpt["model_state_dict"]))
    print("checkpoint epoch index:", ckpt.get("epoch"))
    m.eval()
    return m
