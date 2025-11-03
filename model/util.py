import numpy as np
from PIL import Image
import os
from torchvision import transforms


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


def save_batch(batch, save_dir):
    """
    Save each original and prediction image as separate file
    """
    os.makedirs(save_dir, exist_ok=True)

    batch.cpu()

    for i in range(batch.shape[0]):
        tensor = batch[i]
        image = transforms.PILToTensor()(tensor)
        path = os.path.join(save_dir, f"{i:03d}.png")
        image.save(path)
