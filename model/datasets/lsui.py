import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable
import glob


class LSUI(Dataset):
    """
    LSUI Dataset Module

    Dataset structure:
    LSUI/
    ├── input/
    │   ├── *.jpg (raw underwater images)
    ├── GT/
    │   ├── *.jpg (reference enhanced images)
    """

    def __init__(self, root_dir: str, mode: str = 'train', transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (256, 256), load_pairs: bool = True):
        """
        Initialize LSUI Dataset

        Args:
            root_dir: Root directory of the LSUI dataset
            mode: 'train', 'valid', or 'test'
            transform: Transformations for input images
            image_size: Target image size (height, width)
            load_pairs: If True, loads input-target pairs. If False, loads only inputs.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.load_pairs = load_pairs
        self.image_size = image_size

        self.input_dir = os.path.join(root_dir, 'input')
        self.gt_dir = os.path.join(root_dir, 'GT')

        self.input_images = sorted(
            glob.glob(os.path.join(self.input_dir, '*.jpg')))
        self.gt_images = sorted(
            glob.glob(os.path.join(self.gt_dir, '*.jpg')))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.input_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            If load_pairs=True: (input_image, gt_image)
            If load_pairs=False: input_image
        """
        # Load raw image
        input_path = self.input_images[idx]
        input_image = Image.open(input_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)

        if not self.load_pairs:
            return input_image

        # Load reference image
        gt_path = self.gt_images[idx]
        gt_image = Image.open(gt_path).convert('RGB')

        if self.transform:
            gt_image = self.transform(gt_image)

        return input_image, gt_image

    def get_image_paths(self, idx: int) -> Tuple[str, str]:
        """Get file paths for a given index"""
        return self.input_images[idx], self.gt_images[idx]
