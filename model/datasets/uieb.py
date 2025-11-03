import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable
import glob


class UIEB(Dataset):
    """
    UIEB Dataset Module

    The UIEB dataset contains:
    - Raw underwater images
    - Reference enhanced images

    Dataset structure:
    UIEB/
    ├── raw/
    │   ├── *.jpg (raw underwater images)
    ├── reference/
    │   ├── *.jpg (reference enhanced images)
    """

    def __init__(self, root_dir: str, mode: str = 'train', transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (256, 256), load_pairs: bool = True):
        """
        Initialize UIEB Dataset

        Args:
            root_dir: Root directory of the UIEB dataset
            mode: 'train', 'valid', or 'test'
            transform: Transformations for input images
            image_size: Target image size (height, width)
            load_pairs: If True, loads input-target pairs. If False, loads only inputs.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.load_pairs = load_pairs
        self.image_size = image_size

        self.raw_dir = os.path.join(root_dir, 'raw')
        self.ref_dir = os.path.join(root_dir, 'reference')

        self.raw_images = sorted(
            glob.glob(os.path.join(self.raw_dir, '*.jpg')))
        self.ref_images = sorted(
            glob.glob(os.path.join(self.ref_dir, '*.jpg')))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.raw_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            If load_pairs=True: (input_image, target_image)
            If load_pairs=False: input_image
        """
        # Load raw image
        raw_path = self.raw_images[idx]
        raw_image = Image.open(raw_path).convert('RGB')

        if self.transform:
            raw_image = self.transform(raw_image)

        if not self.load_pairs:
            return raw_image

        # Load reference image
        ref_path = self.ref_images[idx]
        ref_image = Image.open(ref_path).convert('RGB')

        if self.transform:
            ref_image = self.transform(ref_image)

        return raw_image, ref_image

    def get_image_paths(self, idx: int) -> Tuple[str, str]:
        """Get file paths for a given index"""
        return self.raw_images[idx], self.ref_images[idx]
