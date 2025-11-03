import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable
import glob


class UIEBChallenging(Dataset):
    """
    UIEB Challenging Dataset Module

    Dataset structure:
    UIEB/
    ├── challenging/
    │   ├── *.jpg
    """

    def __init__(self, root_dir: str, mode: str = 'test', transform: Optional[Callable] = None):
        """
        Initialize UIEB Challenging Dataset

        Args:
            root_dir: Root directory of the UIEB dataset
            mode: 'train', 'valid', or 'test'
            transform: Transformations for input images
            image_size: Target image size (height, width)
        """
        self.root_dir = root_dir
        self.mode = mode

        self.images_dir = os.path.join(root_dir, 'challenging')
        self.images = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))

        if transform is None:
            self.transform = transforms.ToTensor()

        else:
            self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset
        """
        # Load raw image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    def get_image_paths(self, idx: int) -> Tuple[str, str]:
        """Get file paths for a given index"""
        return self.images[idx]
