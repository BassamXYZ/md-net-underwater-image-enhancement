import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable


class UnderwaterDataset(Dataset):
    """
    Dataset Module

    Dataset structure:
    DataSet/
    ├── input/
    │   ├── *.jpg (raw underwater images)
    ├── target/
    │   ├── *.jpg (reference enhanced images)
    """
    def __init__(self, input_dir: str, target_dir: str, transform: Optional[Callable] = None):
        """
        Initialize Dataset

        Args:
            input_dir: Path of the dataset input directory
            target_dir: Path of the dataset target directory
            transform: Transformations for input images
        """
        self.input_dir, self.target_dir, self.transform = input_dir, target_dir, transform
        self.input_images = sorted(os.listdir(input_dir))

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.input_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset

        Args:
            idx: Index of the sample

        Returns:
            (input_image, target_image)
        """
        name = self.input_images[idx]
        x = Image.open(os.path.join(self.input_dir, name)).convert("RGB")
        y = Image.open(os.path.join(self.target_dir, name)).convert("RGB")
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        return x, y

