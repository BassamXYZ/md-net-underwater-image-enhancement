import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable


class UIEBChallenging(Dataset):
    """
    UIEB Challenging Dataset Module

    Dataset structure:
    UIEB/
    ├── challenging/
    │   ├── *.jpg
    """
    
    def __init__(self, input_dir: str, transform: Optional[Callable] = None):
        """
        Initialize UIEB Challenging Dataset

        Args:
            input_dir: Path of the UIEB dataset directory 
            transform: Transformations for input images
        """
        self.input_dir, self.transform = input_dir, transform
        self.input_images = sorted(os.listdir(input_dir))

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.input_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a sample from the dataset"""
        x = Image.open(os.path.join(self.input_dir, self.input_images[idx])).convert("RGB")
        return self.transform(x) if self.transform else x

