import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
from scipy.io import wavfile
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Subset
import torchaudio
import torchaudio.transforms as T
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split

from config import Config
from mfcc import mfcc


def adjust_length(tensor: torch.Tensor, target_length: int = 16000) -> torch.Tensor:
    """
    Adjusts the length of the clips. Audio files longer than 1 second are clipped, shorter ones are padded with zeros.

    Args:
        tensor (Tensor)
        target_length (int)

    Returns:
        Tensor
    """
    current_length = tensor.size(0)
    if current_length > target_length:
        # clip the clip ;)
        tensor = tensor[:target_length]
    elif current_length < target_length:
        # pad with zeros
        padding = target_length - current_length
        tensor = torch.cat([tensor, torch.zeros(padding, dtype=tensor.dtype)], dim=-1)
    return tensor


def to_mono(data: torch.Tensor) -> torch.Tensor:
    """ converts signal to mono """
    if data.dim() > 1:  
        data = torch.mean(data, dim=0)
    return data
    

def create_class_mapping() -> dict:
    """ creates a mapping from labels to integers """
    return {label: i for i, label in enumerate(Config.MAIN_CLASSES)}


class SpeechCommandsDataset(Dataset):

    def __init__(self, base_dir: str, file_list: List[str], class_map: dict, transform=None):
        self.base_dir = Path(base_dir)
        self.file_paths = [self.base_dir / line.strip() for line in file_list]
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """ gets item and performs initial transformations (signal to mono, adjusting length) """
        path = self.file_paths[idx]
        label = path.parts[-2].lower()

        label_idx = self.class_map.get(label)
        if label_idx is None:
            raise ValueError(f"Error getting label id for {path}")

        waveform, sample_rate = torchaudio.load(path)
        waveform = to_mono(waveform)
        waveform = adjust_length(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label_idx


def get_dataloaders(base_dir: str, all_files: List[str], batch_size: int = 64, transform=None, train_fraction: float = 1.0) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Gets dataloaders
    
    Args:

    Returns:
        train_loader, val_loader, test_loader, class_map
    """
    class_map = create_class_mapping()

    train_files, val_files = train_test_split(all_files, train_size=0.8)

    train_dataset = SpeechCommandsDataset(base_dir, train_files, class_map, transform)
    val_dataset = SpeechCommandsDataset(base_dir, val_files, class_map, transform)

    train_indices = np.random.choice(len(train_dataset), int(train_fraction * len(train_dataset)), replace=False) 
    train_subset = Subset(train_dataset, train_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=3, pin_memory=True)

    return train_loader, val_loader, class_map


def load_dataset(batch_size: int = 64, transform=None, train_fraction: float = 1.0):
    """
    Wrapper for get_dataloaders() method
    
    Args:
        batch_size (int)
        transform: transformations to apply
        train_fraction (float): fraction of training dataset to use

    Returns:
        return train_loader, val_loader, class_map
    """
    base_dir = Path(f'{Config.BASEDIR}').resolve()
    all_files = [str(p.resolve()) for p in base_dir.rglob('*.wav') if str(p.parts[-2]) in Config.MAIN_CLASSES]

    transform = transforms.Compose([
        T.Resample(16000, 8000),
        transforms.Lambda(lambda x: torch.tensor(mfcc(x, sample_rate=8000), dtype=torch.float32))
    ])

    train_loader, val_loader, class_map = get_dataloaders(
        base_dir=base_dir,
        all_files=all_files,
        batch_size=batch_size,
        transform=transform,
        train_fraction=train_fraction,
    )

    return train_loader, val_loader, class_map
