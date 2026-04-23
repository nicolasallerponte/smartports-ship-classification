from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SmartportsDataset(Dataset):
    """
    Custom Dataset for the Smartports CCTV port surveillance dataset.

    Supports two classification tasks:
        - 'ship'   : Ship / No-Ship binary classification (294 images)
        - 'docked' : Docked / Undocked binary classification (184 images with ship)

    Args:
        csv_path   : Path to the labels CSV file (semicolon-separated).
        img_dir    : Directory containing the images.
        task       : Classification task - 'ship' or 'docked'.
        indices    : Optional list of integer indices to subset the dataset
                     (used for K-Fold train/val/test splits).
        transform  : Optional torchvision transform pipeline to apply.
    """

    TASK_CONFIGS = {
        'ship':   {'col': 'Ship/No-Ship',    'csv': 'ship.csv'},
        'docked': {'col': 'Docked/Undocked', 'csv': 'docked.csv'},
    }

    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        task: Literal['ship', 'docked'] = 'ship',
        indices: Optional[list[int]] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        if task not in self.TASK_CONFIGS:
            raise ValueError(f"task must be one of {list(self.TASK_CONFIGS.keys())}")

        self.img_dir   = Path(img_dir)
        self.task      = task
        self.transform = transform
        self.label_col = self.TASK_CONFIGS[task]['col']

        df = pd.read_csv(csv_path, sep=';')
        df.columns = ['filename', 'label']

        # Docked task: keep only images that contain a ship
        if task == 'docked':
            df = df[df['label'] != -1].reset_index(drop=True)

        self.df = df if indices is None else df.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']
        label    = int(row['label'])

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_labels(self) -> list[int]:
        """Returns all labels - used by StratifiedKFold."""
        return self.df['label'].tolist()