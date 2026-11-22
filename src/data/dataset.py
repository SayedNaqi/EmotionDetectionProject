
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


class FER2013Dataset(Dataset):

    def __init__(self, csv_path, usage="Training", transform=None):

        self.csv_path = Path(csv_path)
        self.usage = usage
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        self.data = self.data[
            self.data["Usage"] == usage
        ].reset_index(drop=True)

        if len(self.data) == 0:
            raise ValueError(f"No data found for usage: {usage}")

        self.images = self.data["pixels"].values
        self.labels = self.data["emotion"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        pixels = np.fromstring(
            self.images[index],
            dtype=np.float32,
            sep=" "
        )

        image = pixels.reshape(48, 48) / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    csv_path,
    batch_size=32,
    num_workers=0
):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
    ])

    train_dataset = FER2013Dataset(
        csv_path=csv_path,
        usage="Training",
        transform=train_transform
    )

    val_dataset = FER2013Dataset(
        csv_path=csv_path,
        usage="PublicTest"
    )

    test_dataset = FER2013Dataset(
        csv_path=csv_path,
        usage="PrivateTest"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
