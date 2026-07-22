from __future__ import annotations
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultimodalPlantDataset(Dataset):
    def __init__(self, manifest: str, transform, image_col: str, label_col: str,
                 context_cols: Sequence[str], missingness_cols: Sequence[str], quality_cols: Sequence[str]):
        self.df = pd.read_csv(manifest)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.context_cols = list(context_cols)
        self.missingness_cols = list(missingness_cols)
        self.quality_cols = list(quality_cols)
        required = [image_col, label_col] + self.context_cols + self.missingness_cols + self.quality_cols
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing manifest columns: {missing}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(str(row[self.image_col]))
        if image is None:
            raise FileNotFoundError(row[self.image_col])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        context = np.asarray([row[c] for c in self.context_cols + self.missingness_cols], dtype=np.float32)
        quality = np.asarray([row[c] for c in self.quality_cols], dtype=np.float32)
        return {
            "image": image,
            "context": torch.from_numpy(context),
            "quality": torch.from_numpy(quality),
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "sample_id": str(row.get("sample_id", idx)),
            "image_path": str(row[self.image_col])
        }
