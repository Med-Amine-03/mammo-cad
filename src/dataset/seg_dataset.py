
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF


class SegDataset(Dataset):
    def __init__(self, images: list, masks: list,
                 augment: bool = False):
        assert len(images) == len(masks), \
            f"Mismatch: {len(images)} images vs {len(masks)} masks"
        self.images  = [Path(p) for p in images]
        self.masks   = [Path(p) for p in masks]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img  = cv2.imread(str(self.images[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.masks[idx]),  cv2.IMREAD_GRAYSCALE)

        assert img  is not None, f"Cannot read: {self.images[idx]}"
        assert mask is not None, f"Cannot read: {self.masks[idx]}"

        img  = torch.tensor(img.astype(np.float32)  / 255.0).unsqueeze(0)
        mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img  = TF.rotate(img,  angle)
                mask = TF.rotate(mask, angle)

        return img, mask   


def load_fold_paths(patches_dir: Path):
    images = sorted((patches_dir / "images").glob("*.png"))
    masks  = sorted((patches_dir / "masks").glob("*.png"))
    assert len(images) == len(masks), \
        f"Count mismatch in {patches_dir}"
    assert all(i.name == m.name for i, m in zip(images, masks)), \
        "Image/mask filename mismatch!"
    return images, masks