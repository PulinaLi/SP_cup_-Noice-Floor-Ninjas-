import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir=None, patch_size=256, is_train=True):
        """
        Args:
            noisy_dir: Path to directory with noisy images
            clean_dir: Path to directory with ground truth images (None for test set)
            patch_size: Size of random crops for training
            is_train: Whether to apply data augmentation
        """
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir) if clean_dir else None
        self.patch_size = patch_size
        self.is_train = is_train
        
        valid_ext = {".png", ".jpg", ".jpeg"}
        self.noisy_paths = sorted([p for p in self.noisy_dir.glob("*") if p.suffix.lower() in valid_ext])
        
        if self.clean_dir:
            self.clean_paths = []
            for p in self.noisy_paths:
                # Strip '_noise' suffix to find the corresponding ground truth image
                base_name = p.stem
                if base_name.endswith("_noise"):
                    base_name = base_name[:-6]
                clean_path = self.clean_dir / f"{base_name}{p.suffix}"
                if clean_path.exists():
                    self.clean_paths.append(clean_path)
                else:
                    print(f"Warning: Clean image not found for {p.name}")
                    self.clean_paths.append(None)
        else:
            self.clean_paths = [None] * len(self.noisy_paths)

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_path = self.noisy_paths[idx]
        noisy_img = Image.open(noisy_path).convert("RGB")
        
        if self.clean_paths[idx] is not None:
            clean_path = self.clean_paths[idx]
            clean_img = Image.open(clean_path).convert("RGB")
        else:
            # If no ground truth, return the noisy image as the target (for inference dataloaders)
            clean_img = noisy_img.copy()

        # Convert to tensor (C, H, W), float [0, 1]
        noisy_tensor = TF.to_tensor(noisy_img)
        clean_tensor = TF.to_tensor(clean_img)

        # Apply random crop if training
        if self.is_train and self.patch_size:
            _, h, w = noisy_tensor.shape
            if h >= self.patch_size and w >= self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                noisy_tensor = TF.crop(noisy_tensor, top, left, self.patch_size, self.patch_size)
                clean_tensor = TF.crop(clean_tensor, top, left, self.patch_size, self.patch_size)

        # Data augmentation
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                noisy_tensor = TF.hflip(noisy_tensor)
                clean_tensor = TF.hflip(clean_tensor)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                noisy_tensor = TF.vflip(noisy_tensor)
                clean_tensor = TF.vflip(clean_tensor)
            
            # Random Rotation (90, 180, 270)
            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                noisy_tensor = TF.rotate(noisy_tensor, rot)
                clean_tensor = TF.rotate(clean_tensor, rot)

        return {"noisy": noisy_tensor, "clean": clean_tensor, "name": noisy_path.name}
