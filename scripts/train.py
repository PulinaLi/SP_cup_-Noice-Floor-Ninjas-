import argparse
import time
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Ensure these modules are available in your PYTHONPATH or relative path
from dataset import DenoisingDataset
from model.nafnet import NAFNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# A basic SSIM loss implementation or use structural_similarity from scikit-image
# For simplicity, using a basic L1 loss in this skeleton, but easily extensible to L1 + SSIM
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
    
    def forward(self, pred, target):
        return self.l1_weight * self.l1(pred, target)

def parse_args():
    parser = argparse.ArgumentParser(description="Train NAFNet Denoiser")
    parser.add_argument("--noisy_dir", required=True, type=str, help="Training noisy images")
    parser.add_argument("--clean_dir", required=True, type=str, help="Training clean images")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="scripts/checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Datasets and Loaders
    train_dataset = DenoisingDataset(args.noisy_dir, args.clean_dir, patch_size=args.patch_size, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    model = NAFNet(img_channel=3, width=32, middle_blk_num=12, 
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).to(device)

    # Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.2e} - Time: {elapsed:.2f}s")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
