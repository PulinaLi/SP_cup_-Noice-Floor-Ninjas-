import argparse
import time
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from dataset import DenoisingDataset
from model.hat import HAT

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.l1(pred, target)

def calculate_ssim(clean, pred):
    return ssim_metric(
        clean, pred,
        channel_axis=-1,
        data_range=1.0,
        win_size=7,
        gaussian_weights=False,
        use_sample_covariance=True,
        K1=0.01,
        K2=0.03,
    )

def validate(model, val_loader, device):
    model.eval()
    psnrs, ssims, composites = [], [], []
    
    for batch in val_loader:
        noisy = batch["noisy"]
        clean = batch["clean"]
        
        with torch.no_grad():
            pred = model(noisy.to(device)).cpu()
        
        for i in range(len(noisy)):
            pred_np = pred[i].numpy().transpose(1, 2, 0)
            clean_np = clean[i].numpy().transpose(1, 2, 0)
            noisy_np = noisy[i].numpy().transpose(1, 2, 0)
            
            pred_np = np.clip(pred_np, 0.0, 1.0)
            
            psnr = psnr_metric(clean_np, pred_np, data_range=1.0)
            ssim = calculate_ssim(clean_np, pred_np)
            noisy_psnr = psnr_metric(clean_np, noisy_np, data_range=1.0)
            noisy_ssim = calculate_ssim(clean_np, noisy_np)
            
            delta_psnr = psnr - noisy_psnr
            delta_ssim = ssim - noisy_ssim
            
            N = np.clip(delta_psnr / 15.0, 0, 1)
            S = max(delta_ssim, 0)
            composite = 0.6 * N + 0.4 * S
            
            composites.append(composite)
            psnrs.append(psnr)
            ssims.append(ssim)
            
    model.train()
    return np.mean(composites), np.mean(psnrs), np.mean(ssims)

def parse_args():
    parser = argparse.ArgumentParser(description="Train HAT")
    parser.add_argument("--noisy_dir", required=True, type=str)
    parser.add_argument("--clean_dir", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=50)
    # Scaled down for VRAM
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="scripts/checkpoints")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training HAT on device: {device}")

    full_dataset = DenoisingDataset(args.noisy_dir, args.clean_dir, patch_size=args.patch_size, is_train=True)
    total_imgs = len(full_dataset)
    val_size = min(40, total_imgs // 10)
    train_size = total_imgs - val_size
    
    indices = list(range(total_imgs))
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Ultra-lightweight HAT to prevent OOM
    model = HAT(dim=24, num_blocks=[1, 1, 1, 2], num_heads=[1, 2, 4, 8]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_composite = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
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
        
        val_composite, val_psnr, val_ssim = validate(model, val_loader, device)
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch}/{args.epochs}] - Time: {elapsed:.2f}s")
        print(f"--> Loss: {avg_loss:.6f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | COMPOSITE: {val_composite:.4f}")
        
        if val_composite > best_composite:
            best_composite = val_composite
            save_path = checkpoint_dir / "hat_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** Saved NEW best HAT model with Composite: {best_composite:.4f} ***")

if __name__ == "__main__":
    main()
