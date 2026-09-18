import argparse
import time
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# scikit-image metrics for evaluation
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

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

class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
    
    def forward(self, pred, target):
        return self.l1_weight * self.l1(pred, target)

def calculate_ssim(clean, pred):
    """Calculate SSIM using strict competition parameters."""
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
    """Compute competition metrics on validation set."""
    model.eval()
    psnrs, ssims, composites = [], [], []
    
    # We only evaluate on the first few batches to save time, or the whole val set
    for batch in val_loader:
        noisy = batch["noisy"]
        clean = batch["clean"]
        
        with torch.no_grad():
            pred = model(noisy.to(device)).cpu()
        
        # Compute metrics per image in batch
        for i in range(len(noisy)):
            pred_np = pred[i].numpy().transpose(1, 2, 0)
            clean_np = clean[i].numpy().transpose(1, 2, 0)
            noisy_np = noisy[i].numpy().transpose(1, 2, 0)
            
            # Clip predictions
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

    # Load Full Dataset
    full_dataset = DenoisingDataset(args.noisy_dir, args.clean_dir, patch_size=args.patch_size, is_train=True)
    total_imgs = len(full_dataset)
    
    # Validation Split: Reserve last 40 images for validation (Hold-out method)
    val_size = min(40, total_imgs // 10)
    train_size = total_imgs - val_size
    
    indices = list(range(total_imgs))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # We turn off random cropping for validation by overriding patch_size to None, 
    # but since Subset doesn't let us change attributes, we load full images for val
    # To avoid OOM during val, we'll just evaluate on cropped patches of the same size.
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0)

    print(f"Dataset Split -> Train: {train_size}, Validation: {val_size}")

    # Model
    model = NAFNet(img_channel=3, width=32, middle_blk_num=12, 
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).to(device)

    # Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_composite = -1.0

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
        
        # Validation Evaluation
        print("Evaluating validation set...")
        val_composite, val_psnr, val_ssim = validate(model, val_loader, device)
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch}/{args.epochs}] - Time: {elapsed:.2f}s - LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"--> Train Loss: {avg_loss:.6f} | Val PSNR: {val_psnr:.2f} | Val SSIM: {val_ssim:.4f} | COMPOSITE: {val_composite:.4f}")
        
        # Save Best Model based on Composite Score
        if val_composite > best_composite:
            best_composite = val_composite
            save_path = checkpoint_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** Saved NEW best model with Composite: {best_composite:.4f} ***")

    print(f"Training complete! Best validation composite score: {best_composite:.4f}")

if __name__ == "__main__":
    main()
