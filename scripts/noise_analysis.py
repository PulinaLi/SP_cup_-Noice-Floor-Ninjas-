"""
noise_analysis.py — Analyze noise characteristics of the dataset.
Run from repo root:
    python scripts/noise_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

NOISY_DIR = Path("Dataset/public/noisy")
CLEAN_DIR = Path("Dataset/public/ground_truth")

# Analyze all images
SAMPLE_SIZE = 460


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def main():
    noisy_paths = sorted(NOISY_DIR.glob("*.png"))[:SAMPLE_SIZE]
    
    all_noise_stds = []
    all_noise_means = []
    all_psnrs = []
    per_channel_stds = {"R": [], "G": [], "B": []}
    
    print(f"Analyzing {len(noisy_paths)} image pairs...\n")
    
    for noisy_path in noisy_paths:
        img_id = noisy_path.stem.replace("_noise", "")
        clean_path = CLEAN_DIR / f"{img_id}.png"
        
        if not clean_path.exists():
            continue
        
        noisy = load_image(noisy_path)
        clean = load_image(clean_path)
        
        # Noise residual
        noise = noisy - clean
        
        # Overall stats
        noise_std = np.std(noise)
        noise_mean = np.mean(noise)
        all_noise_stds.append(noise_std)
        all_noise_means.append(noise_mean)
        
        # Per-channel stats
        for c, name in enumerate(["R", "G", "B"]):
            per_channel_stds[name].append(np.std(noise[:, :, c]))
        
        # PSNR of noisy image
        mse = np.mean((noisy - clean) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        all_psnrs.append(psnr)
        
        print(f"  {img_id}: noise_std={noise_std:.4f}, "
              f"noise_mean={noise_mean:.5f}, "
              f"noisy_psnr={psnr:.2f} dB")
    
    print("\n" + "=" * 60)
    print("NOISE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Images analyzed: {len(all_noise_stds)}")
    print(f"\nOverall Noise Statistics:")
    print(f"  Mean noise std:  {np.mean(all_noise_stds):.4f} "
          f"(range: {np.min(all_noise_stds):.4f} – {np.max(all_noise_stds):.4f})")
    print(f"  Mean noise mean: {np.mean(all_noise_means):.5f}")
    print(f"  Mean noisy PSNR: {np.mean(all_psnrs):.2f} dB "
          f"(range: {np.min(all_psnrs):.2f} – {np.max(all_psnrs):.2f})")
    
    print(f"\nPer-Channel Noise Std:")
    for name in ["R", "G", "B"]:
        vals = per_channel_stds[name]
        print(f"  {name}: mean={np.mean(vals):.4f}, "
              f"range={np.min(vals):.4f} – {np.max(vals):.4f}")
    
    # Check for signal-dependent noise
    print(f"\nNoise Level Severity Distribution:")
    stds_arr = np.array(all_noise_stds)
    low = np.sum(stds_arr < 0.03)
    med = np.sum((stds_arr >= 0.03) & (stds_arr < 0.06))
    high = np.sum(stds_arr >= 0.06)
    print(f"  Low noise (std < 0.03):   {low} images")
    print(f"  Med noise (0.03-0.06):    {med} images")
    print(f"  High noise (std >= 0.06): {high} images")


if __name__ == "__main__":
    main()
