"""
advanced_analysis.py — Comprehensive Dataset Noise Analysis
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

NOISY_DIR = Path("Dataset/public/noisy")
CLEAN_DIR = Path("Dataset/public/ground_truth")

def load_image(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def main():
    noisy_paths = sorted(NOISY_DIR.glob("*.png"))
    
    # We will sample 100 images randomly to do deep spatial and frequency analysis
    np.random.seed(42)
    sample_indices = np.random.choice(len(noisy_paths), min(100, len(noisy_paths)), replace=False)
    
    spatial_vars = []
    freq_power_low = []
    freq_power_high = []
    defect_pixel_counts = []
    
    print(f"Deep analyzing {len(sample_indices)} images...")
    
    for idx in sample_indices:
        noisy_path = noisy_paths[idx]
        img_id = noisy_path.stem.replace("_noise", "")
        clean_path = CLEAN_DIR / f"{img_id}.png"
        
        if not clean_path.exists():
            continue
            
        noisy = load_image(noisy_path)
        clean = load_image(clean_path)
        noise = noisy - clean
        
        # 2. Spatial Analysis: 
        # Check variance across different 64x64 patches to see if noise is uniform
        h, w, c = noise.shape
        patch_vars = []
        for i in range(0, h-64, 64):
            for j in range(0, w-64, 64):
                patch_vars.append(np.var(noise[i:i+64, j:j+64, :]))
        spatial_vars.append(np.std(patch_vars) / np.mean(patch_vars)) # coefficient of variation
        
        # Defect pixels (values that are way off despite low local variance)
        defect_pixel_counts.append(np.sum(np.abs(noise) > 0.4))
        
        # 3. Frequency Analysis
        # Check power in low vs high frequencies
        f_transform = np.fft.fft2(noise[:, :, 1]) # Green channel
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        rows, cols = magnitude.shape
        crow, ccol = rows//2, cols//2
        r = 100 # radius
        
        mask = np.zeros((rows, cols), np.uint8)
        cv2 = __import__('cv2')
        cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)
        
        low_freq_power = np.sum(magnitude * mask)
        high_freq_power = np.sum(magnitude * (1 - mask))
        freq_power_low.append(low_freq_power)
        freq_power_high.append(high_freq_power)

    print("\n" + "="*50)
    print("ADVANCED SPATIAL & FREQUENCY ANALYSIS")
    print("="*50)
    
    # 2. Spatial
    print(f"\nSpatial Uniformity:")
    print(f"  Patch Variance CoV: {np.mean(spatial_vars):.4f} (higher means more spatial patterns/non-uniformity)")
    print(f"  Avg defect pixels (deviation > 0.4): {np.mean(defect_pixel_counts):.1f} per image")
    
    # 3. Frequency
    print(f"\nFrequency Power Distribution (Green Channel):")
    print(f"  Low frequency power (radius 100): {np.mean(freq_power_low):.2e}")
    print(f"  High frequency power:             {np.mean(freq_power_high):.2e}")
    print(f"  High/Low Ratio:                   {np.mean(freq_power_high) / np.mean(freq_power_low):.2f}")
    if np.mean(freq_power_high) > np.mean(freq_power_low):
         print("  Conclusion: Noise is primarily high-frequency (white/blue noise characteristics).")
    else:
         print("  Conclusion: Noise has significant low-frequency structural components.")

if __name__ == "__main__":
    main()
