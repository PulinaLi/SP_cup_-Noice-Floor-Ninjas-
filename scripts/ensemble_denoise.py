"""
ensemble_denoise.py - Ensemble Inference for Mora SP Cup 2026

Blends the outputs of NAFNet and Attention U-Net.
"""

import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2

from model.nafnet import NAFNet
from model.unet import AttentionUNet
from classical import (
    adaptive_defect_correction,
    wavelet_denoise,
    bilateral_denoise
)

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble Denoiser")
    parser.add_argument("--noise_dir", required=True, type=Path)
    parser.add_argument("--denoised_dir", required=True, type=Path)
    parser.add_argument("--nafnet_path", type=Path, default=Path("scripts/checkpoints/nafnet_0.643.pth"))
    parser.add_argument("--unet_path", type=Path, default=Path("scripts/checkpoints/best_model.pth"))
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def load_nafnet(path, device):
    model = NAFNet(img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    if path.exists():
        state_dict = torch.load(path, map_location=device)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded NAFNet from {path}")
    model.to(device)
    model.eval()
    return model

def load_unet(path, device):
    model = AttentionUNet(img_ch=3, output_ch=3, filters=32)
    if path.exists():
        state_dict = torch.load(path, map_location=device)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded U-Net from {path}")
    model.to(device)
    model.eval()
    return model

def strip_noise_suffix(stem):
    if stem.lower().endswith("_noise"):
        return stem[:-6]
    return stem

def denoise_ensemble(nafnet, unet, image, device):
    """Run both models and blend the outputs."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        out_naf = nafnet(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_unet = unet(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # 50/50 Blend
    blended = (out_naf * 0.6) + (out_unet * 0.4) # Slightly weight NAFNet higher since it scored better
    blended = np.clip(blended, 0, 1)
    
    # Post-processing
    blended = adaptive_defect_correction(blended)
    blended = wavelet_denoise(blended, wavelet='db4', level=1)
    blended = bilateral_denoise(blended, d=5, sigma_color=25, sigma_space=25)
    
    return blended

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")
    
    nafnet = load_nafnet(args.nafnet_path, device)
    unet = load_unet(args.unet_path, device)
    
    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted([p for p in args.noise_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    
    total_time = 0.0
    for path in paths:
        img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        
        start = time.time()
        result = denoise_ensemble(nafnet, unet, img, device)
        total_time += time.time() - start
        
        out_id = strip_noise_suffix(path.stem)
        Image.fromarray((result * 255).astype(np.uint8)).save(args.denoised_dir / f"{out_id}.png")
        print(f"Ensembled {out_id}.png")
    
    print(f"Processed {len(paths)} images in {total_time:.2f}s ({total_time/len(paths)*1000:.1f} ms/image)")
    print(f"Ensemble Results saved to: {args.denoised_dir}")

if __name__ == "__main__":
    main()
