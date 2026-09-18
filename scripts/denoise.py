"""
denoise.py - Main denoising script for Mora SP Cup 2026

USAGE:
    python scripts/denoise.py --noise_dir x --denoised_dir y

Reads every <id>_noise.<ext> file from --noise_dir
Writes <id>.<ext> to --denoised_dir
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from model.nafnet import NAFNet

def parse_args():
    parser = argparse.ArgumentParser(description="Low-light image denoiser")
    parser.add_argument("--noise_dir", required=True, type=Path,
                        help="Directory containing noisy input images")
    parser.add_argument("--denoised_dir", required=True, type=Path,
                        help="Directory to save denoised output images")
    parser.add_argument("--model_path", type=Path, 
                        default=Path("scripts/checkpoints/best_model.pth"),
                        help="Path to trained model weights")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for inference")
    return parser.parse_args()

def load_model(model_path, device):
    """Load trained model with CPU fallback."""
    # NAFNet default configuration for 992x992
    model = NAFNet(img_channel=3, width=32, middle_blk_num=12, 
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using random weights!")
        
    model.to(device)
    model.eval()
    return model

def adaptive_defect_correction(img_float, base_threshold=0.25, kernel_sizes=[3, 5]):
    """
    Stage 1: Corrects extremely hot/dead pixels which might throw off the CNN.
    """
    corrected = img_float.copy()
    for c in range(img_float.shape[2]):
        channel = img_float[..., c]
        ch_u8 = (channel * 255).astype(np.uint8)
        median_u8 = cv2.medianBlur(ch_u8, kernel_sizes[0])
        median_ref = median_u8.astype(np.float32) / 255.0

        diff_sq = (channel - median_ref) ** 2
        local_var = cv2.GaussianBlur(diff_sq, (15, 15), 0)
        local_std = np.sqrt(local_var + 1e-10)
        adaptive_thresh = base_threshold + 2.0 * local_std

        for k in kernel_sizes:
            k_u8 = cv2.medianBlur(ch_u8, k)
            k_float = k_u8.astype(np.float32) / 255.0
            deviation = np.abs(channel - k_float)
            outlier_mask = deviation > adaptive_thresh
            corrected[..., c] = np.where(outlier_mask, k_float, corrected[..., c])
    return corrected

def strip_noise_suffix(stem):
    if stem.lower().endswith("_noise"):
        return stem[:-6]
    return stem

def denoise_image(model, image, device):
    """Denoise a single image using the hybrid pipeline."""
    # 1. Classical pre-processing (Defect Correction)
    image = adaptive_defect_correction(image)
    
    # 2. Deep learning denoising (NAFNet)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
    
    result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result, 0, 1)
    
    return result

def main():
    args = parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    model = load_model(args.model_path, device)
    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    
    valid_ext = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in args.noise_dir.glob("*") if p.suffix.lower() in valid_ext])
    
    if not paths:
        raise SystemExit(f"No images found in {args.noise_dir}")
    
    total_time = 0.0
    for path in paths:
        img = Image.open(path).convert("RGB")
        img_float = np.asarray(img).astype(np.float32) / 255.0
        
        start = time.time()
        result = denoise_image(model, img_float, device)
        total_time += time.time() - start
        
        out_id = strip_noise_suffix(path.stem)
        out_uint8 = (result * 255).astype(np.uint8)
        Image.fromarray(out_uint8).save(args.denoised_dir / f"{out_id}.png")
        print(f"Processed {out_id}.png")
    
    print(f"Processed {len(paths)} images in {total_time:.2f}s "
          f"({total_time/len(paths)*1000:.1f} ms/image)")
    print(f"Results saved to: {args.denoised_dir}")

if __name__ == "__main__":
    main()
