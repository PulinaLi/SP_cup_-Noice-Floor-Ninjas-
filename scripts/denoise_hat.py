import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2

from model.hat import HAT
from classical import (
    adaptive_defect_correction,
    wavelet_denoise,
    bilateral_denoise
)

def parse_args():
    parser = argparse.ArgumentParser(description="HAT Inference")
    parser.add_argument("--noise_dir", required=True, type=Path)
    parser.add_argument("--denoised_dir", required=True, type=Path)
    parser.add_argument("--model_path", type=Path, default=Path("scripts/checkpoints/hat_best.pth"))
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def load_model(path, device):
    model = HAT(dim=24, num_blocks=[1, 1, 1, 2], num_heads=[1, 2, 4, 8])
    if path.exists():
        state_dict = torch.load(path, map_location=device)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded HAT weights from {path}")
    else:
        print(f"Warning: weights not found at {path}")
    model.to(device)
    model.eval()
    return model

def denoise_image(model, image, device):
    h, w, c = image.shape
    patch_size = 64
    stride = 32
    
    # Calculate padding needed to cover the whole image
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    padded_h, padded_w, _ = image.shape
    result = np.zeros_like(image)
    weights = np.zeros_like(image)
    
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        for y in range(0, padded_h - patch_size + 1, stride):
            for x in range(0, padded_w - patch_size + 1, stride):
                patch = tensor[:, :, y:y+patch_size, x:x+patch_size]
                out_patch = model(patch)
                out_patch = out_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Use a center-weighted window to blend overlapping patches
                result[y:y+patch_size, x:x+patch_size, :] += out_patch
                weights[y:y+patch_size, x:x+patch_size, :] += 1.0
                
    result = result / np.clip(weights, 1.0, None)
    result = np.clip(result, 0, 1)
    
    result = result[:h, :w, :]
    
    # Classical post-processing cascade
    result = adaptive_defect_correction(result)
    result = wavelet_denoise(result, wavelet='db4', level=1)
    result = bilateral_denoise(result, d=5, sigma_color=25, sigma_space=25)
        
    return result

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")
    
    model = load_model(args.model_path, device)
    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    
    paths = sorted([p for p in args.noise_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    
    for path in paths:
        img = Image.open(path).convert("RGB")
        img_float = np.asarray(img).astype(np.float32) / 255.0
        
        result = denoise_image(model, img_float, device)
        
        out_id = path.stem[:-6] if path.stem.endswith("_noise") else path.stem
        out_uint8 = (result * 255).astype(np.uint8)
        Image.fromarray(out_uint8).save(args.denoised_dir / f"{out_id}.png")
        print(f"HAT processed {out_id}.png")

if __name__ == "__main__":
    main()
