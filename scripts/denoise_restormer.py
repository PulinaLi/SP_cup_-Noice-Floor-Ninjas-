import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2

from model.restormer import Restormer
from classical import (
    adaptive_defect_correction,
    wavelet_denoise,
    bilateral_denoise
)

def parse_args():
    parser = argparse.ArgumentParser(description="Restormer Inference")
    parser.add_argument("--noise_dir", required=True, type=Path)
    parser.add_argument("--denoised_dir", required=True, type=Path)
    parser.add_argument("--model_path", type=Path, default=Path("scripts/checkpoints/restormer_best.pth"))
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()

def load_model(path, device):
    model = Restormer(dim=24, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], expansion_factor=2.0)
    if path.exists():
        state_dict = torch.load(path, map_location=device)
        if 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {path}")
    else:
        print(f"Warning: weights not found at {path}")
    model.to(device)
    model.eval()
    return model

def denoise_image(model, image, device):
    # Padding image to make dimensions divisible by 8 (for 3 downsamples)
    h, w, c = image.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h != 0 or pad_w != 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output = model(tensor)
    
    result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result, 0, 1)
    
    # Remove padding
    result = result[:h, :w, :]
    
    # Post-processing
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
        print(f"Restormer processed {out_id}.png")

if __name__ == "__main__":
    main()
