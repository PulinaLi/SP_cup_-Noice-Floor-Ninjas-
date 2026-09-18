# 07 — Codebase Structure

## 7.1 Repository Layout

The competition requires all custom code in `scripts/`. The overall repository must preserve the starter structure:

```
mora_sp_cup_2026/
│
├── baseline/                          # DO NOT MODIFY
│   ├── README.md
│   ├── denoise.py
│   └── requirements.txt
│
├── competition_data/                  # DO NOT MODIFY (except adding images)
│   ├── README.md
│   ├── public/
│   │   ├── ground_truth/             # 001.png – 460.png
│   │   ├── noisy/                    # 001_noise.png – 460_noise.png
│   │   └── denoised/                 # YOUR denoised outputs (public)
│   └── submissions/
│       ├── noisy/                    # 461_noise.png – 480_noise.png
│       └── denoised/                 # YOUR denoised outputs (submission)
│           ├── 461.png
│           ├── ...
│           └── 480.png
│
├── Dataset/                           # DO NOT MODIFY
│   └── DATASET_TERMS.txt
│
├── Delegate handbook/                 # Reference
│   └── Participant_Handbook.pdf
│
├── evaluation/                        # DO NOT MODIFY
│   ├── README.md
│   ├── evaluate.py
│   └── requirements.txt
│
├── scripts/                          # ★ YOUR CODE GOES HERE ★
│   ├── README.md                     # ★ Instructions on how to run
│   ├── requirements.txt              # ★ Dependencies
│   ├── denoise.py                    # ★ Main denoising script (REQUIRED)
│   ├── model/                        # ★ Model architecture definition
│   │   ├── __init__.py
│   │   ├── nafnet.py                 # NAFNet architecture
│   │   └── utils.py                  # Model utilities
│   ├── classical/                    # ★ Classical SP modules
│   │   ├── __init__.py
│   │   ├── wavelet.py               # Wavelet denoising
│   │   ├── bilateral.py             # Bilateral filtering
│   │   ├── defect_correction.py     # Enhanced defect pixel correction
│   │   └── frequency.py             # Frequency domain analysis
│   ├── train.py                      # Training script (for reference)
│   ├── dataset.py                    # Data loading utilities
│   └── config.py                     # Configuration constants
│
├── implementation plan/              # This planning folder
│
├── README.md                         # Project README
└── .gitignore
```

---

## 7.2 Main Script: `scripts/denoise.py`

This is the **most critical file**. It must:
1. Accept `--noise_dir` and `--denoised_dir` arguments
2. Load the trained model (or download from Drive link)
3. Process all noisy images in the input directory
4. Save denoised outputs with correct naming (`<id>.png`, no `_noise` suffix)
5. Run fully offline (no network requests during inference)
6. Work on CPU (with optional GPU acceleration)

### Required Interface

```python
"""
denoise.py — Main denoising script for Mora SP Cup 2026
Team: [TeamName]

USAGE:
    python scripts/denoise.py --noise_dir x --denoised_dir y

Reads every <id>_noise.<ext> file from --noise_dir
Writes <id>.<ext> to --denoised_dir
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image


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
    from model.nafnet import NAFNet
    
    model = NAFNet(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def strip_noise_suffix(stem):
    """Remove '_noise' suffix from filename stem."""
    if stem.lower().endswith("_noise"):
        return stem[:-6]
    return stem


def denoise_image(model, image, device):
    """Denoise a single image using the pipeline."""
    # 1. Classical pre-processing
    from classical.defect_correction import adaptive_defect_correction
    image = adaptive_defect_correction(image)
    
    # 2. Deep learning denoising
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
    
    result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = np.clip(result, 0, 1)
    
    return result


def main():
    args = parse_args()
    
    # Device selection with CPU fallback
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create output directory
    args.denoised_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all noisy images
    valid_ext = {".png", ".jpg", ".jpeg"}
    paths = sorted([p for p in args.noise_dir.glob("*") 
                    if p.suffix.lower() in valid_ext])
    
    if not paths:
        raise SystemExit(f"No images found in {args.noise_dir}")
    
    # Process each image
    total_time = 0.0
    for path in paths:
        img = Image.open(path).convert("RGB")
        img_float = np.asarray(img).astype(np.float32) / 255.0
        
        start = time.time()
        result = denoise_image(model, img_float, device)
        total_time += time.time() - start
        
        # Save with correct naming
        out_id = strip_noise_suffix(path.stem)
        out_uint8 = (result * 255).astype(np.uint8)
        Image.fromarray(out_uint8).save(args.denoised_dir / f"{out_id}.png")
    
    print(f"Processed {len(paths)} images in {total_time:.2f}s "
          f"({total_time/len(paths)*1000:.1f} ms/image)")
    print(f"Results saved to: {args.denoised_dir}")


if __name__ == "__main__":
    main()
```

---

## 7.3 Dependencies: `scripts/requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.8.0
scikit-image>=0.21.0
PyWavelets>=1.5.0
kornia>=0.7.0
```

---

## 7.4 `scripts/README.md` Template

```markdown
# Team [TeamName] — Denoising Solution

## Architecture
[Brief description of the approach]

## Requirements
pip install -r scripts/requirements.txt

## Model Setup
1. Download model weights from: [Google Drive link]
2. Place at: scripts/checkpoints/best_model.pth
3. SHA-256: [checksum]

## Usage

### Denoise images:
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised

### Evaluate (on public set):
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir competition_data/public/denoised --gt_dir competition_data/public/ground_truth

## Official Submission Information
Git Commit SHA: [commit SHA]
Model Checkpoint: best_model.pth
Model Drive Link: [link]
Expected Model Path: scripts/checkpoints/best_model.pth
Model SHA-256: [SHA-256]
```

---

## 7.5 Code Quality Guidelines (Worth 15%)

The 15% code quality mark covers:

### Clean Code
- ✅ Meaningful variable/function names
- ✅ Docstrings on all functions
- ✅ Type hints where appropriate
- ✅ Consistent formatting (use Black/autopep8)
- ✅ Modular design (separate files for model, data, classical SP)

### Adherence to Guidelines
- ✅ All code in `scripts/` only
- ✅ No modifications to `baseline/`, `evaluation/`, other directories
- ✅ Follows the baseline's input/output conventions
- ✅ README with clear instructions

### Runtime Optimization
- ✅ CPU-compatible inference
- ✅ Efficient memory usage (process one image at a time)
- ✅ Avoid unnecessary disk I/O
- ✅ Use `torch.no_grad()` during inference
- ✅ Consider model quantization for faster CPU inference

### No Hardcoding
- ❌ No image-specific processing (e.g., "if image_id == '461': ...")
- ❌ No lookup tables mapping image IDs to parameters
- ❌ No pre-computed results embedded in code
- ❌ **Violation = immediate disqualification**
