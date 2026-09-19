# Team [Your Team Name] - Denoising Solution

## Architecture
Our solution uses a **3-Stage Hybrid Pipeline**:
1. **Classical Pre-processing**: Adaptive defect pixel correction + Wavelet denoising
2. **Deep Learning**: NAFNet (Nonlinear Activation Free Network) model
3. **Classical Post-processing**: Edge-preserving Bilateral filtering

## Setup & Installation

1. Install all dependencies:
```bash
pip install -r scripts/requirements.txt
```

2. Download the model weights:
- Download the `best_model.pth` from the Google Drive link provided below.
- Place the file exactly at: `scripts/checkpoints/best_model.pth`

## Usage

### To denoise the submission images:
```bash
python scripts/denoise.py --noise_dir competition_data/submissions/noisy --denoised_dir competition_data/submissions/denoised
```

### To evaluate on the public set (for self-verification):
```bash
python scripts/denoise.py --noise_dir competition_data/public/noisy --denoised_dir competition_data/public/denoised
python evaluation/evaluate.py --noisy_dir competition_data/public/noisy --pred_dir competition_data/public/denoised --gt_dir competition_data/public/ground_truth
```

## Output Format
The `denoise.py` script automatically strips the `_noise` suffix and outputs perfectly formatted `461.png` to `480.png` images in RGB 992x992 format as required by the competition.

---

> **Important - Submission Freeze**
> After the preliminary-round submission deadline, no new commits may be made to the submitted private GitHub repository.

## Official Submission Information

**Git Commit SHA:**
`16d28b1f3264aa90d69a9a33f37fa9c70a7f97e0`

**Model Checkpoint:**
`best_model.pth`

**Model Drive Link:**
`[TODO: Paste your Google Drive link here]`

**Expected Model Path:**
`scripts/checkpoints/best_model.pth`

**Model SHA-256:**
`3d9827261349407725143f805fec9f0c0982580903eb8ddf6df0ec71e1fc96c7`
