# 06 — Evaluation Strategy

## 6.1 Competition Metrics

### Composite Score Formula

The competition uses improvements over the noisy input as the scoring metric:

```
ΔPSNR = PSNR(denoised, clean) − PSNR(noisy, clean)
ΔSSIM = SSIM(denoised, clean) − SSIM(noisy, clean)

N = clip(ΔPSNR / 15, 0, 1)        ← normalized PSNR improvement
S = max(ΔSSIM, 0)                  ← clamped SSIM improvement

Composite Score = 0.6 × N + 0.4 × S
```

### Score Breakdown
| Component | Weight | Max Value | Meaning |
|-----------|--------|-----------|---------|
| N (Normalized ΔPSNR) | 60% | 1.0 (when ΔPSNR ≥ 15 dB) | Higher = more noise removed |
| S (ΔSSIM) | 40% | ~1.0 | Higher = better structural preservation |
| **Composite** | **100%** | **1.0** | Weighted combination |

### Score Targets

| Level | Composite | ΔPSNR | ΔSSIM | Assessment |
|-------|-----------|-------|-------|------------|
| Baseline (NLM) | 0.263 | +4.51 dB | +0.207 | Must beat this |
| Good | 0.45 | +7.5 dB | +0.30 | Competitive entry |
| Strong | 0.60 | +10.0 dB | +0.40 | Shortlisting range |
| Excellent | 0.75+ | +12.0+ dB | +0.50+ | Top-tier |

---

## 6.2 Self-Evaluation Workflow

### Using the Provided Evaluator

```bash
# 1. Generate denoised images
python scripts/denoise.py --noise_dir competition_data/public/noisy --denoised_dir competition_data/public/denoised

# 2. Evaluate against ground truth
python evaluation/evaluate.py \
    --noisy_dir competition_data/public/noisy \
    --pred_dir competition_data/public/denoised \
    --gt_dir competition_data/public/ground_truth
```

### Output Format
```
==============================================
EVALUATION COMPLETE
==============================================
Images scored:      460
Mean PSNR:           XX.XXXX dB
Mean SSIM:           0.XXXXXX
Mean Delta PSNR:     +X.XXXX dB
Mean Delta SSIM:     +0.XXXXXX

Composite Score:     0.XXXXXXXX
==============================================
```

---

## 6.3 Validation Strategy

### Approach 1: Hold-Out Validation (Recommended for Speed)

```
460 images
├── Train: images [001–420] → 420 pairs
└── Val:   images [421–460] → 40 pairs
```

- Evaluate composite score on 40 validation images
- This is the proxy for hidden 20-image performance
- Quick to compute, one model to manage

### Approach 2: K-Fold Cross-Validation (For Robustness)

```
Fold 1: Train [093–460], Val [001–092]
Fold 2: Train [001–092, 185–460], Val [093–184]
Fold 3: Train [001–184, 277–460], Val [185–276]
Fold 4: Train [001–276, 369–460], Val [277–368]
Fold 5: Train [001–368], Val [369–460]
```

- 5 models, each validated on unseen data
- More robust estimate of generalization
- Can ensemble all 5 models for final prediction

### Approach 3: Stratified Split (Best Estimate)

1. Compute noise level per image
2. Sort images by noise severity
3. Pick every Nth image for validation
4. Ensures all noise levels represented in both sets

---

## 6.4 Evaluation During Training

### Per-Epoch Validation

```python
def validate(model, val_loader, noisy_paths):
    """Compute competition metrics on validation set."""
    psnrs, ssims, composites = [], [], []
    
    for noisy, clean in val_loader:
        with torch.no_grad():
            pred = model(noisy.cuda()).cpu()
        
        # Compute metrics
        pred_np = pred.numpy()[0].transpose(1, 2, 0)
        clean_np = clean.numpy()[0].transpose(1, 2, 0)
        noisy_np = noisy.numpy()[0].transpose(1, 2, 0)
        
        psnr = peak_signal_noise_ratio(clean_np, pred_np, data_range=1.0)
        ssim = structural_similarity(clean_np, pred_np, ...)
        noisy_psnr = peak_signal_noise_ratio(clean_np, noisy_np, data_range=1.0)
        noisy_ssim = structural_similarity(clean_np, noisy_np, ...)
        
        delta_psnr = psnr - noisy_psnr
        delta_ssim = ssim - noisy_ssim
        composite = 0.6 * clip(delta_psnr/15, 0, 1) + 0.4 * max(delta_ssim, 0)
        
        composites.append(composite)
    
    return np.mean(composites)
```

### What to Track

```
Epoch  | Train Loss | Val Loss | Val PSNR | Val SSIM | ΔPSNR  | ΔSSIM  | Composite
-------|------------|----------|----------|----------|--------|--------|----------
  1    | 0.0342     | 0.0315   | 26.2     | 0.721    | +6.4   | +0.257 | 0.359
 10    | 0.0218     | 0.0205   | 28.1     | 0.785    | +8.3   | +0.321 | 0.460
 50    | 0.0152     | 0.0148   | 30.5     | 0.842    | +10.7  | +0.378 | 0.579
100    | 0.0131     | 0.0130   | 31.8     | 0.869    | +12.0  | +0.405 | 0.642
```

---

## 6.5 Per-Image Analysis

### Identify Failure Cases

After training, analyze per-image scores to find:
1. **Worst performers:** Which images have lowest composite score?
2. **Regression cases:** Any images where ΔPSNR or ΔSSIM is negative?
3. **Noise-level correlation:** Do high-noise images always score lower?

```python
# Generate per-image scores CSV
for img_id in val_ids:
    result = evaluate_image(img_id, pred_path, gt_path, noisy_path)
    print(f"{img_id}: PSNR={result['psnr']:.2f}, "
          f"ΔPSNR={result['delta_psnr']:+.2f}, "
          f"Composite={result['composite_score']:.4f}")
```

### Visualization
- Plot ΔPSNR vs. noise severity
- Show worst/best denoised examples side-by-side
- These visuals go into the report

---

## 6.6 SSIM Configuration (Match Official)

The evaluation script uses specific SSIM parameters. **Your validation code MUST match:**

```python
from skimage.metrics import structural_similarity as sk_ssim

ssim = sk_ssim(
    clean, pred,
    channel_axis=-1,
    data_range=1.0,
    win_size=7,
    gaussian_weights=False,
    use_sample_covariance=True,
    K1=0.01,
    K2=0.03,
)
```

> ⚠️ Using different SSIM parameters will give different scores than the official evaluator.

---

## 6.7 Final Submission Verification

Before submitting, run this verification:

```bash
# Denoise submission images
python scripts/denoise.py \
    --noise_dir competition_data/submissions/noisy \
    --denoised_dir competition_data/submissions/denoised

# Verify output
# Check: exactly 20 files (461.png – 480.png)
# Check: all PNG format, RGB, 992×992
# Check: filenames match convention
```

```python
# Verification script
from pathlib import Path
from PIL import Image

denoised_dir = Path("competition_data/submissions/denoised")
expected = [f"{i}.png" for i in range(461, 481)]

for fname in expected:
    path = denoised_dir / fname
    assert path.exists(), f"Missing: {fname}"
    img = Image.open(path)
    assert img.mode == "RGB", f"{fname}: not RGB"
    assert img.size == (992, 992), f"{fname}: wrong size {img.size}"
    print(f"✓ {fname}: {img.size}, {img.mode}")

print(f"\nAll {len(expected)} files verified!")
```
