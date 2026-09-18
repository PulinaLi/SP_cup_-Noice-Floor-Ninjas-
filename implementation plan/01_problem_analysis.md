# 01 — Problem Analysis

## 1.1 Challenge Summary

The Mora SP Cup 2026 is a **low-light image denoising** competition. Camera sensors under limited illumination produce images corrupted by:
- **Read noise** (electronic noise from sensor readout)
- **Thermal noise** (from sensor heating)
- **Grain / shot noise** (photon counting randomness)
- **Visual distortion** artifacts

The dataset uses a **synthetic low-light noise model** applied to clean mobile-phone photographs to reproduce these sensor-related artifacts.

---

## 1.2 Dataset Characteristics

| Property | Value |
|----------|-------|
| **Resolution** | 992 × 992 pixels |
| **Format** | PNG (RGB) |
| **Total images** | 500 noisy + 500 ground truth |
| **Public set** | 460 pairs (001–460) — for training & validation |
| **Preliminary submission set** | 20 noisy only (461–480) — denoise & submit |
| **Final-round set** | 20 pairs (481–500) — hidden, used at physical finals |
| **Noise severity** | Multiple levels (mixed within the dataset) |

### Key Observations (from full 460 image analysis)
- **Noise Severity:** The dataset is heavily skewed towards high noise. Out of 460 images, 419 have high noise (std ≥ 0.06), 41 have medium noise (std 0.03–0.06), and 0 have low noise (std < 0.03).
- **Spatial Distribution:** The noise is somewhat non-uniform across the image, with a patch variance coefficient of variation (CoV) of 0.24. This implies that some spatial patterns exist.
- **Defect Pixels:** There is a significant amount of defect pixels (outliers). We found an average of ~8,757 defect pixels per image (deviation > 0.4 from local median).
- **Color Cast:** The noise standard deviation is fairly uniform across color channels, though slightly higher in the Red channel (R: 0.1088, G: 0.1047, B: 0.1029).
- **Frequency Profile:** The noise is primarily high-frequency (high/low power ratio = 25.4), characteristic of white/blue noise.
- **Brightness Range:** The ground truth images are naturally dim, confirming no brightness enhancement is needed.

---

## 1.3 Baseline Analysis

The provided baseline uses a **purely classical, CPU-only** pipeline:

```
Input → Defect Pixel Correction → Non-Local Means Denoising → Output
```

### Baseline Components

#### Defect Pixel Correction
- Per-channel median-filter outlier detection
- Threshold: 0.25 deviation from local median
- Kernel size: 3×3
- Replaces outlier pixels with median values

#### Non-Local Means (NLM) Denoising
- OpenCV `fastNlMeansDenoisingColored()`
- h = 10.0 (denoising strength)
- Template window: 7×7
- Search window: 21×21

### Baseline Results (460 public images)

| Metric | Value |
|--------|-------|
| Mean PSNR | 24.3360 dB |
| Mean SSIM | 0.671659 |
| Mean ΔPSNR | +4.5103 dB |
| Mean ΔSSIM | +0.207275 |
| **Composite Score** | **0.26329** |

### Baseline Strengths
- Simple, fast, no training needed
- Defect pixel correction handles hot/dead pixels

### Baseline Weaknesses
- NLM tends to over-smooth textures at higher h values
- No awareness of noise severity per image
- No learning from the 460 training pairs
- Cannot adapt to signal-dependent noise patterns
- No frequency-domain processing

---

## 1.4 Noise Analysis Results

We executed a comprehensive noise analysis on the full 460 public image pairs.

**1. Overall Noise Statistics**
- Mean noise std: 0.1063 (range: 0.0308 – 0.1687)
- Mean noise mean: +0.0053
- Mean noisy PSNR: 19.83 dB (range: 15.45 – 30.17 dB)

**2. Spatial Analysis**
- Patch Variance CoV: 0.2400 (moderate non-uniformity, meaning noise variance changes somewhat across different regions of an image)
- Average defect pixels: 8,757 per image (pixels with extreme deviation)

**3. Frequency Domain Analysis**
- Low-frequency power (radius 100): 3.37e+06
- High-frequency power: 8.56e+07
- High/Low Ratio: 25.43
- *Conclusion:* Noise is predominantly high-frequency (like white or blue noise). High-frequency filtering (like wavelets or NLM) will be very effective.

**4. Severity Level Clustering**
- Low noise (std < 0.03): 0 images
- Medium noise (0.03 ≤ std < 0.06): 41 images (8.9%)
- High noise (std ≥ 0.06): 419 images (91.1%)
- *Conclusion:* The model must be robust to very heavy noise, as >90% of the dataset falls into the high-severity category.

**5. Color Channel Analysis (Std Dev)**
- R: 0.1088
- G: 0.1047
- B: 0.1029
- *Conclusion:* The noise is well-balanced across RGB channels, without a dominant Bayer-pattern color bias.

---

## 1.5 What the Noisy-vs-Clean Relationship Tells Us

From the baseline code comments:
> "No brightness/tone correction is applied — input and ground truth are both naturally dim evening/night images, differing mainly by noise, so there is no exposure gap to fix."

This means:
- ✅ No brightness correction needed
- ✅ Focus purely on noise removal
- ✅ Ground truth is also dim — not a low-light enhancement task
- ✅ This simplifies the problem to pure denoising
