# 02 — Solution Architecture

## 2.1 Overall Strategy

We propose a **hybrid approach** combining classical signal processing with deep learning to maximize both the denoising score (35% weight) and the classical SP bonus (10% weight).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FULL DENOISING PIPELINE                         │
│                                                                     │
│  Input (Noisy Image)                                               │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │ Stage 1: Classical   │  ← Classical SP (bonus 10%)              │
│  │ Pre-Processing       │                                           │
│  │  • Defect Pixel Fix  │                                           │
│  │  • Wavelet Denoise   │                                           │
│  │  • Freq. Filtering   │                                           │
│  └─────────┬───────────┘                                           │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐                                           │
│  │ Stage 2: Deep        │  ← Main denoising engine                 │
│  │ Learning Model       │                                           │
│  │  • NAFNet / UNet     │                                           │
│  │  • Trained on 460    │                                           │
│  │    image pairs       │                                           │
│  └─────────┬───────────┘                                           │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────────┐                                           │
│  │ Stage 3: Classical   │  ← Post-processing refinement            │
│  │ Post-Processing      │                                           │
│  │  • Edge Enhancement  │                                           │
│  │  • Color Correction  │                                           │
│  │  • Detail Sharpening │                                           │
│  └─────────┬───────────┘                                           │
│            │                                                        │
│            ▼                                                        │
│  Output (Denoised Image)                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Model Architecture Options

### Option A: NAFNet (Recommended)
**Nonlinear Activation Free Network for Image Restoration**

- State-of-the-art denoising performance
- Efficient architecture (good for CPU inference)
- Uses simple channel attention and gating instead of complex attention
- Relatively lightweight compared to transformer-based models

```
NAFNet Architecture:
  Input → Conv → [NAFBlock × N] → Conv → Output (Residual)
  
  NAFBlock:
    LayerNorm → Conv → SimpleGate → SCA → Conv → Skip Connection
    (SCA = Simple Channel Attention)
```

**Pros:** High PSNR/SSIM, fast inference, well-documented  
**Cons:** May need careful tuning for this specific noise model

### Option B: Restormer (Lightweight Variant)
**Efficient Transformer for High-Resolution Image Restoration**

- Multi-Dconv Head Transposed Attention (MDTA)
- Gated-Dconv Feed-Forward Network (GDFN)
- Excellent for capturing long-range dependencies

**Pros:** Excellent restoration quality  
**Cons:** Heavier computation, may be slow on CPU

### Option C: UNet with Attention
**Classic encoder-decoder with skip connections**

- Simple, well-understood architecture
- Easy to implement and debug
- Good baseline for deep learning approach

**Pros:** Simple, reliable, fast on CPU  
**Cons:** May not achieve top-tier scores

### Option D: DnCNN / DRUNet
**Residual learning for denoising**

- Learn noise residual instead of clean image
- Proven approach with many variants available

**Pros:** Very fast, simple training  
**Cons:** Limited capacity for complex noise patterns

---

## 2.3 Recommended Architecture: NAFNet

### Why NAFNet?
1. **Top-tier performance** on standard denoising benchmarks
2. **Efficient inference** — feasible on CPU within reasonable time
3. **Simple building blocks** — no complex attention mechanisms
4. **Residual learning** — predicts clean image residual
5. **Well-suited for 992×992** — can process patches during training and full images during inference (with overlap-tile)

### NAFNet Configuration

```python
# Suggested NAFNet configuration for this competition
config = {
    'img_channel': 3,          # RGB input
    'width': 32,               # Base channel width (can try 48/64)
    'middle_blk_num': 12,      # Middle blocks
    'enc_blk_nums': [2, 2, 4, 8],   # Encoder blocks per level
    'dec_blk_nums': [2, 2, 2, 2],   # Decoder blocks per level
}
```

### Inference Strategy for 992×992 Images
Since 992×992 is large for a single forward pass on CPU:

1. **Overlap-tile strategy:**
   - Split image into overlapping patches (e.g., 256×256 with 32px overlap)
   - Process each patch independently
   - Blend overlapping regions using weighted average (Gaussian weights)
   - This ensures edge-artifact-free reconstruction

2. **Full-image inference** (if memory allows):
   - Process the entire 992×992 image at once
   - Faster but requires more memory

---

## 2.4 Alternative Approach: Ensemble Strategy

For maximum score, consider an ensemble:

```
Noisy → Model A (NAFNet) ──────────┐
                                     ├─→ Weighted Average → Output
Noisy → Model B (UNet/DnCNN) ─────┘
```

Or a self-ensemble using geometric augmentations:
```
Input → [8 augmented versions] → Model → [De-augment] → Average → Output
```
- 8 augmentations: 4 rotations × 2 flips
- Improves score at the cost of 8× inference time
- Worth it if CPU time budget allows

---

## 2.5 Architecture Decision Matrix

| Criterion | NAFNet | Restormer | UNet+Attn | DnCNN |
|-----------|--------|-----------|-----------|-------|
| PSNR/SSIM performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| CPU inference speed | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Training ease | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Model size | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Implementation complexity | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation:** Start with **NAFNet** as the primary model. If time permits, add a lighter **DnCNN** for ensembling.
