# 03 — Data Pipeline

## 3.1 Dataset Structure

```
Dataset/
├── public/
│   ├── ground_truth/     # 001.png – 460.png (clean images)
│   └── noisy/            # 001_noise.png – 460_noise.png
└── submissions/
    └── noisy/            # 461_noise.png – 480_noise.png (no ground truth)
```

**Image specs:**
- Resolution: 992 × 992 pixels
- Format: PNG, RGB (3 channels)
- Pixel values: 0–255 (uint8)
- Content: Low-light / evening / night scenes from mobile phones

---

## 3.2 Train/Validation Split Strategy

We have **460 image pairs** available. The validation strategy directly impacts our ability to estimate performance on the hidden 20 images (461–480).

### Recommended Split: Stratified by Noise Level

```
460 Image Pairs
    │
    ├── Training Set: ~420 pairs (91%)
    │     (images used for model weight updates)
    │
    └── Validation Set: ~40 pairs (9%)
          (images NEVER seen during training)
          (used to estimate submission score)
```

### Why This Split?
- 40 validation images ≈ 2× the submission set size (20)
- Enough training data for a deep model
- Stratify by noise severity to ensure all difficulty levels represented in both sets

### How to Stratify
1. Compute noise std for each image: `std(noisy - clean)`
2. Sort by noise severity
3. Assign every 11th or 12th image to validation
4. This ensures equal noise-level representation

### Alternative: K-Fold Cross-Validation
- **5-fold CV:** Train 5 models on different 368/92 splits
- Use ensemble of 5 models for final prediction
- More robust but 5× training time
- Worth it if GPU time is available

---

## 3.3 Data Loading

### PyTorch Dataset Class

```python
class DenoisingDataset(Dataset):
    """Paired noisy/clean image dataset for training."""
    
    def __init__(self, noisy_dir, clean_dir, image_ids, 
                 patch_size=256, augment=True):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.image_ids = image_ids
        self.patch_size = patch_size
        self.augment = augment
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        noisy = load_image(self.noisy_dir / f"{img_id}_noise.png")
        clean = load_image(self.clean_dir / f"{img_id}.png")
        
        # Random crop to patch_size
        noisy, clean = random_crop(noisy, clean, self.patch_size)
        
        # Data augmentation
        if self.augment:
            noisy, clean = random_augment(noisy, clean)
        
        return noisy, clean
```

---

## 3.4 Data Augmentation Strategy

Since we have only 460 training pairs, aggressive augmentation is critical.

### Geometric Augmentations
| Augmentation | Description | Probability |
|---|---|---|
| Random Horizontal Flip | Mirror left-right | 0.5 |
| Random Vertical Flip | Mirror top-bottom | 0.5 |
| Random Rotation | 0°, 90°, 180°, 270° | 0.25 each |
| Random Crop | Extract patches from full image | Always |

### Patch-Based Training
- **Training patch size:** 256×256 (or 128×128 for faster training)
- **Patches per image per epoch:** 4–8 random crops
- **Effective training size:** 460 × 6 × 8 (augmentations × patches) ≈ **22,080 samples/epoch**

### What NOT to Do
- ❌ Don't add synthetic noise (the noise model is already in the data)
- ❌ Don't change brightness/contrast (clean images are also dim)
- ❌ Don't mix noisy-noisy pairs (only noisy-clean pairs)
- ❌ Don't resize images (evaluation is at 992×992)

---

## 3.5 Pixel Normalization

```python
# Option A: Scale to [0, 1] (recommended)
image = image.astype(np.float32) / 255.0

# Option B: Scale to [-1, 1]
image = image.astype(np.float32) / 127.5 - 1.0

# Recommendation: Use [0, 1] — same as baseline and evaluation script
```

---

## 3.6 DataLoader Configuration

```python
# Training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,           # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Validation DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=1,           # Full image evaluation
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
```

---

## 3.7 Pre-computed Noise Statistics

Before training, pre-compute and save noise statistics for the entire dataset:

```python
# noise_stats.json
{
    "001": {"noise_std": 0.042, "noise_mean": -0.001, "psnr": 22.1},
    "002": {"noise_std": 0.038, "noise_mean": 0.000, "psnr": 23.5},
    ...
}
```

These statistics help:
- Stratify train/val splits
- Understand noise level distribution
- Guide hyperparameter choices (e.g., NLM strength per severity)
- Inform the report analysis section
