# 04 — Model Training Plan

## 4.1 Training Environment

### Free GPU Options

| Platform | GPU | VRAM | Free Hours | Best For |
|----------|-----|------|------------|----------|
| **Google Colab** | T4 | 16 GB | ~12h/session | Quick experiments |
| **Kaggle** | T4/P100 | 16 GB | 30h/week | Extended training |

### Recommended Approach
- **Primary training:** Kaggle (more predictable GPU allocation)
- **Quick experiments:** Colab (faster startup)
- **Final testing:** Local CPU (verify CPU fallback)

---

## 4.2 Training Recipe

### Phase 1: Initial Training (Epochs 1–100)

```python
training_config = {
    # Model
    'model': 'NAFNet',
    'width': 32,                    # Start with 32, try 48 later
    
    # Data
    'patch_size': 256,              # Random crops from 992×992
    'batch_size': 8,                # Adjust for GPU memory
    'num_workers': 4,
    
    # Optimizer
    'optimizer': 'AdamW',
    'lr': 1e-3,                     # Initial learning rate
    'weight_decay': 1e-3,
    'betas': (0.9, 0.999),
    
    # Scheduler
    'scheduler': 'CosineAnnealingLR',
    'T_max': 100,                   # Total epochs
    'eta_min': 1e-6,                # Minimum LR
    
    # Loss
    'loss': 'L1 + SSIM',           # Combined loss
    'l1_weight': 1.0,
    'ssim_weight': 0.1,
    
    # Training
    'epochs': 100,
    'save_every': 10,
    'eval_every': 5,
}
```

### Phase 2: Fine-tuning (Epochs 101–150)

```python
finetune_config = {
    'lr': 1e-4,                     # Lower learning rate
    'patch_size': 384,              # Larger patches
    'batch_size': 4,                # Smaller batch for larger patches
    'epochs': 50,                   # Additional epochs
    'loss': 'Charbonnier + SSIM',   # Smoother loss for fine-tuning
}
```

### Phase 3: Large-Patch Fine-tuning (Epochs 151–200)

```python
largepatch_config = {
    'lr': 5e-5,
    'patch_size': 512,              # Even larger patches
    'batch_size': 2,
    'epochs': 50,
}
```

---

## 4.3 Loss Functions

### Primary Loss: L1 + SSIM

```python
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, ssim_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(data_range=1.0, channel=3)
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        return (self.l1_weight * self.l1(pred, target) + 
                self.ssim_weight * (1 - self.ssim(pred, target)))
```

### Why L1 + SSIM?
- **L1 Loss:** Directly optimizes for pixel-level accuracy (affects PSNR)
- **SSIM Loss:** Optimizes structural similarity (directly targets competition metric)
- **Combined:** Balances both PSNR and SSIM improvement

### Alternative Losses to Experiment With
| Loss | Pros | Cons |
|------|------|------|
| **MSE (L2)** | Standard, targets PSNR | Can produce blurry results |
| **Charbonnier** | Smooth L1, good gradient properties | Slightly slower convergence |
| **Perceptual (VGG)** | Better visual quality | May not optimize PSNR directly |
| **FFT Loss** | Preserves frequency content | Complex to implement |

---

## 4.4 Optimizer & Scheduler Strategy

### Learning Rate Schedule

```
LR
 │
1e-3 ─┐
      │\
      │ \
      │  \
      │   \───────────────────────────────
      │    \         Cosine Annealing
1e-6 ─┤    \────────────────────────────── 
      └──────┴──────┴──────┴──────┴──────── Epochs
      0      50     100    150    200
      
      Phase 1        Phase 2   Phase 3
      (LR: 1e-3)    (1e-4)    (5e-5)
```

### Warmup Strategy (Optional)
```python
# Linear warmup for first 5 epochs
warmup_epochs = 5
warmup_lr = 1e-5  # Start from 1e-5, ramp to 1e-3
```

---

## 4.5 Training Monitoring

### Metrics to Track (Every Epoch)

| Metric | Where | Purpose |
|--------|-------|---------|
| Training L1 loss | Train set | Convergence monitoring |
| Validation L1 loss | Val set | Overfitting detection |
| Validation PSNR (dB) | Val set | Primary quality metric |
| Validation SSIM | Val set | Structural quality metric |
| ΔPSNR (vs noisy) | Val set | Competition metric proxy |
| ΔSSIM (vs noisy) | Val set | Competition metric proxy |
| **Composite Score** | Val set | **Direct competition metric** |
| Learning rate | - | Schedule verification |

### Early Stopping
- Monitor validation composite score
- Patience: 20 epochs
- Save best model based on composite score (not loss)

### Checkpointing
```python
# Save checkpoints at:
# - Every 10 epochs (rolling, keep last 3)
# - Best validation composite score (keep forever)
# - Final epoch (keep forever)
```

---

## 4.6 GPU Time Budget

### Estimated Training Times (T4 GPU)

| Phase | Epochs | Time/Epoch | Total |
|-------|--------|------------|-------|
| Phase 1 (256×256 patches) | 100 | ~5 min | ~8.3 hours |
| Phase 2 (384×384 patches) | 50 | ~8 min | ~6.7 hours |
| Phase 3 (512×512 patches) | 50 | ~12 min | ~10 hours |
| **Total** | **200** | - | **~25 hours** |

### Kaggle Budget Plan
- Week 1: Phase 1 training (~9h) + experiments (~6h) = 15h
- Week 2: Phase 2 + 3 (~17h) + final eval (~3h) = 20h
- **Total: ~35h** (fits within 2 weeks of Kaggle's 30h/week)

### Colab Backup Plan
- If Kaggle hours run out, use Colab for remaining training
- Split training across sessions with checkpoint saving

---

## 4.7 Reproducibility

```python
# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 4.8 Training Notebook Template

Save as a Colab/Kaggle notebook:

```python
# 1. Setup
!pip install torch torchvision kornia
# Mount Drive / upload dataset

# 2. Data
train_dataset = DenoisingDataset(...)
val_dataset = DenoisingDataset(...)

# 3. Model
model = NAFNet(config).cuda()

# 4. Training loop
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer, loss_fn)
    if epoch % eval_every == 0:
        metrics = evaluate(model, val_loader)
        save_if_best(model, metrics)
    scheduler.step()

# 5. Export best model
torch.save(best_model.state_dict(), 'best_model.pth')
# Upload to Google Drive
```
