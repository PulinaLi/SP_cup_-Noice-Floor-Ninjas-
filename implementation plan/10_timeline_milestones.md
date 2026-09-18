# 10 — Timeline & Milestones

## Competition Window

```
Start:    August 29, 2026
Deadline: September 19, 2026 (assumed 11:59 PM)
Duration: ~21 days
Today:    September 18, 2026 ← ONLY 1 DAY LEFT!
```

> ⚠️ **CRITICAL:** With only ~1 day remaining, this timeline is compressed to an emergency plan.

---

## 🚨 EMERGENCY 1-DAY PLAN (September 18–19)

### Priority: Get a working submission that beats the baseline (0.263)

---

### BLOCK 1: Hours 0–3 (Immediate — Setup & Quick Wins)

| Time | Task | Priority |
|------|------|----------|
| 0:00 | Set up the working environment, install dependencies | 🔴 Critical |
| 0:30 | Run baseline on public set, verify score ≈ 0.263 | 🔴 Critical |
| 1:00 | Analyze noise: compute noise residuals for 10–20 sample images | 🟡 High |
| 1:30 | Tune baseline NLM parameters (try h=12, 15, 20) | 🔴 Critical |
| 2:00 | Add wavelet denoising pre-processing to baseline | 🟡 High |
| 2:30 | Add bilateral filtering post-processing | 🟡 High |
| 3:00 | Evaluate improved classical pipeline → target score > 0.30 | 🔴 Critical |

**Deliverable:** Improved classical pipeline beating baseline

---

### BLOCK 2: Hours 3–8 (Deep Learning — If GPU Available)

| Time | Task | Priority |
|------|------|----------|
| 3:00 | Start Colab/Kaggle notebook | 🔴 Critical |
| 3:30 | Implement simple UNet or load pre-trained DnCNN | 🔴 Critical |
| 4:00 | Set up data pipeline (patches, augmentation) | 🔴 Critical |
| 4:30 | Begin training on 420 images (val: 40) | 🔴 Critical |
| 5:00–7:00 | Monitor training, adjust LR if needed | 🟡 High |
| 7:00 | Evaluate best checkpoint on validation set | 🔴 Critical |
| 7:30 | If DL score > classical score → use DL model | Decision |
| 8:00 | Generate denoised outputs for images 461–480 | 🔴 Critical |

**Deliverable:** Trained model (even if lightly trained)

---

### BLOCK 3: Hours 8–12 (Assembly & Submission)

| Time | Task | Priority |
|------|------|----------|
| 8:00 | Finalize `scripts/denoise.py` with best approach | 🔴 Critical |
| 8:30 | Verify denoise.py produces correct outputs on CPU | 🔴 Critical |
| 9:00 | Run submission verification (20 files, correct format) | 🔴 Critical |
| 9:30 | Create TeamName.zip with 461.png–480.png | 🔴 Critical |
| 10:00 | Write report (use the outline from 09_report_outline.md) | 🔴 Critical |
| 11:00 | Upload model to Google Drive, record SHA-256 | 🟡 High |
| 11:30 | Push code to GitHub, record commit SHA | 🔴 Critical |
| 12:00 | Submit Google Drive folder | 🔴 Critical |

**Deliverable:** Complete submission

---

## FALLBACK PLAN: Classical-Only (If No Time for DL)

If deep learning training isn't possible in time:

```python
# Enhanced classical pipeline
def denoise(image):
    # 1. Adaptive defect pixel correction (improved over baseline)
    image = adaptive_defect_correction(image)
    
    # 2. Wavelet denoising (BayesShrink)
    image = wavelet_denoise(image, wavelet='db4', level=3)
    
    # 3. NLM with tuned parameters
    image = nlm_denoise(image, h=12.0)
    
    # 4. Light bilateral filter for edge preservation
    image = bilateral_filter(image, d=5, sigma_color=30, sigma_space=30)
    
    return image
```

Expected improvement over baseline:
- Baseline: 0.263
- Tuned NLM: ~0.28–0.30
- + Wavelet: ~0.30–0.33
- + Better defect correction: ~0.32–0.35

---

## IDEAL TIMELINE (If Starting Fresh with 21 Days)

For reference / future competitions:

### Week 1: Research & Foundation (Days 1–7)

| Day | Task |
|-----|------|
| 1 | Read handbook, set up repo, run baseline |
| 2 | Noise analysis (statistics, frequency, spatial patterns) |
| 3 | Research model architectures (NAFNet, Restormer, UNet) |
| 4 | Implement data pipeline (dataset, augmentation, splits) |
| 5 | Implement NAFNet architecture |
| 6 | First training run (small config, verify pipeline) |
| 7 | Review Week 1, plan adjustments |

### Week 2: Training & Optimization (Days 8–14)

| Day | Task |
|-----|------|
| 8 | Phase 1 training (256×256 patches, 100 epochs) |
| 9 | Continue Phase 1 training |
| 10 | Evaluate Phase 1, begin Phase 2 (384×384) |
| 11 | Phase 2 training + classical SP integration |
| 12 | Phase 3 (512×512) + hyperparameter tuning |
| 13 | Ensemble experiments (self-ensemble, model ensemble) |
| 14 | Finalize best model, run full evaluation |

### Week 3: Submission & Polish (Days 15–21)

| Day | Task |
|-----|------|
| 15 | Generate final denoised outputs (461–480) |
| 16 | Write report (draft) |
| 17 | Polish report, create diagrams |
| 18 | Finalize scripts/denoise.py, test on CPU |
| 19 | Upload model, push code, create ZIP |
| 20 | Final review and submission |
| 21 | Buffer day (fix any issues) |

---

## Key Milestones Summary

| Milestone | Target | Status |
|-----------|--------|--------|
| Baseline reproduced (0.263) | ASAP | ☐ |
| Classical improvement (>0.30) | +3h | ☐ |
| DL model training started | +4h | ☐ |
| DL model beating baseline | +8h | ☐ |
| denoise.py finalized | +9h | ☐ |
| Submission images generated | +9.5h | ☐ |
| Report written | +11h | ☐ |
| Everything submitted | +12h | ☐ |
