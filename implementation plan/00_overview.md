# 🏆 Mora SP Cup 2026 — Full Implementation Plan

## Low-Light Image Denoising & Enhancement Challenge

> **Competition Period:** August 29 – September 19, 2026  
> **Preliminary Deadline:** September 19, 2026  
> **Final Round:** TBA (Physical — University of Moratuwa)

---

## 📋 Plan Documents Index

| # | Document | Description |
|---|----------|-------------|
| 00 | [Overview](./00_overview.md) | This file — master roadmap and quick reference |
| 01 | [Problem Analysis](./01_problem_analysis.md) | Dataset details, noise characteristics, baseline review |
| 02 | [Solution Architecture](./02_solution_architecture.md) | Proposed denoising approach (classical + deep learning) |
| 03 | [Data Pipeline](./03_data_pipeline.md) | Data loading, augmentation, train/val split strategy |
| 04 | [Model Training Plan](./04_model_training_plan.md) | Training recipe, hyperparameters, GPU resource plan |
| 05 | [Classical SP Integration](./05_classical_sp_integration.md) | Classical signal-processing techniques for bonus marks |
| 06 | [Evaluation Strategy](./06_evaluation_strategy.md) | Metrics, self-evaluation, validation methodology |
| 07 | [Codebase Structure](./07_codebase_structure.md) | Repository layout, `scripts/denoise.py` design |
| 08 | [Submission Checklist](./08_submission_checklist.md) | Complete deliverables checklist & compliance rules |
| 09 | [Report Outline](./09_report_outline.md) | Report structure (max 5 pages, 12pt, 1-inch margins) |
| 10 | [Timeline & Milestones](./10_timeline_milestones.md) | Day-by-day execution schedule |

---

## 🎯 Scoring Breakdown (Preliminary Round)

| Component | Weight | What to Optimize |
|-----------|--------|------------------|
| Report & technical analysis | **40%** | Architecture explanation, justification, clear flow |
| Classical SP techniques (beyond baseline) | **10%** | Wavelet denoising, bilateral filtering, frequency domain |
| Denoising score on 20 hidden images (461–480) | **35%** | Composite Score = 0.6 × N(ΔPSNR) + 0.4 × max(ΔSSIM, 0) |
| Code quality, guidelines adherence, runtime | **15%** | Clean code, proper structure, fast CPU inference |

### Composite Score Formula
```
ΔPSNR = PSNR_denoised − PSNR_noisy
ΔSSIM = SSIM_denoised − SSIM_noisy
N     = clip(ΔPSNR / 15, 0, 1)
S     = max(ΔSSIM, 0)

Composite Score = 0.6 × N + 0.4 × S
```

> **Baseline composite score ≈ 0.26** — submissions must exceed this to be ranked.

---

## 🚀 High-Level Strategy

### Phase 1: Analysis & Baseline (Days 1–2)
- Analyze noise types and severity levels in the dataset
- Run the provided NLM baseline and record scores
- Set up training/validation splits

### Phase 2: Model Development (Days 3–12)
- Implement a deep-learning denoiser (e.g., NAFNet / Restormer / UNet variant)
- Integrate classical SP pre/post-processing for bonus marks
- Train on Colab/Kaggle (free GPU)

### Phase 3: Optimization & Validation (Days 13–17)
- Hyperparameter tuning and ensemble strategies
- Validate on held-out sets
- Ensure CPU fallback works

### Phase 4: Submission (Days 18–21)
- Generate final denoised images (461–480)
- Write the report (max 5 pages)
- Package ZIP + PDF, freeze code & model

---

## ⚠️ Critical Rules to Remember

1. **All custom code goes in `scripts/` only** — do NOT modify other directories
2. **Output naming:** `461.png` – `480.png` (no `_noise` suffix)
3. **Code must run offline** — no network requests during inference
4. **CPU fallback required** — test on CPU before submission
5. **Submission freeze** — no commits after deadline, no model changes
6. **No hardcoding** — image-specific mappings = immediate disqualification
7. **Report:** max 5 pages, 12pt font, 1-inch margins
