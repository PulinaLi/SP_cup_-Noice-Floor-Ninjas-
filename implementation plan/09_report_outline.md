# 09 — Report Outline

## Report Constraints

| Constraint | Requirement |
|------------|-------------|
| **Max pages** | 5 pages |
| **Font size** | 12-point |
| **Margins** | 1-inch on all sides |
| **Filename** | `TeamName_Report.pdf` |
| **Cover page** | Not required |
| **Style** | Clear flow, proper justification, to the point |

> ⚠️ Failure to follow these formatting requirements may lead to a penalty or disqualification.

---

## Proposed Report Structure

### Page 1: Introduction & Problem Analysis (~1 page)

#### 1. Introduction (0.3 pages)
- Brief overview of the low-light denoising challenge
- Significance of the problem in real-world photography
- High-level summary of our approach

#### 2. Problem Analysis & Dataset Observations (0.7 pages)
- **Noise characteristics:** Types of noise observed (read noise, thermal, shot noise, defect pixels)
- **Severity levels:** Distribution of noise intensity across the dataset
- **Noise statistics:** Mean, variance, signal-dependency analysis
- **Frequency-domain observations:** Power spectral density of noise
- Include 1–2 key figures:
  - Noise residual histogram
  - Example noisy vs. clean comparison

---

### Page 2: Solution Architecture (~1 page)

#### 3. Solution Architecture (1 page)
- **Pipeline diagram** showing the full processing chain:
  ```
  Input → Classical Pre-processing → Deep Learning Model → Post-processing → Output
  ```
- **Architecture diagram** of the deep learning model (NAFNet/chosen model)
- **Key design choices:**
  - Why hybrid approach (classical + DL)
  - Why the chosen model architecture
  - How the pipeline components interact
- Include architecture diagram (block diagram preferred)

---

### Page 3: Methodology (~1 page)

#### 4. Classical Signal Processing Techniques (0.4 pages)
- **Wavelet denoising:** BayesShrink, multi-level decomposition
- **Bilateral filtering:** Edge-preserving smoothing
- **Defect pixel correction:** Adaptive multi-scale approach
- **Frequency analysis:** DFT-based noise characterization
- Mathematical formulations (brief)
- Why each technique was chosen for this noise model

#### 5. Deep Learning Approach (0.6 pages)
- Model architecture details (layers, channels, attention)
- Training strategy:
  - Patch-based training (256×256 → 384×384 → 512×512)
  - Data augmentation methods
  - Loss function (L1 + SSIM)
  - Optimizer & scheduler configuration
  - Training/validation split strategy

---

### Page 4: Alternatives & Validation (~1 page)

#### 6. Alternatives Considered (0.4 pages)
- **Models evaluated:** NAFNet vs. UNet vs. DnCNN vs. Restormer
- Why the chosen model outperformed alternatives
- Brief comparison table with scores on validation set
- What didn't work and why

#### 7. Validation Approach (0.6 pages)
- Train/validation split strategy (how many images, how stratified)
- Evaluation metrics used (PSNR, SSIM, ΔPSNR, ΔSSIM, Composite)
- Self-evaluation using `evaluation/evaluate.py`
- Overfitting prevention measures
- Per-noise-level analysis of results

---

### Page 5: Results (~1 page)

#### 8. Results (0.7 pages)
- **Quantitative results table:**

| Method | PSNR (dB) | SSIM | ΔPSNR | ΔSSIM | Composite |
|--------|-----------|------|-------|-------|-----------|
| Noisy input | — | — | — | — | — |
| Baseline (NLM) | 24.34 | 0.672 | +4.51 | +0.207 | 0.263 |
| Classical only | — | — | — | — | — |
| DL only | — | — | — | — | — |
| **Our solution** | **—** | **—** | **—** | **—** | **—** |

- Visual comparison: noisy → baseline → ours → ground truth
- Per-severity-level performance analysis
- Runtime performance (ms/image on CPU)

#### 9. Conclusion (0.3 pages)
- Summary of key contributions
- Strengths and limitations of the approach
- Potential improvements for the final round

---

## Writing Tips

### Do's ✅
- Use concise, technical language
- Include quantitative results for every claim
- Show before/after visual examples
- Use proper figure captions and table labels
- Reference the evaluation formula
- Include architecture diagrams

### Don'ts ❌
- Don't pad with unnecessary background (the judges know image processing)
- Don't include a cover page (waste of page budget)
- Don't use vague language ("our method performs well")
- Don't forget to mention classical SP techniques (10% of marks!)
- Don't exceed 5 pages (may result in penalty)

---

## Figure Budget

With only 5 pages, be strategic about figures:

| Figure | Est. Space | Purpose |
|--------|-----------|---------|
| Pipeline diagram | 0.25 page | Show overall approach |
| Model architecture | 0.25 page | Show network design |
| Noise analysis (1–2 plots) | 0.25 page | Support problem analysis |
| Visual comparison strip | 0.3 page | Show denoising quality |
| Results table | 0.15 page | Quantitative summary |
| **Total figures** | **~1.2 pages** | |
| **Remaining for text** | **~3.8 pages** | |
