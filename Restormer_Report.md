# Mora SP Cup 2026 - Comprehensive Project Report

**Team Name:** Noise Floor Ninjas  
**Date:** September 19, 2026

---

## 1. Introduction
This document serves as a comprehensive log of the entire developmental journey, architectural iterations, and experimental methodologies our team employed for the Mora SP Cup 2026 Image Denoising challenge. Our goal was to push the boundaries of extreme low-light synthetic noise restoration while strictly adhering to VRAM and computational constraints. 

Over the course of the project, we experimented with multiple state-of-the-art architectures, from Convolutional Neural Networks (CNNs) to Vision Transformers, and implemented ensemble strategies to maximize our final composite score (Delta PSNR + Delta SSIM).

---

## 2. The Core Problem & Domain Shift Fix
Early in our process, we discovered a critical issue: **Domain Shift**. 
Initially, we attempted to apply classical denoising filters (like Bilateral and Non-Local Means) *before* feeding the images into our deep learning models. This severely confused the neural networks, as they were trained on pure, mathematically generated synthetic noise, but were being fed images that were already blurred and artifact-laden by the classical filters. 

**The Solution:** We inverted the pipeline. We fed the raw, noisy `float32` tensors directly into our deep learning models first. Then, we applied a cascade of classical filters as a *post-processing* polish. This instantly unlocked our ability to score above 0.60.

---

## 3. Architectural Iterations & Experiments

We aggressively iterated through several AI architectures to find the optimal balance between edge preservation and noise suppression.

### Iteration 1: NAFNet (Nonlinear Activation Free Network)
Our first major success was implementing NAFNet. By removing complex non-linear activation functions and relying on SimpleGate blocks, NAFNet proved to be incredibly computationally efficient. 
- **Validation Score:** 0.6434 Composite.
- **Outcome:** A massive success. This formed the baseline for our `TeamName.zip` submission.

### Iteration 2: Attention U-Net
To try and beat the NAFNet, we built a custom Attention U-Net. This architecture utilized attention gates in the skip connections to help the network focus specifically on structural edges rather than background variance. 
- **Validation Score:** 0.577 Composite.
- **Outcome:** While it sharpened edges beautifully, it struggled with the heavy noise floor variance compared to NAFNet. 

### Iteration 3: The Ensemble Strategy
To leverage the strengths of both models, we built an inference script that ran noisy images through *both* the NAFNet and the Attention U-Net simultaneously. We then mathematically fused their outputs (weighting NAFNet at 60% and U-Net at 40%).
- **Outcome:** A robust output that smoothed the heavy noise while sharpening the edges, proving the viability of model ensembling. 

### Final Architecture: Lightweight Restormer (State-of-the-Art)
For our final, ultimate push, we discarded CNNs entirely and implemented a Vision Transformer: the **Restormer**. 
Because Transformers require massive amounts of VRAM, we custom-engineered a lightweight Restormer specifically tailored to train on a 6GB RTX 4050 GPU by optimizing channel widths and blocks.
- **MDTA (Multi-Dconv Head Transposed Attention):** Captures global image dependencies across feature channels rather than spatial dimensions.
- **GDFN (Gated-Dconv Feed-Forward Network):** Enhances informative features while suppressing noisy activations.
- **Validation Score:** **0.6467 Composite.**
- **Outcome:** Our highest score yet. The Restormer successfully surpassed all previous models and ensembles.

---

## 4. The Classical Post-Processing Cascade

Regardless of which deep learning engine we used, every image went through our custom post-processing filter cascade to eliminate any remaining statistical anomalies:

1. **Adaptive Defect Correction:** Eliminates isolated "hot" or "dead" pixels caused by the sensor limits.
2. **Wavelet Denoising (db4, level 1):** Soft-thresholding applied to high-frequency subbands to remove granular high-frequency noise.
3. **Bilateral Filtering (d=5, sigma=25):** A final edge-preserving smoothing pass to completely flatten the variance in uniform dark areas, satisfying the aesthetic requirement for a "clean" image.

---

## 5. Final Submissions & Checksums

We have prepared two highly competitive submissions based on this developmental journey. 

**Submission 1: The Original Baseline (TeamName.zip)**
- **Architecture:** NAFNet + Post-Processing
- **Composite Score:** 0.6434
- **Model File:** `best_model.pth`

**Submission 2: The Final Push (Restormer_Submission.zip)**
- **Architecture:** Lightweight Restormer + Post-Processing
- **Composite Score:** 0.6467 (Highest)
- **Model File:** `restormer_best.pth`
- **Model Checkpoint SHA-256:** `318288c2f370806da3fd5762f4b4adcae7c3e8f332e60b18dfb6e719f2e652ac`

**Codebase Checksum:**
- **Git Commit Hash:** `16d28b1f3264aa90d69a9a33f37fa9c70a7f97e0`
