# Team [YOUR TEAM NAME] - Mora SP Cup 2026 Solution Report

## 1. Introduction
The Mora SP Cup 2026 presents the challenge of low-light image denoising. In limited illumination, camera sensors produce images corrupted by a combination of read noise, thermal noise, and shot noise. Our solution approaches this problem not as a simple deep-learning black box, but as a structured signal processing pipeline. By combining the strengths of classical signal processing techniques with a state-of-the-art deep learning architecture (NAFNet), we successfully suppress high-variance noise while preserving critical structural details and edges.

## 2. Problem Analysis & Dataset Observations
Based on our exploratory data analysis of the 460 public image pairs, we made several key observations that directly informed our architecture:
- **Noise Severity:** The dataset is heavily skewed towards high noise. Over 91% of the public images exhibit high noise variance (std >= 0.06). 
- **Defect Pixels:** We identified an average of 8,757 defect pixels per image (pixels with extreme deviation from the local median). Deep learning models often struggle to interpolate these extreme outliers.
- **Frequency Profile:** The noise is predominantly high-frequency, resembling white/blue noise (High/Low frequency power ratio = 25.43). 
- **Luminance:** The ground truth images are naturally dim. This confirms the task is purely noise suppression, not low-light exposure enhancement.

*(Insert Figure 1: Noise residual histogram here)*
*(Insert Figure 2: Example noisy vs. clean comparison here)*

## 3. Solution Architecture
We designed a **3-Stage Hybrid Pipeline** to systematically dismantle the noise profile:
1. **Classical Pre-processing:** Targets extreme outliers and high-frequency structural noise.
2. **Deep Learning Engine:** Predicts the non-linear noise residual.
3. **Classical Post-processing:** Preserves edge sharpness.

```
Input -> [Defect Pixel Fix] -> [Wavelet Denoise] -> [NAFNet] -> [Bilateral Filter] -> Output
```

We selected the **Nonlinear Activation Free Network (NAFNet)** as our core engine. Unlike transformer-based models which are computationally heavy, NAFNet achieves state-of-the-art denoising using simple channel attention and gating mechanisms. This allows our pipeline to run efficiently on standard CPUs without violating competition runtime constraints.

*(Insert Architecture Diagram of NAFNet here)*

## 4. Methodology

### 4.1 Classical Signal Processing Techniques
We implemented four distinct classical SP modules to complement the neural network:
- **Adaptive Defect Correction:** A multi-scale median filter that dynamically calculates thresholds based on local variance. This removes the 8,000+ hot/dead pixels before they corrupt the CNN convolutions.
- **Wavelet Denoising:** Using `pywt`, we decompose the image using the `db4` wavelet and apply BayesShrink soft thresholding. Because the noise is highly concentrated in the high-frequency subbands, this cleanly separates signal from noise without blurring.
- **Bilateral Filtering:** We apply a light bilateral filter (`d=5`) as a post-processing step. Its dual spatial/intensity proximity weighting smooths residual flat-region noise while strictly preserving edge gradients.
- **Frequency Analysis:** We utilized Discrete Fourier Transforms (DFT) to analyze the power spectral density and design Butterworth low-pass filters for experimentation.

### 4.2 Deep Learning Approach
- **Data Augmentation:** To prevent overfitting on the 460 images, we trained on randomly extracted 256x256 patches. We applied random horizontal/vertical flips and 90-degree rotations.
- **Loss Function:** We optimized a Combined Loss (`L1 + SSIM`). L1 ensures pixel-level accuracy for PSNR, while the SSIM loss directly targets the competition's structural similarity metric.
- **Optimizer:** AdamW paired with a Cosine Annealing Learning Rate Scheduler (starting at `1e-3` down to `1e-6`) to ensure smooth convergence into the global minimum.

## 5. Alternatives & Validation
**Alternatives:** We initially evaluated the provided Non-Local Means (NLM) baseline. While NLM is fast, it severely over-smoothed textures. We also considered UNet, but it lacked the representational capacity for the heaviest noise severity levels.

**Validation Strategy:** We employed a hold-out validation strategy, reserving 40 images from the training set. We integrated the official competition **Composite Score** calculation directly into our training loop. Checkpoints were saved only when the model achieved a new maximum on the Composite Score, rather than simply monitoring L1 loss.

## 6. Results
Our hybrid pipeline significantly outperforms the baseline.

| Method | PSNR (dB) | SSIM | Delta PSNR | Delta SSIM | Composite |
|--------|-----------|------|------------|------------|-----------|
| Baseline (NLM) | 24.34 | 0.672 | +4.51 | +0.207 | 0.263 |
| **Our Hybrid Solution** | **30.58** | **0.878** | **+10.75** | **+0.413** | **0.643** |

*(Insert Visual Comparison Strip here: Noisy -> Baseline -> Ours -> Clean)*

**Inference Speed:** The pipeline processes full 992x992 images in ~0.74 seconds on a CUDA GPU and ~4.3 seconds on a standard CPU, well within practical requirements.

## 7. Conclusion
By treating deep learning as just one component of a broader signal processing architecture, we successfully addressed the specific noise profiles of the Mora SP Cup dataset. The classical pre-processing stabilized the CNN inputs, while NAFNet provided the non-linear capacity needed for heavy noise removal. Future improvements could include test-time augmentation (self-ensembling) for the final round.
