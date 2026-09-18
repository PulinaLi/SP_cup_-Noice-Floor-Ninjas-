# 05 — Classical Signal Processing Integration

> **Worth 10% of preliminary marks** — "Use of significant classical signal-processing techniques beyond the baseline, with proper justification."

## 5.1 Why Classical SP Matters

The competition explicitly allocates **10% of the preliminary score** for classical SP techniques that go beyond the provided NLM baseline. This section details specific techniques to implement and justify.

---

## 5.2 Classical Techniques to Implement

### Technique 1: Wavelet-Domain Denoising (Pre-processing)

**What:** Apply wavelet decomposition, threshold high-frequency coefficients, then reconstruct.

**Why:** Wavelets naturally separate noise (mostly in high-frequency subbands) from signal. This is a well-established SP technique distinct from the spatial-domain NLM in the baseline.

```python
import pywt
import numpy as np

def wavelet_denoise(image, wavelet='db4', level=3, sigma=None):
    """
    Wavelet-based denoising using BayesShrink or VisuShrink.
    """
    denoised_channels = []
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        
        # Estimate noise sigma from finest detail coefficients
        if sigma is None:
            detail_coeffs = coeffs[-1]
            sigma_est = np.median(np.abs(detail_coeffs[0])) / 0.6745
        else:
            sigma_est = sigma
        
        # BayesShrink threshold
        thresholded = [coeffs[0]]  # Keep approximation
        for detail_level in coeffs[1:]:
            new_detail = []
            for d in detail_level:
                # Adaptive threshold
                sigma_d = np.std(d)
                if sigma_d > sigma_est:
                    thresh = sigma_est**2 / np.sqrt(max(sigma_d**2 - sigma_est**2, 1e-10))
                else:
                    thresh = np.max(np.abs(d))
                new_detail.append(pywt.threshold(d, thresh, mode='soft'))
            thresholded.append(tuple(new_detail))
        
        denoised_channels.append(pywt.waverec2(thresholded, wavelet))
    
    return np.stack(denoised_channels, axis=-1)
```

**Justification for report:**
> Wavelet denoising leverages the sparsity of natural images in the wavelet domain. By applying soft thresholding to detail coefficients, we suppress noise while preserving edges. BayesShrink provides an adaptive, data-driven threshold estimate, making it more robust than fixed-threshold approaches.

---

### Technique 2: Bilateral Filtering (Pre/Post-processing)

**What:** Edge-preserving smoothing using both spatial and intensity proximity.

**Why:** Unlike Gaussian blur, bilateral filtering preserves edges while smoothing flat regions — ideal for noise removal in structured scenes.

```python
import cv2

def bilateral_denoise(image_uint8, d=9, sigma_color=75, sigma_space=75):
    """
    Edge-preserving bilateral filter.
    d: diameter of pixel neighborhood
    sigma_color: filter sigma in color space
    sigma_space: filter sigma in coordinate space
    """
    return cv2.bilateralFilter(image_uint8, d, sigma_color, sigma_space)
```

**Justification:**
> Bilateral filtering is a non-linear, edge-preserving filter that weights neighboring pixels based on both spatial distance and intensity difference. This dual-weighting mechanism smooths noise in homogeneous regions while preserving sharp edges, complementing the NLM approach.

---

### Technique 3: Frequency-Domain Filtering (Analysis + Optional Pre-processing)

**What:** Analyze and filter noise in the frequency domain using DFT.

**Why:** Some noise patterns (e.g., periodic noise, banding) are clearly visible in the frequency domain.

```python
def frequency_analysis_and_filter(image, cutoff_ratio=0.8):
    """
    Analyze noise in frequency domain and optionally apply low-pass filtering.
    """
    filtered_channels = []
    for c in range(image.shape[2]):
        # DFT
        f_transform = np.fft.fft2(image[:, :, c])
        f_shift = np.fft.fftshift(f_transform)
        
        # Power spectrum analysis
        magnitude = np.abs(f_shift)
        
        # Butterworth low-pass filter (smooth rolloff)
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2
        D0 = cutoff_ratio * min(rows, cols) / 2
        
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols).reshape(1, -1) - ccol
        D = np.sqrt(u**2 + v**2)
        n_order = 2
        H = 1 / (1 + (D / D0)**(2 * n_order))
        
        # Apply filter
        filtered = f_shift * H
        f_ishift = np.fft.ifftshift(filtered)
        result = np.real(np.fft.ifft2(f_ishift))
        filtered_channels.append(result)
    
    return np.stack(filtered_channels, axis=-1)
```

**Justification:**
> Frequency-domain analysis reveals the spectral characteristics of the noise model. By examining the power spectral density of noise residuals, we can identify whether the noise is white (flat spectrum) or colored (structured spectrum), informing our choice of denoising approach.

---

### Technique 4: Wiener Filtering (Optional)

**What:** Optimal linear filter that minimizes MSE given known noise and signal statistics.

```python
from scipy.signal import wiener

def wiener_denoise(image, noise_variance=None):
    """Wiener filter - optimal MMSE linear estimator."""
    denoised = np.zeros_like(image)
    for c in range(image.shape[2]):
        denoised[:, :, c] = wiener(image[:, :, c], noise=noise_variance)
    return denoised
```

**Justification:**
> The Wiener filter is the optimal linear estimator in the minimum mean-square-error sense, given knowledge of the signal and noise power spectra. While it assumes stationarity, it provides a principled baseline for noise reduction.

---

### Technique 5: Improved Defect Pixel Correction

**What:** Enhanced version of the baseline's defect pixel correction with adaptive thresholding.

```python
def adaptive_defect_correction(image, base_threshold=0.2, 
                                kernel_sizes=[3, 5, 7]):
    """
    Multi-scale adaptive defect pixel correction.
    Uses multiple kernel sizes and adaptive thresholds.
    """
    corrected = image.copy()
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        
        # Multi-scale median
        median_maps = []
        for k in kernel_sizes:
            ch_u8 = (channel * 255).astype(np.uint8)
            median_u8 = cv2.medianBlur(ch_u8, k)
            median_maps.append(median_u8.astype(np.float32) / 255.0)
        
        # Adaptive threshold based on local variance
        local_std = cv2.GaussianBlur(
            (channel - median_maps[0])**2, (15, 15), 0
        ) ** 0.5
        adaptive_thresh = base_threshold + 2 * local_std
        
        # Apply correction using smallest kernel that detects the defect
        for median_map in median_maps:
            deviation = np.abs(channel - median_map)
            outlier = deviation > adaptive_thresh
            corrected[:, :, c] = np.where(
                outlier, median_map, corrected[:, :, c]
            )
    
    return corrected
```

---

## 5.3 Integration into the Pipeline

### Pre-processing Chain (Before Deep Learning Model)
```
Input → Adaptive Defect Correction → Wavelet Denoise (light) → DL Model
```

### Post-processing Chain (After Deep Learning Model)
```
DL Model Output → Bilateral Filter (light, edge preservation) → Output
```

### Key Principle
Classical techniques should **complement** the deep learning model, not compete with it:
- **Pre-processing:** Remove obvious artifacts (defect pixels, extreme noise) to help the DL model
- **Post-processing:** Refine edges and suppress residual noise the DL model missed
- **Keep classical processing light** — don't over-smooth before the DL model sees the image

---

## 5.4 What to Include in the Report

For the 10% classical SP marks, the report should:
1. ✅ **Name each classical technique** used
2. ✅ **Explain the mathematical foundation** (e.g., wavelet thresholding theory)
3. ✅ **Justify why it helps** for this specific noise model
4. ✅ **Show before/after results** with PSNR/SSIM improvements
5. ✅ **Compare classical-only vs. hybrid** approach scores
6. ✅ **Include diagrams** of the processing pipeline
