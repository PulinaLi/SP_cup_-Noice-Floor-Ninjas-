"""
denoise.py — Enhanced Low-Light Image Denoiser
Mora SP Cup 2026

Hybrid pipeline combining classical signal-processing techniques with
optimised parameter tuning for low-light image denoising.

Pipeline:
    1. Adaptive multi-scale defect-pixel correction
    2. Wavelet-domain denoising (BayesShrink)
    3. Tuned Non-Local Means denoising
    4. Edge-preserving bilateral post-filtering

USAGE:
    python scripts/denoise.py --noise_dir competition_data/submissions/noisy \
                              --denoised_dir competition_data/submissions/denoised
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pywt

VALID_EXT = {".png", ".jpg", ".jpeg"}
NOISE_SUFFIX = "_noise"


# ---------------------------------------------------------------------------
# 1. Adaptive Multi-Scale Defect Pixel Correction
# ---------------------------------------------------------------------------
def adaptive_defect_correction(img_float, base_threshold=0.20,
                               kernel_sizes=(3, 5)):
    """
    Improved defect-pixel (hot/dead pixel) correction using multi-scale
    median filtering with adaptive thresholds.

    For each channel, compute the local median at multiple scales and
    replace pixels whose deviation from the median exceeds an adaptive
    threshold derived from local variance.
    """
    corrected = img_float.copy()
    for c in range(img_float.shape[2]):
        channel = img_float[..., c]

        # Compute median at smallest scale for reference
        ch_u8 = (channel * 255).astype(np.uint8)
        median_u8 = cv2.medianBlur(ch_u8, kernel_sizes[0])
        median_ref = median_u8.astype(np.float32) / 255.0

        # Local standard deviation (for adaptive threshold)
        diff_sq = (channel - median_ref) ** 2
        local_var = cv2.GaussianBlur(diff_sq, (15, 15), 0)
        local_std = np.sqrt(local_var + 1e-10)
        adaptive_thresh = base_threshold + 2.0 * local_std

        # Apply correction with each kernel size
        for k in kernel_sizes:
            k_u8 = cv2.medianBlur(ch_u8, k)
            k_float = k_u8.astype(np.float32) / 255.0
            deviation = np.abs(channel - k_float)
            outlier_mask = deviation > adaptive_thresh
            corrected[..., c] = np.where(outlier_mask, k_float,
                                         corrected[..., c])

    return corrected


# ---------------------------------------------------------------------------
# 2. Wavelet-Domain Denoising (BayesShrink)
# ---------------------------------------------------------------------------
def wavelet_denoise(img_float, wavelet="db4", level=2):
    """
    Wavelet-based denoising using BayesShrink soft thresholding.

    Decomposes each colour channel into wavelet sub-bands, estimates the
    noise standard deviation from the finest detail coefficients using the
    Median Absolute Deviation (MAD) estimator, and applies BayesShrink
    adaptive thresholds to the detail coefficients.
    """
    denoised = np.zeros_like(img_float)

    for c in range(img_float.shape[2]):
        channel = img_float[..., c]
        coeffs = pywt.wavedec2(channel, wavelet, level=level)

        # Noise estimate from finest-scale diagonal detail
        detail_finest = coeffs[-1]  # tuple (cH, cV, cD) at finest level
        # Use the diagonal sub-band for MAD noise estimation
        sigma_noise = np.median(np.abs(detail_finest[2])) / 0.6745

        # Threshold each detail level (keep approximation untouched)
        thresholded = [coeffs[0]]
        for detail_level in coeffs[1:]:
            new_detail = []
            for d in detail_level:
                sigma_d = np.std(d)
                # BayesShrink threshold
                if sigma_d > sigma_noise and sigma_noise > 0:
                    sigma_signal_sq = max(sigma_d ** 2 - sigma_noise ** 2,
                                          0)
                    if sigma_signal_sq > 0:
                        thresh = sigma_noise ** 2 / np.sqrt(sigma_signal_sq)
                    else:
                        thresh = np.max(np.abs(d))
                else:
                    thresh = np.max(np.abs(d))
                new_detail.append(pywt.threshold(d, thresh, mode="soft"))
            thresholded.append(tuple(new_detail))

        reconstructed = pywt.waverec2(thresholded, wavelet)
        # Handle potential size mismatch from wavelet reconstruction
        h, w = channel.shape
        denoised[..., c] = reconstructed[:h, :w]

    return np.clip(denoised, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Non-Local Means Denoising (Tuned)
# ---------------------------------------------------------------------------
def denoise_nlm(img_uint8, h=12.0, h_color=12.0,
                template_window=7, search_window=21):
    """
    Non-Local Means denoising with tuned parameters.

    NLM exploits self-similarity in natural images by averaging patches
    that are similar across the image, weighted by patch distance.
    """
    return cv2.fastNlMeansDenoisingColored(
        img_uint8, None, h, h_color, template_window, search_window
    )


# ---------------------------------------------------------------------------
# 4. Bilateral Post-Filter (Edge-Preserving)
# ---------------------------------------------------------------------------
def bilateral_postfilter(img_uint8, d=5, sigma_color=30, sigma_space=30):
    """
    Light bilateral filter for post-processing edge preservation.

    The bilateral filter smooths noise while preserving edges by using
    both spatial and intensity-domain proximity weights.
    """
    return cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------
def process_image(img_float, nlm_h=12.0, use_wavelet=True,
                  use_bilateral=True, use_defect_correction=True):
    """
    Full denoising pipeline:
        Defect correction -> Wavelet -> NLM -> Bilateral
    """
    # Stage 1: Defect pixel correction
    if use_defect_correction:
        img_float = adaptive_defect_correction(img_float)

    # Stage 2: Wavelet denoising (light pass)
    if use_wavelet:
        img_float = wavelet_denoise(img_float, wavelet="db4", level=2)

    # Stage 3: Non-Local Means (main denoiser)
    img_uint8 = (img_float * 255).astype(np.uint8)
    img_uint8 = denoise_nlm(img_uint8, h=nlm_h, h_color=nlm_h)

    # Stage 4: Bilateral post-filter (edge preservation)
    if use_bilateral:
        img_uint8 = bilateral_postfilter(img_uint8, d=5,
                                         sigma_color=25, sigma_space=25)

    return img_uint8.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Filename Handling
# ---------------------------------------------------------------------------
def strip_noise_suffix(stem):
    """Remove '_noise' suffix from filename stem."""
    if stem.lower().endswith(NOISE_SUFFIX):
        return stem[: -len(NOISE_SUFFIX)]
    return stem


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--noise_dir", required=True, type=Path,
                    help="Directory containing noisy input images")
    ap.add_argument("--denoised_dir", required=True, type=Path,
                    help="Directory to save denoised output images")
    ap.add_argument("--nlm_h", type=float, default=12.0,
                    help="NLM denoising strength (default: 12.0)")
    ap.add_argument("--skip_wavelet", action="store_true",
                    help="Skip wavelet denoising stage")
    ap.add_argument("--skip_bilateral", action="store_true",
                    help="Skip bilateral post-filter stage")
    ap.add_argument("--skip_defect_correction", action="store_true",
                    help="Skip defect-pixel correction stage")
    args = ap.parse_args()

    args.denoised_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in args.noise_dir.glob("*") if p.suffix.lower() in VALID_EXT
    )
    if not paths:
        raise SystemExit(f"No images found in {args.noise_dir}")

    total_time = 0.0
    for i, path in enumerate(paths, 1):
        img = Image.open(path).convert("RGB")
        img_float = np.asarray(img).astype(np.float32) / 255.0

        start = time.time()
        result = process_image(
            img_float,
            nlm_h=args.nlm_h,
            use_wavelet=not args.skip_wavelet,
            use_bilateral=not args.skip_bilateral,
            use_defect_correction=not args.skip_defect_correction,
        )
        elapsed = time.time() - start
        total_time += elapsed

        out_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        out_id = strip_noise_suffix(path.stem)
        Image.fromarray(out_uint8).save(args.denoised_dir / f"{out_id}.png")

        if i % 50 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {elapsed:.2f}s — {out_id}.png")

    avg_ms = total_time / len(paths) * 1000
    print(f"\nProcessed {len(paths)} images in {total_time:.2f}s "
          f"({avg_ms:.1f} ms/image, CPU)")
    print(f"Results written to: {args.denoised_dir}")
    print("Filenames match submission convention (<id>.png, no suffix).")


if __name__ == "__main__":
    main()
