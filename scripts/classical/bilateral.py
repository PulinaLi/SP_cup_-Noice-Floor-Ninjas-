import cv2
import numpy as np

def bilateral_denoise(image, d=5, sigma_color=50, sigma_space=50):
    """
    Edge-preserving bilateral filter.
    d: diameter of pixel neighborhood
    sigma_color: filter sigma in color space
    sigma_space: filter sigma in coordinate space
    """
    # OpenCV expects uint8 for bilateralFilter in most setups, or float32.
    # Since our pipeline uses float32 [0, 1], we can pass float32 directly.
    # But color/space sigmas might need tuning depending on scale.
    # We will scale to 255 for robust OpenCV behavior, then scale back.
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
    return filtered.astype(np.float32) / 255.0
