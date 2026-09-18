import numpy as np
from scipy.signal import wiener

def frequency_analysis_and_filter(image, cutoff_ratio=0.8):
    """
    Analyze noise in frequency domain and optionally apply low-pass filtering.
    """
    filtered_channels = []
    for c in range(image.shape[2]):
        # DFT
        f_transform = np.fft.fft2(image[:, :, c])
        f_shift = np.fft.fftshift(f_transform)
        
        # Butterworth low-pass filter (smooth rolloff)
        rows, cols = image.shape[:2]
        crow, ccol = rows // 2, cols // 2
        D0 = cutoff_ratio * min(rows, cols) / 2
        
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols).reshape(1, -1) - ccol
        D = np.sqrt(u**2 + v**2)
        n_order = 2
        
        # Avoid division by zero
        H = 1 / (1 + (D / (D0 + 1e-5))**(2 * n_order))
        
        # Apply filter
        filtered = f_shift * H
        f_ishift = np.fft.ifftshift(filtered)
        result = np.real(np.fft.ifft2(f_ishift))
        filtered_channels.append(result)
    
    return np.clip(np.stack(filtered_channels, axis=-1), 0, 1)

def wiener_denoise(image, noise_variance=None):
    """Wiener filter - optimal MMSE linear estimator."""
    denoised = np.zeros_like(image)
    for c in range(image.shape[2]):
        denoised[:, :, c] = wiener(image[:, :, c], noise=noise_variance)
    return np.clip(denoised, 0, 1)
