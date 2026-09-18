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
            # Use median absolute deviation
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
        
        # Reconstruct and fix any minor shape mismatches from pywt padding
        reconstructed = pywt.waverec2(thresholded, wavelet)
        h, w = channel.shape
        denoised_channels.append(reconstructed[:h, :w])
    
    return np.clip(np.stack(denoised_channels, axis=-1), 0, 1)
