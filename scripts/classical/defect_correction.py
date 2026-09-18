import cv2
import numpy as np

def adaptive_defect_correction(image, base_threshold=0.2, kernel_sizes=[3, 5]):
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
        diff_sq = (channel - median_maps[0])**2
        local_var = cv2.GaussianBlur(diff_sq, (15, 15), 0)
        local_std = np.sqrt(local_var + 1e-10)
        
        adaptive_thresh = base_threshold + 2.0 * local_std
        
        # Apply correction using smallest kernel that detects the defect
        for median_map in median_maps:
            deviation = np.abs(channel - median_map)
            outlier = deviation > adaptive_thresh
            corrected[:, :, c] = np.where(outlier, median_map, corrected[:, :, c])
            
    return corrected
