import numpy as np
import cv2
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt

def fourier_features(img_path):
    # Read image in color
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process each channel separately
    channels = []
    ffts = []
    
    for channel in cv2.split(img):
        # Normalize channel to [0,1]
        channel = channel.astype(np.float32) / 255.0
        
        # Apply FFT
        fft = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft)
        ffts.append(fft_shift)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shift)
        # Apply log transform to compress dynamic range
        log_magnitude = np.log1p(magnitude)
        
        # Normalize to [0,1] range for visualization
        log_magnitude = (log_magnitude - np.min(log_magnitude)) / (np.max(log_magnitude) - np.min(log_magnitude))
        channels.append(log_magnitude)
    
    return ffts, np.stack(channels, axis=-1)

def save_fourier_image(mag, out_path):
    # mag is already normalized to [0,1] for each channel
    plt.imsave(out_path, mag)

def reconstruct_image(mag, out_path):
    # Inverse shift the magnitude spectrum back
    fft = np.fft.ifftshift(mag)
    # Take inverse FFT to get back to spatial domain
    img = np.fft.ifft2(fft)
    # Get real component and scale to 0-1 range
    img = np.real(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    plt.imsave(out_path, img, cmap='gray')

def filter_fourier_image(mag):
    # print(mag.shape)
    y, x = np.meshgrid(mag)
    center_y, center_x = mag.shape[0] // 2, mag.shape[1] // 2
    radius = min(center_x, center_y) // 2
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    # mask = np.zeros(mag.shape)
    # mask[mask.shape[0] // 2 - 100:mask.shape[0] // 2 + 100, mask.shape[1] // 2 - 100:mask.shape[1] // 2 + 100] = 1
    filtered_mag = mag * mask
    return filtered_mag

def apply_fourier_filter(
    fft_shifted: np.ndarray,
    filter_type: str,
    cutoff_freq: float,
    order: int = 1, # Only used for Butterworth
    gain_low: float = 0.5, # Only used for HighEmphasis
    gain_high: float = 1.5 # Only used for HighEmphasis
) -> np.ndarray:
    """
    Applies a specified frequency domain filter to a shifted FFT image.

    Args:
        fft_shifted: The complex 2D numpy array representing the centered
                     (shifted) Fourier Transform of the image.
        filter_type: The type of filter to apply. Options:
                     'LPF_ideal', 'HPF_ideal',
                     'LPF_gaussian', 'HPF_gaussian',
                     'LPF_butterworth', 'HPF_butterworth',
                     'HighEmphasis' (uses Gaussian HPF as base)
        cutoff_freq: The cutoff frequency (D0). Represents radius for Ideal,
                     standard deviation-like parameter for Gaussian, cutoff
                     frequency for Butterworth.
        order: The order 'n' for Butterworth filters. Higher order means
               sharper transition. (Default: 1)
        gain_low: The gain applied to low frequencies in HighEmphasis filter.
                  (Default: 0.5)
        gain_high: The gain applied to high frequencies (based on HPF) in
                   HighEmphasis filter. (Default: 1.5)

    Returns:
        filtered_image: The filtered image in the spatial domain (real part,
                        clipped to [0, 255] and converted to uint8).
    """
    if cutoff_freq <= 0:
        raise ValueError("cutoff_freq must be positive")

    rows, cols = fft_shifted.shape
    crow, ccol = rows // 2 , cols // 2

    # Create frequency distance grid (D)
    x = np.arange(cols)
    y = np.arange(rows)
    u, v = np.meshgrid(x, y)
    D = np.sqrt((u - ccol)**2 + (v - crow)**2)

    # --- Determine the filter mask H ---
    H = np.zeros((rows, cols), dtype=np.float32)

    if filter_type == 'LPF_ideal':
        H[D <= cutoff_freq] = 1
    elif filter_type == 'HPF_ideal':
        H[D > cutoff_freq] = 1

    elif filter_type == 'LPF_gaussian':
        H = np.exp(-D**2 / (2 * cutoff_freq**2))
    elif filter_type == 'HPF_gaussian':
        # Avoid division by zero if cutoff_freq is tiny near center
        # Technically, HPF = 1 - LPF. For pure HPF, center should be 0.
        if cutoff_freq == 0: H[:,:] = 1.0 # Pass everything if cutoff is 0
        else: H = 1 - np.exp(-D**2 / (2 * cutoff_freq**2))
        # Explicitly ensure the DC component (center) is zero for pure HPF
        # H[crow, ccol] = 0 # Optional: uncomment for stricter HPF

    elif filter_type == 'LPF_butterworth':
        # Avoid division by zero D=0, handle potential warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            H = 1 / (1 + (D / cutoff_freq)**(2 * order))
        H[D == 0] = 1 # Manually set DC component for LPF

    elif filter_type == 'HPF_butterworth':
         # Avoid division by zero D=0
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = 1 + (cutoff_freq / D)**(2 * order)
        H = 1 / denom
        H[D == 0] = 0 # Manually set DC component for HPF

    elif filter_type == 'HighEmphasis':
        # Based on Gaussian HPF
        if cutoff_freq == 0: hpf_base = np.ones_like(D) # Pass everything if cutoff is 0
        else: hpf_base = 1 - np.exp(-D**2 / (2 * cutoff_freq**2))
        # Optional: Explicitly ensure DC component is 0 for the HPF part if needed
        # hpf_base[crow, ccol] = 0
        H = gain_low + gain_high * hpf_base

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}. Valid types are: "
                         "'LPF_ideal', 'HPF_ideal', 'LPF_gaussian', 'HPF_gaussian', "
                         "'LPF_butterworth', 'HPF_butterworth', 'HighEmphasis'")

    # --- Apply the filter and inverse transform ---
    fft_filtered_shifted = fft_shifted * H

    # Inverse shift
    fft_filtered = np.fft.ifftshift(fft_filtered_shifted)

    # Inverse FFT
    image_filtered = np.fft.ifft2(fft_filtered)

    # Take real part and normalize/clip
    image_filtered = np.real(image_filtered)

    # Simple clipping to [0, 255] for uint8 output
    # More sophisticated normalization might be needed depending on the goal
    image_filtered = np.clip(image_filtered, 0, 255)
    image_filtered = image_filtered.astype(np.uint8)

    return image_filtered

def apply_fourier_filter_color(
    fft_shifts,
    filter_type: str,
    cutoff_freq: float,
    order: int = 1,
    gain_low: float = 0.5,
    gain_high: float = 1.5
) -> np.ndarray:
    """Apply Fourier filter to each color channel separately"""
    filtered_channels = []
    
    for fft_shift in fft_shifts:
        filtered = apply_fourier_filter(fft_shift, filter_type, cutoff_freq, order, gain_low, gain_high)
        filtered_channels.append(filtered)
    
    # Stack channels back together
    return np.stack(filtered_channels, axis=-1)

def apply_all(input_img):
    fft_shifts, features = fourier_features(input_img)
    save_fourier_image(features, output_img)
    
    # Apply high-emphasis filter to all channels
    filtered = apply_fourier_filter_color(
        fft_shifts,
        "HighEmphasis",
        cutoff_freq=10,
        gain_low=5.0,
        gain_high=1.0
    )
    
    # Normalize filtered image to [0,1] before saving
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    return filtered

if __name__ == "__main__":
    input_img = "input.png"
    output_img = "fourier_output.png"
    filtered_output = "filtered_image.png"

    # Get the Fourier features for all channels
    fft_shifts, features = fourier_features(input_img)
    save_fourier_image(features, output_img)
    
    # Apply high-emphasis filter to all channels
    filtered = apply_fourier_filter_color(
        fft_shifts,
        "HighEmphasis", 
        cutoff_freq=10,    # Edge detail control
        gain_low=5.0,      # Original structure preservation
        gain_high=1.0      # Edge enhancement amount
    )
    
    # Normalize filtered image to [0,1] before saving
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    plt.imsave(filtered_output, filtered)