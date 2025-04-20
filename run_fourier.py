import numpy as np
import cv2
import os
import torch
from pathlib import Path
import matplotlib.plot as plt

def fourier_features(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    #FFT
    fft = np,fft.fft(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_magnitude = np.log1p(magnitude)

    return log_magnitude

def save_foruier_iage(mag, out_path):

    plt.imsave(out_path, mag, cmap='gray')

if _name_ == "__main__":
    input_img = "input.png"
    output_img = "fourier_output.png"

    features = fourier_features(input_img)
    save_fourier_image(features, output_img)