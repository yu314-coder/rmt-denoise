"""
Quality metrics for evaluating denoising performance.

Provides PSNR and SSIM computed from numpy arrays without requiring
external image quality libraries.
"""

import numpy as np


def compute_psnr(clean, denoised, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

        PSNR = 10 * log10(max_val^2 / MSE)

    Parameters
    ----------
    clean : np.ndarray
        Ground-truth clean image(s).
    denoised : np.ndarray
        Denoised image(s), same shape as clean.
    max_val : float, optional
        Maximum possible pixel value (default 1.0 for [0, 1] images).

    Returns
    -------
    float
        PSNR in decibels. Returns 100.0 if MSE is effectively zero.
    """
    mse = np.mean((clean - denoised)**2)
    if mse < 1e-15:
        return 100.0
    return 10 * np.log10(max_val**2 / mse)


def compute_ssim(img1, img2, max_val=1.0, window_size=7):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Uses uniform-filter (box filter) local statistics following the
    original Wang et al. (2004) formulation:

        SSIM(x, y) = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
                     -------------------------------------------
                     (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2)

    where C1 = (0.01 * L)^2, C2 = (0.03 * L)^2, L = max_val.

    Parameters
    ----------
    img1 : np.ndarray
        First image (2D array).
    img2 : np.ndarray
        Second image (2D array), same shape as img1.
    max_val : float, optional
        Dynamic range of the images (default 1.0).
    window_size : int, optional
        Size of the uniform averaging window (default 7).

    Returns
    -------
    float
        Mean SSIM value over the image.
    """
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = uniform_filter(img1.astype(np.float64), window_size)
    mu2 = uniform_filter(img2.astype(np.float64), window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1.astype(np.float64)**2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2.astype(np.float64)**2, window_size) - mu2_sq
    sigma12 = uniform_filter(img1.astype(np.float64) * img2.astype(np.float64),
                              window_size) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))
