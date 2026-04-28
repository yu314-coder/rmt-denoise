"""
de-noise: Image Denoising via Random Matrix Theory
====================================================

Two methods:
  - MPLawDenoiser: Standard Marčenko-Pastur law eigenvalue thresholding
  - GeneralizedCovDenoiser: Generalized covariance matrix with H = β·δ_a + (1-β)·δ_1

Usage:
    from denoise import MPLawDenoiser, GeneralizedCovDenoiser

    mp = MPLawDenoiser()
    denoised = mp.denoise(noisy_images)

    gc = GeneralizedCovDenoiser()
    denoised = gc.denoise(noisy_images)
"""

__version__ = "2.2.2"

from .mp_law import MPLawDenoiser
from .generalized_cov import GeneralizedCovDenoiser
from .noise import (
    add_gaussian_noise,
    add_laplacian_noise,
    add_mixture_gaussian_noise,
    add_structured_noise,
)
from .metrics import compute_psnr, compute_ssim
from .core import (
    g_function,
    compute_support_bounds,
    compute_discriminant,
    compute_explicit_support,
)
from .io import load_folder, split_train_test

__all__ = [
    "MPLawDenoiser",
    "GeneralizedCovDenoiser",
    "load_folder",
    "split_train_test",
    "add_gaussian_noise",
    "add_laplacian_noise",
    "add_mixture_gaussian_noise",
    "add_structured_noise",
    "compute_psnr",
    "compute_ssim",
    "g_function",
    "compute_support_bounds",
    "compute_discriminant",
    "compute_explicit_support",
]
