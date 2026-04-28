"""
Noise generation utilities for denoising experiments.

Provides functions to add various types of noise to image stacks,
each returning both the noisy images and the true noise variance.

Supports:
  - i.i.d. Gaussian noise
  - i.i.d. Laplacian noise
  - Mixture of Gaussians noise
  - Structured (heteroscedastic) Gaussian noise matching the
    generalized covariance model H = beta*delta_a + (1-beta)*delta_1
"""

import numpy as np


def add_gaussian_noise(images, sigma=0.1):
    """
    Add i.i.d. Gaussian noise N(0, sigma^2) to images.

    Parameters
    ----------
    images : np.ndarray
        Input images. Shape (n, H, W) for a stack or (H, W) for a single image.
    sigma : float, optional
        Noise standard deviation (default 0.1).

    Returns
    -------
    noisy : np.ndarray
        Noisy images clipped to [0, 1], same shape as input.
    variance : float
        True noise variance (sigma^2).
    """
    noise = np.random.normal(0, sigma, images.shape)
    return np.clip(images + noise, 0, 1), sigma**2


def add_laplacian_noise(images, b=0.07):
    """
    Add i.i.d. Laplacian noise with scale parameter b.

    The Laplacian distribution has variance = 2 * b^2.

    Parameters
    ----------
    images : np.ndarray
        Input images. Shape (n, H, W) for a stack or (H, W) for a single image.
    b : float, optional
        Laplacian scale parameter (default 0.07).

    Returns
    -------
    noisy : np.ndarray
        Noisy images clipped to [0, 1], same shape as input.
    variance : float
        True noise variance (2 * b^2).
    """
    noise = np.random.laplace(0, b, images.shape)
    return np.clip(images + noise, 0, 1), 2 * b**2


def add_mixture_gaussian_noise(images, sigma1=0.1, sigma2=0.3, beta=0.1, mu2=0.0):
    """
    Add mixture of Gaussians noise:
        (1-beta) * N(0, sigma1^2) + beta * N(mu2, sigma2^2)

    Overall variance = (1-beta)*sigma1^2 + beta*sigma2^2

    (The full variance including the mean shift would add
    beta*(1-beta)*mu2^2, but this function returns the simpler form
    matching the source implementation.)

    Parameters
    ----------
    images : np.ndarray
        Input images. Shape (n, H, W) for a stack or (H, W) for a single image.
    sigma1 : float, optional
        Std dev of the primary Gaussian component (default 0.1).
    sigma2 : float, optional
        Std dev of the secondary Gaussian component (default 0.3).
    beta : float, optional
        Mixing weight for the secondary component (default 0.1).
    mu2 : float, optional
        Mean of the secondary Gaussian component (default 0.0).

    Returns
    -------
    noisy : np.ndarray
        Noisy images clipped to [0, 1], same shape as input.
    variance : float
        True noise variance: (1-beta)*sigma1^2 + beta*sigma2^2.
    """
    n = images.size
    mask = np.random.binomial(1, beta, images.shape)
    noise1 = np.random.normal(0, sigma1, images.shape)
    noise2 = np.random.normal(mu2, sigma2, images.shape)
    noise = (1 - mask) * noise1 + mask * noise2
    total_var = (1 - beta) * sigma1**2 + beta * sigma2**2
    return np.clip(images + noise, 0, 1), total_var


def add_structured_noise(images, a=5.0, beta=0.1, sigma=0.1):
    """
    Add heteroscedastic Gaussian noise matching
    H = beta*delta_a + (1-beta)*delta_1.

    Multiply by T^{1/2} = diag(sqrt(a), ..., sqrt(a), 1, ..., 1) then
    add Gaussian.  A fraction beta of the dimensions have noise variance
    sigma^2 * a; the rest have noise variance sigma^2.

    Parameters
    ----------
    images : np.ndarray
        Input images. Shape (n, H, W) for a stack or (H, W) for a single image.
    a : float, optional
        Eigenvalue ratio for the high-noise dimensions (default 5.0).
    beta : float, optional
        Fraction of dimensions with elevated noise (default 0.1).
    sigma : float, optional
        Base noise standard deviation (default 0.1).

    Returns
    -------
    noisy : np.ndarray
        Noisy images clipped to [0, 1], same shape as input.
    variance : float
        Average noise variance: sigma^2 * (1 + beta*(a-1)).
    """
    shape = images.shape
    if images.ndim == 3:
        n, H, W = shape
        p = H * W
        flat = images.reshape(n, p)

        # Create T^{1/2} diagonal
        n_high = int(beta * p)
        scale = np.ones(p)
        scale[:n_high] = np.sqrt(a)

        noise = np.random.normal(0, sigma, flat.shape)
        noise = noise * scale  # heteroscedastic

        noisy = np.clip(flat + noise, 0, 1).reshape(shape)
        total_var = sigma**2 * (1 + beta * (a - 1))
        return noisy, total_var
    else:
        H, W = shape
        p = H * W
        flat = images.flatten()

        n_high = int(beta * p)
        scale = np.ones(p)
        scale[:n_high] = np.sqrt(a)

        noise = np.random.normal(0, sigma, p)
        noise = noise * scale

        noisy = np.clip(flat + noise, 0, 1).reshape(shape)
        total_var = sigma**2 * (1 + beta * (a - 1))
        return noisy, total_var
