"""
Core math and utilities for generalized covariance denoising.

Implements the g(t) function, support bounds, P4 polynomial, discriminant,
and shared SVD reconstruction / matrix conversion helpers.

Based on:
  - Yu (2025): Geometric Analysis of the Eigenvalue Range of the Generalized
    Covariance Matrix (support bounds via g(t) function, Theorems 2.2 and 3.1)
"""

import numpy as np
from scipy.optimize import brentq


# ============================================================================
# g(t) function and its derivative (Yu 2025, Theorem 3.1)
# ============================================================================

def g_function(t, a, beta, y):
    """
    The critical-point function g(t) from Yu (2025):

        g(t) = y*beta*(a-1)*t + (a*t+1)*((y-1)*t - 1)
               -----------------------------------------
                       (a*t+1)*(t^2 + t)

    Equivalently:  g(t) = -1/t + y*(beta*a/(1+a*t) + (1-beta)/(1+t))

    Parameters
    ----------
    t : float
        Evaluation point (must be nonzero).
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    y : float
        Aspect ratio p/n (y > 0).

    Returns
    -------
    float
        Value of g(t).
    """
    # Use the second form for numerical stability
    if abs(t) < 1e-15:
        return np.inf
    return -1.0 / t + y * (beta * a / (1.0 + a * t) + (1.0 - beta) / (1.0 + t))


def g_derivative(t, a, beta, y):
    """
    Derivative g'(t) = 1/t^2 - y*beta*a^2/(1+at)^2 - y*(1-beta)/(1+t)^2

    Parameters
    ----------
    t : float
        Evaluation point.
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    y : float
        Aspect ratio p/n.

    Returns
    -------
    float
        Value of g'(t).
    """
    return (1.0 / t**2
            - y * beta * a**2 / (1.0 + a * t)**2
            - y * (1.0 - beta) / (1.0 + t)**2)


# ============================================================================
# Support edge computation
# ============================================================================

def compute_G_plus(a, beta, y):
    """
    Upper noise edge: G_+(a, beta, y) = min_{-1/a < t < 0} g(t)

    This is the minimum of g on (-1/a, 0), which exists and is unique
    since g is convex there (g'' > 0).

    Parameters
    ----------
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    y : float
        Aspect ratio p/n.

    Returns
    -------
    float
        The upper edge G_+ of the noise eigenvalue support.
    """
    if a <= 1.0 + 1e-12 or beta < 1e-12:
        # Standard MP case: G+ = (1 + sqrt(y))^2
        return (1.0 + np.sqrt(y))**2

    # Find the minimizer in (-1/a, 0) using Brent's method on g'
    left = -1.0 / a + 1e-10
    right = -1e-10

    try:
        t_star = brentq(lambda t: g_derivative(t, a, beta, y), left, right)
        return g_function(t_star, a, beta, y)
    except (ValueError, RuntimeError):
        # Fallback: grid search
        ts = np.linspace(left, right, 10000)
        gs = np.array([g_function(t, a, beta, y) for t in ts])
        return np.min(gs)


def compute_G_minus(a, beta, y):
    """
    Lower noise edge:
      G_-(a, beta, y) = max_{t > 0} g(t)     if y >= 1
                       = max_{t < -1} g(t)    if 0 < y < 1

    Parameters
    ----------
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    y : float
        Aspect ratio p/n.

    Returns
    -------
    float
        The lower edge G_- of the noise eigenvalue support.
    """
    if a <= 1.0 + 1e-12 or beta < 1e-12:
        # Standard MP case
        return (1.0 - np.sqrt(y))**2

    if y >= 1.0:
        # Find maximizer on (0, inf) -- g has unique max there
        try:
            # g' goes from +inf at 0+ to negative as t->inf
            # Search for root of g'
            t_star = brentq(lambda t: g_derivative(t, a, beta, y), 1e-8, 1e6)
            return g_function(t_star, a, beta, y)
        except (ValueError, RuntimeError):
            ts = np.logspace(-6, 4, 10000)
            gs = np.array([g_function(t, a, beta, y) for t in ts])
            return np.max(gs)
    else:
        # y < 1: find maximizer on (-inf, -1)
        try:
            t_star = brentq(lambda t: g_derivative(t, a, beta, y), -1e6, -1.0 - 1e-8)
            return g_function(t_star, a, beta, y)
        except (ValueError, RuntimeError):
            ts = np.linspace(-100, -1.001, 10000)
            gs = np.array([g_function(t, a, beta, y) for t in ts])
            return np.max(gs)


def compute_support_bounds(a, beta, sigma2, y):
    """
    Compute the support of the noise eigenvalue distribution.

    Parameters
    ----------
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    sigma2 : float
        Noise variance.
    y : float
        Aspect ratio p/n.

    Returns
    -------
    lambda_lower : float
        Lower bound: sigma^2 * G_-(a, beta, y).
    lambda_upper : float
        Upper bound: sigma^2 * G_+(a, beta, y).
    """
    G_plus = compute_G_plus(a, beta, y)
    G_minus = compute_G_minus(a, beta, y)
    return sigma2 * G_minus, sigma2 * G_plus


# ============================================================================
# P4 polynomial and discriminant Delta (Yu 2025, Theorem 2.2)
# ============================================================================

def compute_P4_coefficients(a, beta, y):
    """
    Coefficients of P_4(t) = c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0
    which determines g'(t) = 0 and hence the support structure.

    Parameters
    ----------
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    y : float
        Aspect ratio p/n.

    Returns
    -------
    c4, c3, c2, c1, c0 : float
        Coefficients from degree 4 down to degree 0.
    """
    c4 = a**2 * (1 - y)
    c3 = 2 * a**2 * (1 - y * beta) + 2 * a * (1 - y * (1 - beta))
    c2 = a**2 * (1 - y * beta) + 4 * a + 1 - y * (1 - beta)
    c1 = 2 * a + 2
    c0 = 1.0
    return c4, c3, c2, c1, c0


def compute_discriminant(a, beta, y):
    """
    Compute the discriminant Delta from Yu (2025) eq. (6)/(13).

    Delta < 0 => two disjoint intervals (Case 1)
    Delta > 0 => single interval (Case 2)
    Delta = 0 => degenerate (Case 3)

    Parameters
    ----------
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    y : float
        Aspect ratio p/n.

    Returns
    -------
    float
        The discriminant Delta.
    """
    c4, c3, c2, c1, c0 = compute_P4_coefficients(a, beta, y)

    D = 3 * c3**2 - 8 * c4 * c2
    E = -c3**3 + 4 * c4 * c3 * c2 - 8 * c4**2 * c1
    F = 3 * c3**4 + 16 * c4**2 * c2**2 - 16 * c4 * c3**2 * c2 \
        + 16 * c4**2 * c3 * c1 - 64 * c4**3 * c0

    A = D**2 - 3 * F
    B = D * F - 9 * E**2
    C = F**2 - 3 * D * E**2

    Delta = B**2 - 4 * A * C
    return Delta


def compute_explicit_support(a, beta, sigma2, y):
    """
    Compute the explicit support intervals using the discriminant.

    When Delta >= 0, the noise eigenvalue distribution has a single
    connected support interval.  When Delta < 0, there are two disjoint
    intervals.

    Parameters
    ----------
    a : float
        Population eigenvalue ratio (a >= 1).
    beta : float
        Mixing weight in [0, 1].
    sigma2 : float
        Noise variance.
    y : float
        Aspect ratio p/n.

    Returns
    -------
    list of (float, float)
        List of (lower, upper) tuples for each support interval.
    """
    Delta = compute_discriminant(a, beta, y)
    G_plus = compute_G_plus(a, beta, y)
    G_minus = compute_G_minus(a, beta, y)

    if Delta >= 0:
        # Single interval
        return [(sigma2 * G_minus, sigma2 * G_plus)]
    else:
        # Two disjoint intervals -- need the 4 critical points
        c4, c3, c2, c1, c0 = compute_P4_coefficients(a, beta, y)
        roots = np.roots([c4, c3, c2, c1, c0])
        real_roots = np.sort(roots[np.abs(roots.imag) < 1e-8].real)

        if len(real_roots) >= 4:
            x1, x2, x3, x4 = real_roots[:4]
            g_vals = [g_function(x, a, beta, y) for x in real_roots[:4]]
            g_vals_sorted = sorted(enumerate(g_vals), key=lambda x: x[1])

            # Two intervals
            return [
                (sigma2 * min(g_vals), sigma2 * sorted(g_vals)[1]),
                (sigma2 * sorted(g_vals)[2], sigma2 * max(g_vals))
            ]
        else:
            return [(sigma2 * G_minus, sigma2 * G_plus)]


# ============================================================================
# SVD reconstruction helper
# ============================================================================

def svd_denoise(X, signal_mask, svs, U, Vt, n):
    """
    Shared SVD reconstruction: zero out noise components and rebuild X.

    Given the SVD components and a boolean mask indicating which singular
    values correspond to signal, reconstruct the denoised data matrix.

    Parameters
    ----------
    X : np.ndarray, shape (p, n)
        Original (centered) data matrix.
    signal_mask : np.ndarray of bool, shape (k,)
        True for signal components, False for noise.
    svs : np.ndarray, shape (k,)
        Singular values (or shrunk singular values).
    U : np.ndarray, shape (p, k)
        Left singular vectors.
    Vt : np.ndarray, shape (k, n)
        Right singular vectors (transposed).
    n : int
        Number of observations (columns of original X).

    Returns
    -------
    X_denoised : np.ndarray, shape (p, n)
        Reconstructed denoised matrix.
    """
    shrunk_svs = svs * signal_mask[:len(svs)]
    X_denoised = (U[:, :len(shrunk_svs)] * shrunk_svs) @ Vt[:len(shrunk_svs), :]
    return X_denoised


# ============================================================================
# Image <-> matrix conversion
# ============================================================================

def images_to_matrix(images):
    """
    Convert a stack of grayscale images to a data matrix.

    Parameters
    ----------
    images : np.ndarray, shape (n, H, W)
        Stack of n grayscale images.

    Returns
    -------
    X : np.ndarray, shape (p, n)
        Data matrix where p = H * W and each column is a flattened image.
    """
    n, H, W = images.shape
    return images.reshape(n, H * W).T


def matrix_to_images(X, H, W):
    """
    Convert a data matrix back to a stack of grayscale images.

    Parameters
    ----------
    X : np.ndarray, shape (p, n)
        Data matrix where p = H * W.
    H : int
        Image height.
    W : int
        Image width.

    Returns
    -------
    images : np.ndarray, shape (n, H, W)
        Stack of n grayscale images.
    """
    return X.T.reshape(-1, H, W)
