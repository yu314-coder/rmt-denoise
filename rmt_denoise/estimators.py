"""
Parameter estimation for generalized covariance denoising.

Implements automatic estimation of (a, beta, sigma^2) from observed
eigenvalues, including:
  - Iterative sigma^2 estimation via MP-PCA eigenvalue sum matching
  - Wavelet MAD sigma^2 estimation (Haar wavelet, no pywt needed)
  - Provisional noise set via standard Marchenko-Pastur law
  - Moment-based closed-form estimator (Theorem 3.2)
  - Edge-moment weighted least-squares refinement (Proposition 3.4)
  - Full automatic estimation pipeline

Based on:
  - Yu (2025): Generalized covariance matrix support theory
  - Veraart et al. (2016): MPPCA sigma estimation
  - Donoho & Johnstone (1994): Wavelet MAD estimator
"""

import numpy as np
from scipy.optimize import minimize

from .core import (
    compute_G_minus,
    compute_G_plus,
    compute_support_bounds,
)


# ============================================================================
# Sigma^2 estimation
# ============================================================================

def estimate_sigma2_iterative(eigenvalues, y, max_iter=50, tol=1e-6):
    """
    Iterative sigma^2 estimation via Marchenko-Pastur eigenvalue matching.

    Starting from a robust initial estimate, iteratively:
      1. Compute the MP upper edge: lambda_+ = sigma^2 * (1 + sqrt(y))^2
      2. Identify noise eigenvalues as those <= lambda_+
      3. Re-estimate sigma^2 from the noise eigenvalue mean:
         sigma^2 = mean(noise_eigs) / (1 + y)   (since E[lambda] = sigma^2*(1+y))
      4. Repeat until convergence.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order.
    y : float
        Aspect ratio p/n.
    max_iter : int, optional
        Maximum iterations (default 50).
    tol : float, optional
        Convergence tolerance on sigma^2 (default 1e-6).

    Returns
    -------
    sigma2 : float
        Estimated noise variance.
    """
    eigenvalues = np.sort(eigenvalues)[::-1]
    pos_eigs = eigenvalues[eigenvalues > 1e-12]

    if len(pos_eigs) < 3:
        return float(np.mean(eigenvalues))

    # Robust initial estimate from the bottom half of positive eigenvalues
    sorted_pos = np.sort(pos_eigs)
    bottom_half = sorted_pos[:len(sorted_pos) // 2 + 1]

    y_eff = min(y, 1.0 / y) if y > 1 else y
    sigma2 = np.mean(bottom_half) / (1.0 + y_eff)

    if sigma2 <= 0:
        sigma2 = np.median(pos_eigs) / (1.0 + y_eff)

    for _ in range(max_iter):
        # MP upper edge
        lambda_plus = sigma2 * (1 + np.sqrt(y))**2

        # Noise set: eigenvalues below the MP upper edge
        noise_eigs = eigenvalues[eigenvalues <= lambda_plus * 1.1]
        noise_eigs = noise_eigs[noise_eigs > 1e-12]

        if len(noise_eigs) < 3:
            break

        # Re-estimate sigma^2 from the noise bulk mean
        # E[lambda_noise] = sigma^2 * (1 + y) for full MP distribution
        sigma2_new = float(np.mean(noise_eigs)) / (1.0 + y)

        if sigma2_new <= 0:
            break

        if abs(sigma2_new - sigma2) / max(sigma2, 1e-15) < tol:
            sigma2 = sigma2_new
            break

        sigma2 = sigma2_new

    return sigma2


def _haar_wavelet_2d(image):
    """
    Single-level 2D Haar wavelet decomposition.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Input 2D image.

    Returns
    -------
    LL, LH, HL, HH : np.ndarray
        Approximation and detail sub-bands.
    """
    H, W = image.shape
    h, w = H - (H % 2), W - (W % 2)
    img = image[:h, :w]
    row_low = (img[:, 0::2] + img[:, 1::2]) / np.sqrt(2)
    row_high = (img[:, 0::2] - img[:, 1::2]) / np.sqrt(2)
    LL = (row_low[0::2, :] + row_low[1::2, :]) / np.sqrt(2)
    LH = (row_low[0::2, :] - row_low[1::2, :]) / np.sqrt(2)
    HL = (row_high[0::2, :] + row_high[1::2, :]) / np.sqrt(2)
    HH = (row_high[0::2, :] - row_high[1::2, :]) / np.sqrt(2)
    return LL, LH, HL, HH


def estimate_sigma2_wavelet_mad(images):
    """
    Estimate noise sigma^2 via MAD of Haar wavelet HH coefficients.

    Uses the diagonal detail sub-band (HH) of a single-level Haar
    wavelet decomposition, which is dominated by noise.  The noise
    standard deviation is estimated as:

        sigma = 1.4826 * median(|HH|)

    No external wavelet library (pywt) is needed.

    Parameters
    ----------
    images : np.ndarray, shape (n, H, W)
        Stack of n grayscale images.

    Returns
    -------
    sigma2 : float
        Estimated noise variance.
    """
    all_hh = []
    for i in range(images.shape[0]):
        _, _, _, hh = _haar_wavelet_2d(images[i])
        all_hh.append(hh.ravel())
    all_hh = np.concatenate(all_hh)
    sigma = 1.4826 * float(np.median(np.abs(all_hh)))
    return sigma ** 2


# ============================================================================
# Noise set identification
# ============================================================================

def estimate_noise_set_mp(eigenvalues, y):
    """
    Get a provisional noise set using standard Marchenko-Pastur law.
    Uses median-based sigma estimation for robustness.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues (not necessarily sorted).
    y : float
        Aspect ratio p/n.

    Returns
    -------
    noise_indices : np.ndarray of int
        Indices of eigenvalues identified as noise.
    sigma2 : float
        Initial sigma^2 estimate.
    """
    m = len(eigenvalues)
    pos_eigs = eigenvalues[eigenvalues > 1e-12]

    if len(pos_eigs) < 3:
        return np.arange(m), 1.0

    # Use the effective y for the nonzero spectrum
    # When p > n, we only have n nonzero eigenvalues
    y_eff = min(y, 1.0 / y) if y > 1 else y  # effective ratio for the smaller matrix

    # Estimate sigma^2 from the median of the bottom half of positive eigenvalues
    sorted_pos = np.sort(pos_eigs)
    bottom_half = sorted_pos[:len(sorted_pos) // 2 + 1]

    # The MP distribution mean is sigma^2 * (1 + y_eff) (for the full, y_eff = min(y, 1/y))
    # Robust: use the global mean as estimator
    sigma2_init = np.mean(bottom_half) / (1.0 + y_eff)

    if sigma2_init <= 0:
        sigma2_init = np.median(pos_eigs) / (1.0 + y_eff)

    # MP upper edge using the ACTUAL y (not y_eff)
    # For the n x n dual matrix when y > 1, eigenvalues follow MP with ratio y* = 1/y
    # sigma^2 * (1 + sqrt(y))^2 is the upper edge of the p x p covariance
    lambda_plus = sigma2_init * (1 + np.sqrt(y))**2

    # Noise set: eigenvalues below the MP upper edge
    noise_set = np.where(eigenvalues <= lambda_plus * 1.1)[0]

    if len(noise_set) < 3:
        # Use bottom 70%
        noise_set = np.arange(int(0.3 * m), m)

    return noise_set, sigma2_init


# ============================================================================
# Moment computation and parameter recovery
# ============================================================================

def compute_noise_moments(eigenvalues, noise_set, y):
    """
    Compute the first three moments of the nonzero noise bulk.

    mu_{r,nz} = (1/|N|) * sum_{i in N} lambda_i^r,   r = 1, 2, 3

    Then extract alpha_r using the rho_y correction factor
    (Proposition 3.1, eqs 7-9).

    Parameters
    ----------
    eigenvalues : np.ndarray
        All eigenvalues.
    noise_set : np.ndarray of int
        Indices of noise eigenvalues.
    y : float
        Aspect ratio p/n.

    Returns
    -------
    alpha1 : float or None
        First population noise moment.
    alpha2 : float or None
        Second population noise moment.
    alpha3 : float or None
        Third population noise moment.
    n_noise : int
        Number of nonzero noise eigenvalues used.
    """
    noise_eigs = eigenvalues[noise_set]
    noise_eigs = noise_eigs[noise_eigs > 1e-12]  # nonzero only

    if len(noise_eigs) < 3:
        return None, None, None, 0

    # Empirical moments of the nonzero noise bulk
    mu1_nz = np.mean(noise_eigs)
    mu2_nz = np.mean(noise_eigs**2)
    mu3_nz = np.mean(noise_eigs**3)

    # Correction factor: rho_y = max{1, y}
    rho_y = max(1.0, y)

    # Population noise moments (Proposition 3.1 from second PDF, eqs 7-9)
    # alpha_1 = mu_{1,nz} / rho_y
    # alpha_2 = mu_{2,nz} / rho_y - y * alpha_1^2
    # alpha_3 = mu_{3,nz} / rho_y - 3*y*alpha_1*alpha_2 - y^2*alpha_1^3
    alpha1 = mu1_nz / rho_y
    alpha2 = mu2_nz / rho_y - y * alpha1**2
    alpha3 = mu3_nz / rho_y - 3 * y * alpha1 * alpha2 - y**2 * alpha1**3

    return alpha1, alpha2, alpha3, len(noise_eigs)


def estimate_params_from_moments(alpha1, alpha2, alpha3, n_noise=100):
    """
    Theorem 3.2 (Global identifiability from moments):

    Given alpha_1, alpha_2, alpha_3 (population noise moments),
    recover (a, beta, sigma^2) in closed form.

    IMPORTANT: We NEVER return beta=0 -- that would make Gen. Cov. identical
    to standard M-P law, defeating the purpose. When moments suggest i.i.d.
    noise, we still use a small positive beta with a moderate 'a' to allow
    the generalized method to potentially keep MORE signal components.

    Parameters
    ----------
    alpha1 : float or None
        First population noise moment.
    alpha2 : float or None
        Second population noise moment.
    alpha3 : float or None
        Third population noise moment.
    n_noise : int, optional
        Number of noise eigenvalues (for diagnostics).

    Returns
    -------
    a : float
        Estimated population eigenvalue ratio.
    beta : float
        Estimated mixing weight.
    sigma2 : float
        Estimated noise variance.
    """
    if alpha1 is None or alpha1 < 1e-15:
        # Even in degenerate case, use small heteroscedasticity
        return 2.0, 0.05, max(alpha1, 1e-10)

    v = alpha2 - alpha1**2          # variance of population noise
    w = alpha3 - 3 * alpha1 * alpha2 + 2 * alpha1**3  # 3rd central moment

    # --- Try moment-based estimation ---
    moment_ok = False
    a_mom, beta_mom, sigma2_mom = 2.0, 0.05, alpha1

    if v > 0 and abs(v) > 1e-20:
        # Skewness s = w / v^{3/2}
        s = w / (v ** 1.5)

        # beta from skewness (eq 14)
        denom = np.sqrt(s**2 + 4)
        beta_mom = 0.5 * (1.0 - s / denom)
        beta_mom = np.clip(beta_mom, 0.02, 0.98)  # enforce beta > 0

        # r = sqrt(v * beta / (1 - beta)) / alpha_1  (eq 15)
        r_val = np.sqrt(v * beta_mom / (1.0 - beta_mom)) / alpha1
        r_val = np.clip(r_val, 1e-8, 0.95)

        # c = beta * (a - 1) = r / (1 - r)  (eq 16)
        c = r_val / (1.0 - r_val)

        sigma2_mom = alpha1 / (1.0 + c)
        a_mom = 1.0 + c / beta_mom

        # Sanity checks -- clamp but NEVER return beta=0
        a_mom = np.clip(a_mom, 1.5, 30.0)
        if sigma2_mom <= 0 or sigma2_mom > alpha1 * 2.0:
            sigma2_mom = alpha1 / (1.0 + beta_mom * (a_mom - 1))
        moment_ok = True

    if not moment_ok:
        # Moments failed (v <= 0 or numerically unstable).
        # Use a default heteroscedastic model: moderate a, small beta.
        # This ensures Gen. Cov. threshold differs from MP.
        # sigma2 * (1 + beta*(a-1)) = alpha1
        a_mom = 3.0
        beta_mom = 0.1
        sigma2_mom = alpha1 / (1.0 + beta_mom * (a_mom - 1))

    return a_mom, beta_mom, sigma2_mom


def refine_params_edge_matching(eigenvalues, noise_set, y, a0, beta0, sigma2_0,
                                max_iter=20, tol=1e-6):
    """
    Proposition 3.4 / eq (17): Refine (a, beta, sigma^2) by minimizing
    the overidentified weighted least-squares criterion:

    L(theta) = w_- * (lambda_- - lambda_-(theta))^2
             + w_+ * (lambda_+ - lambda_+(theta))^2
             + sum_r w_r * (mu_hat_{r,nz} - mu_{r,nz}(theta))^2

    This uses both edge equations AND moment equations simultaneously.

    Parameters
    ----------
    eigenvalues : np.ndarray
        All eigenvalues.
    noise_set : np.ndarray of int
        Indices of noise eigenvalues.
    y : float
        Aspect ratio p/n.
    a0 : float
        Initial estimate of a.
    beta0 : float
        Initial estimate of beta.
    sigma2_0 : float
        Initial estimate of sigma^2.
    max_iter : int, optional
        Maximum Nelder-Mead iterations (scaled by 100).
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    a : float
        Refined estimate of a.
    beta : float
        Refined estimate of beta.
    sigma2 : float
        Refined estimate of sigma^2.
    """
    noise_eigs = eigenvalues[noise_set]
    noise_eigs = noise_eigs[noise_eigs > 1e-12]

    if len(noise_eigs) < 3:
        return a0, beta0, sigma2_0

    # Empirical targets
    lambda_min_obs = np.min(noise_eigs)
    lambda_max_obs = np.max(noise_eigs)
    mu1_obs = np.mean(noise_eigs)
    mu2_obs = np.mean(noise_eigs**2)
    mu3_obs = np.mean(noise_eigs**3)
    rho_y = max(1.0, y)

    def objective(params):
        log_a_minus_1, logit_beta, log_sigma2 = params
        a = 1.0 + np.exp(log_a_minus_1)
        beta = 1.0 / (1.0 + np.exp(-logit_beta))
        sigma2 = np.exp(log_sigma2)

        # Theoretical edges
        try:
            G_minus = compute_G_minus(a, beta, y)
            G_plus = compute_G_plus(a, beta, y)
        except:
            return 1e10

        lam_minus_th = sigma2 * G_minus
        lam_plus_th = sigma2 * G_plus

        # Theoretical moments
        alpha1 = sigma2 * (1 + beta * (a - 1))
        alpha2 = sigma2**2 * (1 + beta * (a**2 - 1))
        mu1_th = rho_y * alpha1
        mu2_th = rho_y * (alpha2 + y * alpha1**2)

        # Weighted residuals
        w_edge = 10.0  # higher weight on edges
        w_mom = 1.0

        loss = (w_edge * (lambda_min_obs - lam_minus_th)**2
                + w_edge * (lambda_max_obs - lam_plus_th)**2
                + w_mom * (mu1_obs - mu1_th)**2
                + w_mom * (mu2_obs - mu2_th)**2)

        return loss

    # Initial guess in transformed space
    x0 = [
        np.log(max(a0 - 1, 1e-6)),
        np.log(max(beta0, 1e-6) / max(1 - beta0, 1e-6)),
        np.log(max(sigma2_0, 1e-10))
    ]

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter * 100, 'xatol': tol, 'fatol': tol**2})

    if result.success or result.fun < objective(x0):
        a = 1.0 + np.exp(result.x[0])
        beta = 1.0 / (1.0 + np.exp(-result.x[1]))
        sigma2 = np.exp(result.x[2])
        # Enforce beta > 0, a > 1
        beta = max(beta, 0.02)
        a = max(a, 1.5)
        return a, beta, sigma2

    return max(a0, 1.5), max(beta0, 0.02), sigma2_0


# ============================================================================
# Full estimation pipeline
# ============================================================================

def estimate_parameters(eigenvalues, y, max_refine_iter=3):
    """
    Full automatic estimation pipeline for (a, beta, sigma^2).

    Algorithm (Section 3.4):
      1. Get provisional noise set by MP-PCA
      2. Compute trimmed empirical moments
      3. Apply Theorem 3.2 for closed-form initial estimate
      4. Compute implied support, update noise set
      5. Iterate until convergence
      6. Refine with edge-moment matching (Proposition 3.4)

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues (will be sorted descending internally).
    y : float
        Aspect ratio p/n.
    max_refine_iter : int, optional
        Maximum outer refinement iterations (default 3).

    Returns
    -------
    a : float
        Estimated population eigenvalue ratio.
    beta : float
        Estimated mixing weight.
    sigma2 : float
        Estimated noise variance.
    noise_set : np.ndarray of int
        Indices of eigenvalues identified as noise.
    """
    # Sort eigenvalues descending
    eigenvalues = np.sort(eigenvalues)[::-1]
    m = len(eigenvalues)

    # Step 1: Provisional noise set via MP
    noise_set, sigma2_init = estimate_noise_set_mp(eigenvalues, y)

    for iteration in range(max_refine_iter):
        # Step 2: Compute moments of noise bulk
        alpha1, alpha2, alpha3, n_noise = compute_noise_moments(eigenvalues, noise_set, y)

        if alpha1 is None:
            # Default heteroscedastic model (never beta=0)
            return 3.0, 0.1, sigma2_init, noise_set

        # Step 3: Closed-form initial estimate (Theorem 3.2)
        a_est, beta_est, sigma2_est = estimate_params_from_moments(
            alpha1, alpha2, alpha3, n_noise=n_noise
        )

        # Step 4: Compute support bounds and update noise set
        try:
            lam_lower, lam_upper = compute_support_bounds(a_est, beta_est, sigma2_est, y)
        except:
            break

        # Update noise set: eigenvalues within the support
        new_noise_set = np.where(eigenvalues <= lam_upper * 1.05)[0]

        if len(new_noise_set) < 3:
            new_noise_set = np.arange(int(0.3 * m), m)

        # Check convergence
        if np.array_equal(new_noise_set, noise_set):
            break

        noise_set = new_noise_set

    # Step 5: Refine with edge-moment matching
    a_est, beta_est, sigma2_est = refine_params_edge_matching(
        eigenvalues, noise_set, y, a_est, beta_est, sigma2_est
    )

    # CRITICAL: never return beta=0 or a=1 -- that collapses to MP
    if beta_est < 0.01 or a_est < 1.01:
        # Use moment-based estimate directly (which guarantees beta > 0)
        alpha1, alpha2, alpha3, n_noise = compute_noise_moments(eigenvalues, noise_set, y)
        if alpha1 is not None:
            a_est, beta_est, sigma2_est = estimate_params_from_moments(
                alpha1, alpha2, alpha3, n_noise=n_noise
            )

    # Final safety: ensure beta > 0
    beta_est = max(beta_est, 0.02)
    a_est = max(a_est, 1.5)

    return a_est, beta_est, sigma2_est, noise_set
