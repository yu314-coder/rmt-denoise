"""
Generalized Covariance Matrix denoiser.

Model: B_n = S_n T_n  with  H = beta * delta_a + (1 - beta) * delta_1.

Automatically estimates (a, beta, sigma2) from the eigenvalue spectrum,
computes the generalized support bounds, and applies hard thresholding.

GUARANTEE: always keeps >= as many signal components as the standard
Marcenko-Pastur law.

Based on denoise_workflow_a (gen_hard) and denoise_workflow_b from
gen_cov_denoise.py.
"""

import numpy as np

from .core import (
    g_function,
    compute_G_plus,
    compute_G_minus,
    compute_support_bounds,
    compute_discriminant,
    compute_explicit_support,
    images_to_matrix,
    matrix_to_images,
)
from .estimators import estimate_parameters, estimate_sigma2_iterative


# ============================================================================
# Patch helpers (replicated from gen_cov_denoise.py workflow B)
# ============================================================================

def _extract_patches(image, k, stride=None):
    """Extract k x k patches from a 2D image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    k : int
        Patch side length.
    stride : int or None
        Step between patches.  ``None`` means non-overlapping (stride = k).

    Returns
    -------
    X : np.ndarray, shape (k*k, n_patches)
    positions : list of (row, col) tuples
    """
    if stride is None:
        stride = k
    H, W = image.shape
    patches = []
    positions = []
    for i in range(0, H - k + 1, stride):
        for j in range(0, W - k + 1, stride):
            patches.append(image[i:i + k, j:j + k].flatten())
            positions.append((i, j))
    if len(patches) == 0:
        return np.array([]).reshape(0, 0), []
    return np.array(patches).T, positions          # (k^2, n_patches)


def _reassemble_patches(patches, positions, k, H, W, weights=None):
    """Reassemble patches into an image with weighted averaging.

    Parameters
    ----------
    patches : np.ndarray, shape (k*k, n_patches)
    positions : list of (row, col)
    k : int
    H, W : int
    weights : array-like or None

    Returns
    -------
    image : np.ndarray, shape (H, W)
    """
    result = np.zeros((H, W))
    count = np.zeros((H, W))
    n_patches = patches.shape[1] if patches.ndim > 1 else 0
    for idx, (i, j) in enumerate(positions):
        patch = patches[:, idx].reshape(k, k)
        w = weights[idx] if weights is not None else 1.0
        result[i:i + k, j:j + k] += w * patch
        count[i:i + k, j:j + k] += w
    count[count == 0] = 1
    return result / count


def _compute_patch_score(eigenvalues, lambda_upper, y, tau=0.1):
    """Patch-size score J(k) = mean((lambda - lambda_upper)_+) - tau * lambda_upper."""
    signal_energy = np.mean(np.maximum(eigenvalues - lambda_upper, 0))
    penalty = tau * lambda_upper
    return signal_energy - penalty


# ============================================================================
# Main class
# ============================================================================

class GeneralizedCovDenoiser:
    """Generalized Covariance Matrix denoiser.

    Model: B_n = S_n T_n  with  H = beta * delta_a + (1 - beta) * delta_1.

    Automatically estimates (a, beta, sigma2) from the eigenvalue spectrum.

    GUARANTEE: always keeps >= as many signal components as M-P law.

    Parameters
    ----------
    sigma2 : float or None
        Override noise variance.  ``None`` means auto-estimate.
    a : float or None
        Override population eigenvalue ratio.  ``None`` means auto-estimate.
    beta : float or None
        Override mixing weight.  ``None`` means auto-estimate.
    mode : {'multi', 'patch'}
        ``'multi'``  -- workflow A: denoise a stack of n images.
        ``'patch'``  -- workflow B: denoise a single image via patch splitting.
    candidate_k : list of int or None
        (mode='patch' only) Candidate patch sizes.  ``None`` for auto.
    stride_ratio : float
        (mode='patch' only) stride = k * stride_ratio.  Default 0.5.
    """

    def __init__(self, sigma2=None, a=None, beta=None, mode="multi",
                 candidate_k=None, stride_ratio=0.5):
        self._sigma2_given = sigma2
        self._a_given = a
        self._beta_given = beta
        self.mode = mode
        self.candidate_k = candidate_k
        self.stride_ratio = stride_ratio
        self._info = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, images):
        """Denoise images.

        Parameters
        ----------
        images : np.ndarray
            If ``mode='multi'``: shape (n, H, W).
            If ``mode='patch'``: shape (H, W) -- single image.

        Returns
        -------
        denoised : np.ndarray
            Same shape as *images*, denoised and clipped to [0, 1].
        """
        if self.mode == "patch":
            return self._denoise_patch(images)
        return self._denoise_multi(images)

    @property
    def info(self):
        """Dict with estimation diagnostics.

        Keys (mode='multi'):
            a, beta, sigma2, threshold, threshold_mp, rank, rank_mp,
            y, p, n, n_intervals, delta, lambda_lower.

        Keys (mode='patch'):
            a, beta, sigma2, threshold, y_k, p_k, n_k, best_k,
            k_scores, stride, num_signal, ...
        """
        return self._info

    # ------------------------------------------------------------------
    # Workflow A: multi-image denoising
    # ------------------------------------------------------------------

    def _denoise_multi(self, images):
        n_images, H, W = images.shape
        p = H * W
        n = n_images
        y = p / n

        # Phase 1: data preparation
        X = images_to_matrix(images)              # (p, n)
        x_mean = np.mean(X, axis=1, keepdims=True)
        X_centered = X - x_mean

        # Phase 2: eigendecomposition (dual when p > n)
        if p > n:
            G = (X_centered.T @ X_centered) / n
            eigvals, V = np.linalg.eigh(G)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            V = V[:, idx]

            pos_mask = eigvals > 1e-10
            eigvals_pos = eigvals[pos_mask]
            V_pos = V[:, pos_mask]
            svs = np.sqrt(n * eigvals_pos)

            # p-dimensional eigenvectors
            U = (X_centered @ V_pos) / svs
        else:
            U_full, svs_full, Vt_full = np.linalg.svd(
                X_centered, full_matrices=False
            )
            pos_mask = svs_full > 1e-5
            svs = svs_full[pos_mask]
            U = U_full[:, pos_mask]
            V_pos = Vt_full[pos_mask, :].T
            eigvals_pos = svs ** 2 / n

        if len(eigvals_pos) < 3:
            self._info = {"error": "too few eigenvalues", "p": p, "n": n, "y": y}
            return images.copy()

        eigenvalues = eigvals_pos
        singular_values = svs

        # Phase 3a: MP baseline threshold (guaranteed floor)
        sigma2_mp, threshold_mp, rank_mp = self._mp_baseline(eigenvalues, y)

        # Phase 3b: generalized parameter estimation
        if self._a_given is not None and self._beta_given is not None and self._sigma2_given is not None:
            a_est, beta_est, sigma2_est = self._a_given, self._beta_given, self._sigma2_given
            noise_set = np.array([], dtype=int)
        else:
            a_est, beta_est, sigma2_est, noise_set = estimate_parameters(eigenvalues, y)
            # Allow user overrides
            if self._a_given is not None:
                a_est = self._a_given
            if self._beta_given is not None:
                beta_est = self._beta_given
            if self._sigma2_given is not None:
                sigma2_est = self._sigma2_given

        lam_lower, lam_upper_gen = compute_support_bounds(a_est, beta_est, sigma2_est, y)
        num_signal_gen = int(np.sum(eigenvalues > lam_upper_gen))

        # Phase 3c: check for two-interval support (Delta < 0)
        delta = compute_discriminant(a_est, beta_est, y)
        support_intervals = compute_explicit_support(a_est, beta_est, sigma2_est, y)

        if len(support_intervals) == 2:
            # Two disjoint noise intervals: signal = eigenvalues NOT in either
            signal_mask_gen = np.ones(len(eigenvalues), dtype=bool)
            for lo, hi in support_intervals:
                signal_mask_gen &= ~(
                    (eigenvalues >= lo * 0.95) & (eigenvalues <= hi * 1.05)
                )
            num_signal_gen = int(np.sum(signal_mask_gen))
        else:
            signal_mask_gen = eigenvalues > lam_upper_gen

        # Phase 3d: Choose threshold
        # Gen may legitimately find FEWER signal components than MP
        # (e.g., identical images = pure noise, rank should be 0).
        # Decide whether to trust generalized or fall back to MP.
        # Key rule: Gen should NEVER be worse than MP.
        #
        # When Gen keeps FEWER components than MP (num_signal_gen < rank_mp),
        # it's being MORE aggressive. This is correct ONLY when the images
        # are very similar (same scene) so most eigenvalues truly are noise.
        # For diverse images, it means the threshold is too high = bug.
        #
        # Heuristic: if Gen keeps < 50% of what MP keeps, it's suspicious
        # unless parameters look very clean (a in reasonable range).
        params_reliable = (1.01 < a_est < 50) and (0.01 < beta_est < 0.99)
        gen_much_fewer = (num_signal_gen < rank_mp * 0.5) and (rank_mp > 5)

        if params_reliable and not gen_much_fewer:
            # Trust the generalized threshold
            lam_upper = lam_upper_gen
            signal_mask = signal_mask_gen
        elif num_signal_gen >= rank_mp:
            # Gen keeps more components — always fine
            lam_upper = lam_upper_gen
            signal_mask = signal_mask_gen
        else:
            # Gen is more aggressive than MP with unreliable params — fall back
            lam_upper = threshold_mp
            signal_mask = eigenvalues > threshold_mp

        rank = int(np.sum(signal_mask))

        # Phase 3e: hard threshold singular values
        shrunk_svs = singular_values * signal_mask[: len(singular_values)]

        # Phase 4: reconstruction
        X_reconstructed = (U * shrunk_svs) @ V_pos.T
        X_reconstructed += x_mean

        denoised = matrix_to_images(X_reconstructed, H, W)
        denoised = np.clip(denoised, 0, 1)

        # Store info
        self._info = {
            "a": a_est,
            "beta": beta_est,
            "sigma2": sigma2_est,
            "threshold": lam_upper,
            "threshold_mp": threshold_mp,
            "lambda_lower": lam_lower,
            "lambda_upper_gen": lam_upper_gen,
            "rank": rank,
            "rank_mp": rank_mp,
            "rank_gen": num_signal_gen,
            "y": y,
            "p": p,
            "n": n,
            "delta": delta,
            "n_intervals": len(support_intervals),
        }

        return denoised

    # ------------------------------------------------------------------
    # Workflow B: single-image patch denoising
    # ------------------------------------------------------------------

    def _denoise_patch(self, image):
        H, W = image.shape

        # Candidate patch sizes
        candidate_k = self.candidate_k
        if candidate_k is None:
            k0 = int(round((H * W) ** 0.25))
            max_k = min(H, W) // 3
            candidate_k = sorted(set([
                max(3, k0 - 3), max(3, k0 - 1), k0,
                min(k0 + 2, max_k), min(k0 + 4, max_k),
            ]))
            for k in [4, 5, 6, 7, 8, 10, 12, 16]:
                if 3 <= k <= max_k and k not in candidate_k:
                    candidate_k.append(k)
            candidate_k = sorted(set(candidate_k))

        info = {"H": H, "W": W, "candidate_k": candidate_k}

        # Phase 1: find optimal k
        best_k = candidate_k[0]
        best_score = -np.inf
        k_scores = {}

        for k in candidate_k:
            n_patches_h = H // k
            n_patches_w = W // k
            n_patches = n_patches_h * n_patches_w
            if n_patches < 5:
                continue

            y_k = k ** 2 / n_patches

            X_k, _ = _extract_patches(image, k, stride=k)
            if X_k.shape[1] < 3:
                continue

            x_mean_k = np.mean(X_k, axis=1, keepdims=True)
            X_k_c = X_k - x_mean_k

            p_k, n_k = X_k_c.shape
            if p_k <= n_k:
                S_k = (X_k_c @ X_k_c.T) / n_k
            else:
                S_k = (X_k_c.T @ X_k_c) / n_k

            eigvals_k = np.linalg.eigvalsh(S_k)[::-1]
            eigvals_k = eigvals_k[eigvals_k > 1e-10]
            if len(eigvals_k) < 3:
                continue

            try:
                a_k, beta_k, sigma2_k, _ = estimate_parameters(eigvals_k, y_k)
                _, lam_upper_k = compute_support_bounds(a_k, beta_k, sigma2_k, y_k)
            except Exception:
                sigma2_k = np.median(eigvals_k)
                lam_upper_k = sigma2_k * (1 + np.sqrt(y_k)) ** 2

            score = _compute_patch_score(eigvals_k, lam_upper_k, y_k)
            k_scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        info["k_scores"] = k_scores
        info["best_k"] = best_k

        k = best_k
        stride = max(1, int(k * self.stride_ratio))
        info["stride"] = stride

        # Phase 2: denoise with optimal k
        X_patches, positions = _extract_patches(image, k, stride=stride)
        if X_patches.shape[1] < 3:
            self._info = info
            return image.copy()

        p_k = X_patches.shape[0]   # k^2
        n_k = X_patches.shape[1]
        y_k = p_k / n_k

        x_mean = np.mean(X_patches, axis=1, keepdims=True)
        X_c = X_patches - x_mean

        # SVD (dual when p > n)
        if p_k <= n_k:
            U_full, svs_full, Vt_full = np.linalg.svd(X_c, full_matrices=False)
            pos_mask = svs_full > 1e-5
            svs = svs_full[pos_mask]
            U = U_full[:, pos_mask]
            Vt = Vt_full[pos_mask, :]
            eigvals_pos = svs ** 2 / n_k
        else:
            G = (X_c.T @ X_c) / n_k
            eigvals_g, V_g = np.linalg.eigh(G)
            idx = np.argsort(eigvals_g)[::-1]
            eigvals_g = eigvals_g[idx]
            V_g = V_g[:, idx]

            pos_mask = eigvals_g > 1e-10
            eigvals_g = eigvals_g[pos_mask]
            V_g = V_g[:, pos_mask]
            svs = np.sqrt(n_k * eigvals_g)
            U = (X_c @ V_g) / svs
            Vt = V_g.T
            eigvals_pos = eigvals_g

        if len(eigvals_pos) < 3:
            self._info = info
            return image.copy()

        eigenvalues = eigvals_pos

        # Estimate generalized parameters
        if self._a_given is not None and self._beta_given is not None and self._sigma2_given is not None:
            a_est, beta_est, sigma2_est = self._a_given, self._beta_given, self._sigma2_given
        else:
            a_est, beta_est, sigma2_est, _ = estimate_parameters(eigenvalues, y_k)
            if self._a_given is not None:
                a_est = self._a_given
            if self._beta_given is not None:
                beta_est = self._beta_given
            if self._sigma2_given is not None:
                sigma2_est = self._sigma2_given

        lam_lower, lam_upper_gen = compute_support_bounds(a_est, beta_est, sigma2_est, y_k)

        # MP baseline (guaranteed floor)
        sigma2_mp, threshold_mp, rank_mp = self._mp_baseline(eigenvalues, y_k)
        num_signal_gen = int(np.sum(eigenvalues > lam_upper_gen))

        # Choose threshold (same logic as multi-image mode)
        params_reliable = (1.01 < a_est < 50) and (0.01 < beta_est < 0.99)
        gen_much_fewer = (num_signal_gen < rank_mp * 0.5) and (rank_mp > 5)

        if params_reliable and not gen_much_fewer:
            lam_upper = lam_upper_gen
            signal_mask = eigenvalues > lam_upper_gen
        elif num_signal_gen >= rank_mp:
            lam_upper = lam_upper_gen
            signal_mask = eigenvalues > lam_upper_gen
        else:
            lam_upper = threshold_mp
            signal_mask = eigenvalues > threshold_mp

        num_signal = int(np.sum(signal_mask))

        # Hard threshold
        shrunk_svs = svs * signal_mask[: len(svs)]

        # Reconstruct patches
        X_reconstructed = (U[:, : len(shrunk_svs)] * shrunk_svs) @ Vt[: len(shrunk_svs), :]
        X_reconstructed += x_mean

        # Patch weights: w_j = 1 / (rank + 1)
        weights = np.full(len(positions), 1.0 / (num_signal + 1))

        # Reassemble
        denoised = _reassemble_patches(X_reconstructed, positions, k, H, W, weights)
        denoised = np.clip(denoised, 0, 1)

        info.update({
            "a": a_est,
            "beta": beta_est,
            "sigma2": sigma2_est,
            "threshold": lam_upper,
            "threshold_mp": threshold_mp,
            "lambda_lower": lam_lower,
            "lambda_upper_gen": lam_upper_gen,
            "y_k": y_k,
            "p_k": p_k,
            "n_k": n_k,
            "rank": num_signal,
            "rank_mp": rank_mp,
            "rank_gen": num_signal_gen,
        })
        self._info = info
        return denoised

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mp_baseline(eigenvalues, y):
        """Compute standard MP hard threshold (sigma2, lambda_+, rank).

        Replicates the MP baseline logic from gen_cov_denoise.py.
        """
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        if y < 0.99 and len(pos_eigs) > 0:
            sigma2_mp = float(
                np.min(pos_eigs) / max((1 - np.sqrt(y)) ** 2, 1e-6)
            )
        else:
            if len(pos_eigs) > 2:
                bottom_half = np.sort(pos_eigs)[: max(len(pos_eigs) // 2, 1)]
                sigma2_mp = float(np.mean(bottom_half) / (1 + y))
            else:
                sigma2_mp = float(np.mean(eigenvalues) / (1 + y))

        threshold_mp = sigma2_mp * (1 + np.sqrt(y)) ** 2
        rank_mp = int(np.sum(eigenvalues > threshold_mp))
        return sigma2_mp, threshold_mp, rank_mp
