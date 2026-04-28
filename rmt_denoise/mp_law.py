"""
Marcenko-Pastur law image denoiser.

Estimates noise variance sigma2, computes the MP upper edge
lambda_+ = sigma2 * (1 + sqrt(y))^2, keeps eigenvalues above
the threshold, and reconstructs via PCA.

Based on the mp_hard branch of denoise_workflow_a in gen_cov_denoise.py.
"""

import numpy as np

from .core import images_to_matrix, matrix_to_images
from .estimators import estimate_sigma2_iterative


class MPLawDenoiser:
    """Marcenko-Pastur law image denoiser.

    Estimates noise variance sigma2, computes threshold
    lambda_+ = sigma2 * (1 + sqrt(y))^2, keeps eigenvalues above
    threshold, reconstructs via PCA.

    Parameters
    ----------
    sigma2 : float or None
        If given, use this noise variance directly.
        If None, auto-estimate from the eigenvalue spectrum.
    """

    def __init__(self, sigma2=None):
        self._sigma2_given = sigma2
        self._info = {}

    def denoise(self, images):
        """Denoise images via Marcenko-Pastur hard thresholding.

        Parameters
        ----------
        images : np.ndarray, shape (n, H, W)
            Stack of noisy grayscale images with values in [0, 1].

        Returns
        -------
        denoised : np.ndarray, shape (n, H, W)
            Denoised images clipped to [0, 1].
        """
        n_images, H, W = images.shape
        p = H * W
        n = n_images
        y = p / n

        # ------------------------------------------------------------------
        # Phase 1: Data preparation
        # ------------------------------------------------------------------
        X = images_to_matrix(images)          # (p, n)
        x_mean = np.mean(X, axis=1, keepdims=True)
        X_centered = X - x_mean

        # ------------------------------------------------------------------
        # Phase 2: Eigendecomposition (dual formulation when p > n)
        # ------------------------------------------------------------------
        if p > n:
            # Use n x n Gram matrix: G = (1/n) X_c^T X_c
            G = (X_centered.T @ X_centered) / n
            eigvals, V = np.linalg.eigh(G)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            V = V[:, idx]

            # Keep only positive eigenvalues
            pos_mask = eigvals > 1e-10
            eigvals_pos = eigvals[pos_mask]
            V_pos = V[:, pos_mask]
            svs = np.sqrt(n * eigvals_pos)

            # Recover p-dimensional eigenvectors: U = X_c V / s
            U = (X_centered @ V_pos) / svs
        else:
            # Direct SVD
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

        # ------------------------------------------------------------------
        # Phase 3: Estimate sigma2 and compute MP threshold
        # ------------------------------------------------------------------
        if self._sigma2_given is not None:
            sigma2_est = self._sigma2_given
        else:
            # Robust estimation identical to gen_cov_denoise.py mp_hard branch
            pos_eigs = eigenvalues[eigenvalues > 1e-10]
            if y < 0.99 and len(pos_eigs) > 0:
                sigma2_est = float(
                    np.min(pos_eigs) / max((1 - np.sqrt(y)) ** 2, 1e-6)
                )
            else:
                if len(pos_eigs) > 2:
                    bottom_half = np.sort(pos_eigs)[: max(len(pos_eigs) // 2, 1)]
                    sigma2_est = float(np.mean(bottom_half) / (1 + y))
                else:
                    sigma2_est = float(np.mean(eigenvalues) / (1 + y))

        threshold = sigma2_est * (1 + np.sqrt(y)) ** 2
        signal_mask = eigenvalues > threshold
        rank = int(np.sum(signal_mask))

        # Hard threshold singular values
        shrunk_svs = svs * signal_mask[: len(svs)]

        # ------------------------------------------------------------------
        # Phase 4: Reconstruction
        # ------------------------------------------------------------------
        X_reconstructed = (U * shrunk_svs) @ V_pos.T
        X_reconstructed += x_mean

        denoised = matrix_to_images(X_reconstructed, H, W)
        denoised = np.clip(denoised, 0, 1)

        # Store info
        self._info = {
            "sigma2": sigma2_est,
            "threshold": threshold,
            "rank": rank,
            "y": y,
            "p": p,
            "n": n,
        }

        return denoised

    @property
    def info(self):
        """Dict with estimation diagnostics.

        Keys: sigma2, threshold, rank, y, p, n.
        """
        return self._info
