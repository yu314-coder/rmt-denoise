"""
Generalized Covariance Matrix denoiser — oracle best (a, β) route.

Model: B_n = S_n T_n  with  H = β·δ_a + (1−β)·δ_1.

Workflow:

    1. Take a stack of n noisy images and a clean reference for ONE of them.
    2. Centre: X̃ = X − X̄.
    3. SVD once.
    4. SciPy `differential_evolution` over (log a, β) chooses the (a, β) whose
       gen-cov acceptance test selects the rank that maximises PSNR of the
       reconstructed test column vs the clean reference. The PSNR objective
       and the final reconstruction both apply:

           projection (rank r̂) → centring re-add (X̄)
           → clip to [0, 1] → T(a, β)  (if apply_t)
           → color_resize           (if color_resize)

    5. Return the full (n, H, W) stack reconstructed at the chosen rank with
       the same post-processing applied to every column.

This is "oracle" mode: it requires the clean reference of the test image to
score (a, β). Use it for synthetic benchmarks, MRI phantoms, or any setup
where a ground-truth image is available.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np

from .core import (
    compute_G_minus,
    compute_G_plus,
    images_to_matrix,
    matrix_to_images,
)
from .io import load_folder, split_train_test


# ============================================================================
# Main class
# ============================================================================


class GeneralizedCovDenoiser:
    """Generalized Covariance Matrix denoiser — best (a, β) by oracle search.

    Model: B_n = S_n T_n  with  H = β·δ_a + (1−β)·δ_1.

    The pipeline always:

      * centres the data (X̃ = X − X̄),
      * applies T(a, β) = diag(√a, ..., √a, 1, ..., 1) with ⌊p·β⌋ √a entries,
      * applies the color-resize ``y = (x − min(x)) / max(x_before_subtract)``.

    Parameters
    ----------
    a_bracket : (float, float), default (0.01, 1.0)
        Search range for the population mass location ``a`` (linear scale).
    beta_bracket : (float, float), default (0.01, 0.99)
        Search range for the mixing weight ``β``.
    seed : int, default 42
        Seed for the differential-evolution Sobol initialisation.
    de_kwargs : dict or None
        Override / augment the kwargs passed to
        ``scipy.optimize.differential_evolution``.
    """

    def __init__(
        self,
        a_bracket: Tuple[float, float] = (0.01, 1.0),
        beta_bracket: Tuple[float, float] = (0.01, 0.99),
        seed: int = 42,
        de_kwargs: Optional[dict] = None,
    ):
        self.a_bracket = tuple(a_bracket)
        self.beta_bracket = tuple(beta_bracket)
        self.seed = int(seed)
        self.de_kwargs = dict(de_kwargs or {})
        self._info: dict = {}
        self.a: float = float('nan')
        self.beta: float = float('nan')
        self.rank: int = 0
        self.sigma2: float = 0.0
        self.psnr_test: float = float('nan')

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_T_diag(img2d: np.ndarray, a: float, beta: float) -> np.ndarray:
        """Diagonal T: first ⌊p·β⌋ entries = √a, rest = 1, then clip [0, 1]."""
        a = float(max(a, 0.0)); beta = float(min(max(beta, 0.0), 1.0))
        flat = np.asarray(img2d, dtype=np.float64).ravel().copy()
        p = flat.size
        k = int(round(p * beta))
        if k > 0:
            flat[:k] = flat[:k] * np.sqrt(a)
        return np.clip(flat.reshape(img2d.shape), 0.0, 1.0)

    @staticmethod
    def _color_resize(img2d: np.ndarray) -> np.ndarray:
        """y = (x − min(x)) / max(x_before_subtract), clipped to [0, 1]."""
        x = np.asarray(img2d, dtype=np.float64)
        x_max = float(x.max()) if x.size else 0.0
        if x_max <= 0.0:
            return np.clip(x, 0.0, 1.0)
        y = (x - float(x.min())) / x_max
        return np.clip(y, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def info(self) -> dict:
        """Diagnostics from the most recent denoise call.

        Keys: ``a``, ``beta``, ``sigma2``, ``rank``, ``psnr_test``,
        ``test_index``, ``n_evals``, ``y``, ``p``, ``n``, ``apply_t``,
        ``color_resize``, ``method``.
        """
        return self._info

    def denoise(
        self,
        images: np.ndarray,
        clean: np.ndarray,
        test_index: int = -1,
    ) -> np.ndarray:
        """Run the oracle best (a, β) gen-cov denoiser.

        Parameters
        ----------
        images : np.ndarray, shape (n, H, W)
            Noisy training stack. The column at ``test_index`` is the test
            image whose denoised PSNR drives the (a, β) search.
        clean : np.ndarray, shape (H, W)
            Clean ground-truth reference for the test column.
        test_index : int, default -1
            Which column of *images* is the oracle's test image.

        Returns
        -------
        denoised : np.ndarray, shape (n, H, W)
            Reconstructed stack with T(a, β) and color-resize applied.
        """
        from scipy.optimize import differential_evolution

        if images.ndim != 3:
            raise ValueError(f"images must be (n, H, W); got {images.shape}")
        n_images, H, W = images.shape
        p = H * W
        n = n_images
        y = p / n
        if test_index < 0:
            test_index += n
        if not 0 <= test_index < n:
            raise IndexError(f"test_index out of range: {test_index}")

        clean_2d = np.asarray(clean, dtype=np.float64)
        if clean_2d.shape != (H, W):
            raise ValueError(
                f"clean shape {clean_2d.shape} does not match images {(H, W)}"
            )

        # Centre and SVD. Use np.linalg.svd directly (no dual-eigh path) for
        # bit-for-bit consistency with the in-app best_a_beta CPU path.
        X = images_to_matrix(images)               # (p, n)
        x_mean = np.mean(X, axis=1, keepdims=True)
        X_centered = X - x_mean
        U_full, svs_full, Vt_full = np.linalg.svd(X_centered, full_matrices=False)
        pos = svs_full > 1e-5
        U = U_full[:, pos]
        svs = svs_full[pos]
        Vt = Vt_full[pos, :]
        lam = (svs ** 2) / n

        m = len(lam)
        kmax = m - 1
        csum = np.cumsum(lam)
        tot = float(csum[-1])
        lam_end = float(lam[-1])

        x_test_centered = X_centered[:, test_index]
        x_mean_flat = x_mean.ravel()
        dv_cache: dict = {}

        def _proj(r: int) -> np.ndarray:
            if r in dv_cache:
                return dv_cache[r]
            if r <= 0 or r >= m:
                dv = np.zeros_like(x_test_centered)
            else:
                Ur = U[:, :r]
                dv = Ur @ (Ur.T @ x_test_centered)
            dv_cache[r] = dv
            return dv

        def _r_sigma2(a_j: float, b_j: float) -> Tuple[int, float]:
            for k in range(0, kmax + 1):
                Lk = m - k
                if Lk < 2:
                    return m, 0.0
                gamma_k = Lk / float(n)
                g_lo = compute_G_minus(a_j, b_j, gamma_k)
                g_hi = compute_G_plus(a_j, b_j, gamma_k)
                W_k = max(float(g_hi) - float(g_lo), 1e-15)
                tail_sum = tot - (csum[k - 1] if k > 0 else 0.0)
                lam_k1 = float(lam[k])
                if lam_k1 <= lam_end:
                    return m, 0.0
                sigma2_k = max((lam_k1 - lam_end) / W_k, 1e-30)
                if tail_sum >= Lk * sigma2_k:
                    return k, sigma2_k
            return m, 0.0

        a_lo, a_hi = float(self.a_bracket[0]), float(self.a_bracket[1])
        THETA_LO, THETA_HI = float(np.log(a_lo)), float(np.log(a_hi))
        BETA_LO, BETA_HI = float(self.beta_bracket[0]), float(self.beta_bracket[1])

        best = {'psnr': -1e30, 'a': float('nan'), 'b': float('nan'),
                'r': 0, 'sigma2': 0.0}
        n_evals = [0]

        def _neg_psnr(theta: np.ndarray) -> float:
            n_evals[0] += 1
            t_a = float(np.clip(theta[0], THETA_LO, THETA_HI))
            b_j = float(np.clip(theta[1], BETA_LO, BETA_HI))
            a_j = float(np.exp(t_a))
            chosen_k, sig2 = _r_sigma2(a_j, b_j)
            r_a = chosen_k if chosen_k < m else 0
            dv = _proj(r_a)
            dv_full = dv + x_mean_flat
            img = np.clip(dv_full.reshape(H, W), 0.0, 1.0)
            img = self._apply_T_diag(img, a_j, b_j)
            img = self._color_resize(img)
            mse = float(np.mean((clean_2d - img) ** 2))
            psnr = (10.0 * np.log10(1.0 / mse)) if mse > 0 else 99.0
            if psnr > best['psnr']:
                best.update(psnr=psnr, a=a_j, b=b_j, r=r_a, sigma2=sig2)
            return -psnr

        de_args = dict(
            bounds=[(THETA_LO, THETA_HI), (BETA_LO, BETA_HI)],
            strategy='best1bin', popsize=20, mutation=(0.5, 1.5),
            recombination=0.7, tol=1e-4, maxiter=80, seed=self.seed,
            polish=True, init='sobol',
        )
        de_args.update(self.de_kwargs)
        try:
            differential_evolution(_neg_psnr, **de_args)
        except Exception:
            pass

        a_hat = best['a'] if np.isfinite(best['a']) else 1.0
        beta_hat = best['b'] if np.isfinite(best['b']) else 0.99
        rank = int(best['r'])
        sigma2_hat = float(best['sigma2'])

        if rank > 0:
            shrunk = np.zeros_like(svs)
            keep = min(rank, len(svs))
            shrunk[:keep] = svs[:keep]
            X_rec = (U * shrunk) @ Vt
        else:
            X_rec = np.zeros_like(X_centered)
        X_rec = X_rec + x_mean
        denoised = matrix_to_images(X_rec, H, W)
        denoised = np.clip(denoised, 0.0, 1.0)
        for i in range(denoised.shape[0]):
            denoised[i] = self._apply_T_diag(denoised[i], a_hat, beta_hat)
            denoised[i] = self._color_resize(denoised[i])

        self.a = float(a_hat)
        self.beta = float(beta_hat)
        self.rank = int(rank)
        self.sigma2 = float(sigma2_hat)
        self.psnr_test = float(best['psnr'])
        self._info = {
            "a": self.a,
            "beta": self.beta,
            "sigma2": self.sigma2,
            "rank": self.rank,
            "psnr_test": self.psnr_test,
            "test_index": int(test_index),
            "n_evals": int(n_evals[0]),
            "y": y, "p": p, "n": n,
            "method": "best_a_beta_oracle",
        }
        return denoised

    # ------------------------------------------------------------------
    # Folder convenience
    # ------------------------------------------------------------------

    def denoise_folder(
        self,
        folder: str,
        n_train: int,
        test: Union[int, str],
        size: Optional[Tuple[int, int]] = None,
        noise_fn=None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """End-to-end run on a folder of images.

        Loads every image in *folder*, picks *n_train* training images plus the
        single image identified by *test* as the oracle reference, optionally
        injects noise via *noise_fn*, then runs the best (a, β) oracle search.

        Parameters
        ----------
        folder : str
            Path to a folder of images.
        n_train : int
            Number of training images to pick (0 = all but the test image).
        test : int or str
            Test image — integer index or file name.
        size : (H, W) or None
            Resize every image to this exact size. If None, all images must
            already have the same shape.
        noise_fn : callable or None
            Function ``noise_fn(images: (n, H, W)) -> (n, H, W)`` applied to
            the train+test stack before denoising. If None, no noise is added
            (use this when the images already contain real noise).
        seed : int
            Seed used to subsample the training set.

        Returns
        -------
        denoised : np.ndarray, shape (n_train+1, H, W)
            Denoised stack.
        clean_test : np.ndarray, shape (H, W)
            The clean test image used as the oracle reference.
        noisy_test : np.ndarray, shape (H, W)
            The test image AFTER noise injection (== clean_test if noise_fn=None).
        names : list[str]
            File names of the loaded folder, aligned with the load order.
        """
        images, names = load_folder(folder, size=size)
        train_imgs, clean_test, _ = split_train_test(
            images, names, n_train=n_train, test=test, seed=seed
        )
        # Append the clean test image as the LAST column, then add noise jointly.
        stack_clean = np.concatenate([train_imgs, clean_test[None, :, :]], axis=0)
        if noise_fn is not None:
            stack_noisy = noise_fn(stack_clean)
        else:
            stack_noisy = stack_clean
        denoised = self.denoise(stack_noisy, clean=clean_test, test_index=-1)
        noisy_test = stack_noisy[-1]
        return denoised, clean_test, noisy_test, names
