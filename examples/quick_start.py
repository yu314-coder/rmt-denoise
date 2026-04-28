"""
Quick start example for de-noise library.
Demonstrates both M-P Law and Generalized Covariance denoising.
"""
import sys
import numpy as np
from pathlib import Path

# Add parent to path so we can import denoise
sys.path.insert(0, str(Path(__file__).parent.parent))

from rmt_denoise import (
    MPLawDenoiser,
    GeneralizedCovDenoiser,
    add_gaussian_noise,
    add_structured_noise,
    compute_psnr,
    compute_ssim,
)


def main():
    print("=" * 70)
    print("de-noise: Quick Start Example")
    print("=" * 70)

    # --- Create test data ---
    np.random.seed(42)
    n, H, W = 200, 50, 50
    print(f"\nGenerating {n} clean images ({H}x{W}), y = {H*W}/{n} = {H*W/n}")

    # Use a simple pattern as "clean" image, replicated n times
    Y, X = np.mgrid[:H, :W]
    base = np.clip(1.0 - np.sqrt((X - 25)**2 + (Y - 25)**2) / 30, 0, 1) * 0.6
    base += 0.15 * np.sin(2 * np.pi * X / 15)
    base = np.clip(base, 0, 1)
    clean_images = np.stack([base] * n)

    # --- Test 1: Gaussian noise ---
    print("\n--- Test 1: Gaussian Noise (sigma=0.1) ---")
    noisy, true_var = add_gaussian_noise(clean_images, sigma=0.1)
    noisy_psnr = compute_psnr(clean_images[0], noisy[0])
    print(f"  Noisy PSNR:  {noisy_psnr:.1f} dB")

    # M-P Law
    mp = MPLawDenoiser()
    den_mp = mp.denoise(noisy)
    mp_psnr = compute_psnr(clean_images[0], den_mp[0])
    print(f"  M-P Law:     {mp_psnr:.1f} dB  (rank={mp.info['rank']}, sigma2={mp.info['sigma2']:.4f})")

    # Generalized Covariance
    gc = GeneralizedCovDenoiser()
    den_gc = gc.denoise(noisy)
    gc_psnr = compute_psnr(clean_images[0], den_gc[0])
    print(f"  Gen. Cov.:   {gc_psnr:.1f} dB  (rank={gc.info['rank']}, a={gc.info.get('a', '?')}, beta={gc.info.get('beta', '?')})")
    print(f"  Gen - MP:    {gc_psnr - mp_psnr:+.1f} dB")

    # --- Test 2: Structured (heteroscedastic) noise ---
    print("\n--- Test 2: Structured Noise (a=5, beta=0.15, sigma=0.1) ---")
    noisy2, true_var2 = add_structured_noise(clean_images, a=5.0, beta=0.15, sigma=0.1)
    noisy2_psnr = compute_psnr(clean_images[0], noisy2[0])
    print(f"  Noisy PSNR:  {noisy2_psnr:.1f} dB")

    mp2 = MPLawDenoiser()
    den_mp2 = mp2.denoise(noisy2)
    mp2_psnr = compute_psnr(clean_images[0], den_mp2[0])
    print(f"  M-P Law:     {mp2_psnr:.1f} dB  (rank={mp2.info['rank']})")

    gc2 = GeneralizedCovDenoiser()
    den_gc2 = gc2.denoise(noisy2)
    gc2_psnr = compute_psnr(clean_images[0], den_gc2[0])
    print(f"  Gen. Cov.:   {gc2_psnr:.1f} dB  (rank={gc2.info['rank']}, a={gc2.info.get('a', '?'):.2f}, beta={gc2.info.get('beta', '?'):.3f})")
    print(f"  Gen - MP:    {gc2_psnr - mp2_psnr:+.1f} dB")

    # --- Test 3: Single-image patch denoising ---
    print("\n--- Test 3: Single-Image Patch Denoising ---")
    single_noisy = noisy[0]  # just one image
    gc3 = GeneralizedCovDenoiser(mode='patch')
    den_patch = gc3.denoise(single_noisy)
    patch_psnr = compute_psnr(clean_images[0], den_patch)
    print(f"  Patch denoise: {patch_psnr:.1f} dB  (k={gc3.info.get('best_k', '?')})")

    print("\n" + "=" * 70)
    print("Done! Generalized Covariance should always be >= M-P Law.")
    print("=" * 70)


if __name__ == "__main__":
    main()
