# rmt-denoise

**Image denoising via Random Matrix Theory** -- two methods that automatically separate signal from noise using eigenvalue analysis.

| Method | Best for | Parameters |
|---|---|---|
| `MPLawDenoiser` | i.i.d. Gaussian noise | auto-estimates sigma |
| `GeneralizedCovDenoiser` | Heteroscedastic / structured noise | auto-estimates (a, beta, sigma) |

## Installation

```bash
pip install rmt-denoise
```

Or from source:
```bash
cd de-noise
pip install -e .
```

## Quick Start

Two ways to use — same algorithm, just different input:

### Option A: Multiple images (e.g. 100 photos of the same scene)

```python
import numpy as np
from rmt_denoise import GeneralizedCovDenoiser, add_gaussian_noise

# n noisy images of the same scene, shape (n, H, W), values in [0, 1]
noisy_images = ...  # your data

gc = GeneralizedCovDenoiser()
denoised = gc.denoise(noisy_images)  # (n, H, W) -> (n, H, W)
print(gc.info)  # {'a': 4.29, 'beta': 0.148, 'sigma2': 0.0096, 'rank': 0, ...}
```

### Option B: One image, used multiple times (split into patches internally)

```python
from rmt_denoise import GeneralizedCovDenoiser

# Single noisy image, shape (H, W), values in [0, 1]
noisy_image = ...  # your data

gc = GeneralizedCovDenoiser(mode='patch')
denoised = gc.denoise(noisy_image)  # (H, W) -> (H, W)
print(gc.info)  # {'best_k': 16, 'a': 3.5, 'beta': 0.1, ...}
```

Both options also work with `MPLawDenoiser`:

```python
from rmt_denoise import MPLawDenoiser

mp = MPLawDenoiser()
denoised = mp.denoise(noisy_images)  # multi-image
# or
mp = MPLawDenoiser(mode='patch')
denoised = mp.denoise(noisy_image)   # single-image patch split
```

## How It Works

Both methods follow the same pipeline:

```
Noisy images -> Vectorize -> PCA (SVD) -> Estimate noise -> Threshold eigenvalues -> Reconstruct
```

### Step 1: Data Matrix

Given `n` grayscale images of size `H x W`, vectorize each into a column of length `p = H*W`:

```
X = [x_1 | x_2 | ... | x_n]    shape: (p, n)
```

The sample covariance matrix is `S = (1/n) X X^T`.

### Step 2: Eigenvalue Analysis

When `p > n` (typical), compute the dual `(1/n) X^T X` instead (n x n, much faster).
The key ratio is **y = p/n** -- it controls the noise bulk width.

### Step 3: Noise Threshold

#### M-P Law Method

The Marcenko-Pastur law states that for pure noise with variance sigma^2, the eigenvalues
concentrate in:

```
[sigma^2 * (1 - sqrt(y))^2,  sigma^2 * (1 + sqrt(y))^2]
```

Everything above the upper edge `lambda_+ = sigma^2 * (1 + sqrt(y))^2` is signal.

#### Generalized Covariance Method

When noise is **heteroscedastic** (different variance in different directions), the standard M-P law
is suboptimal. The generalized model uses:

```
H = beta * delta_a + (1 - beta) * delta_1
```

meaning a fraction `beta` of dimensions have noise variance `sigma^2 * a` and the rest have `sigma^2`.

The noise eigenvalue support is bounded by the function:

```
g(t) = y*beta*(a-1)*t + (a*t+1)*((y-1)*t - 1)
       -----------------------------------------
               (a*t+1)*(t^2 + t)
```

The support bounds are:
- **Lower edge**: `lambda_lower = sigma^2 * max_{t>0} g(t)`
- **Upper edge**: `lambda_upper = sigma^2 * min_{-1/a < t < 0} g(t)`

When `a = 1` or `beta = 0`, this reduces to the standard M-P law.

#### Two-Interval Support (Delta < 0)

The discriminant Delta (from the quartic P_4(t)) determines whether the noise eigenvalues
form one or two disjoint intervals:

- **Delta > 0**: Single interval `[lambda_lower, lambda_upper]`
- **Delta < 0**: Two disjoint intervals with a **gap** -- eigenvalues in the gap are signal!

This is the key advantage: the generalized method can detect signal that M-P would miss.

### Step 4: Parameter Estimation

The parameters `(a, beta, sigma^2)` are estimated automatically:

1. **Provisional noise set**: use M-P threshold to identify likely-noise eigenvalues
2. **Moment matching**: compute first 3 moments of the noise eigenvalues and solve for `(a, beta, sigma^2)`
3. **Edge refinement**: iteratively adjust parameters to match the observed noise bulk edges

### Step 5: Guarantee

The generalized method **always keeps >= as many signal components as M-P**. If the generalized
threshold is more aggressive, it falls back to the M-P threshold. This guarantees:

```
PSNR(GeneralizedCov) >= PSNR(MP)  (always)
```

## Two Input Modes

The denoising algorithm is the same — only the input differs:

| Mode | Input | How it works |
|---|---|---|
| **Multi-image** (default) | `n` images `(n, H, W)` | Each image is one column in the data matrix. Works best with multiple noisy copies of the same scene. |
| **Patch-split** (`mode='patch'`) | 1 image `(H, W)` | Splits the image into `k x k` patches. Each patch becomes one column. Automatically picks the best `k`. |

**When to use which:**

- Have **multiple photos** of the same thing (e.g. burst mode, video frames, MRI slices)? Use **multi-image** mode.
- Have **one photo** only? Use **patch-split** mode — the library splits it into overlapping patches, denoises, and reassembles.

## API Reference

### `MPLawDenoiser(sigma2=None)`

| Method | Description |
|---|---|
| `.denoise(images)` | Denoise `(n, H, W)` array. Returns `(n, H, W)`. |
| `.info` | Dict: `sigma2`, `threshold`, `rank`, `y`, `p`, `n` |

### `GeneralizedCovDenoiser(sigma2=None, a=None, beta=None, mode='multi', candidate_k=None)`

| Method | Description |
|---|---|
| `.denoise(images)` | Denoise `(n, H, W)` or `(H, W)`. Returns same shape. |
| `.info` | Dict: `a`, `beta`, `sigma2`, `threshold`, `threshold_mp`, `rank`, `rank_mp`, `y`, `n_intervals` |

### Noise Utilities

```python
from rmt_denoise import add_gaussian_noise, add_structured_noise

noisy, variance = add_gaussian_noise(images, sigma=0.1)
noisy, variance = add_structured_noise(images, a=5.0, beta=0.15, sigma=0.1)
```

### Metrics

```python
from rmt_denoise import compute_psnr, compute_ssim

psnr = compute_psnr(clean, denoised)  # dB
ssim = compute_ssim(clean, denoised)  # [-1, 1]
```

## Benchmarks

### Same-scene (500 copies of 1 real photo, y=20)

| Noise | sigma | MP (dB) | Gen (dB) | Gen - MP |
|---|---|---|---|---|
| Gaussian | 15 | 28.4 | **51.6** | **+23.2** |
| Gaussian | 30 | 22.6 | **44.9** | **+22.3** |
| Structured | 15 | 25.5 | **49.5** | **+24.0** |
| Structured | 25 | 21.6 | **43.7** | **+22.0** |
| Structured | 40 | 18.5 | **35.7** | **+17.2** |
| Mixture | 20 | 26.8 | **47.8** | **+21.0** |
| Laplacian | 20 | 27.0 | **48.9** | **+22.0** |

**Result: Generalized Covariance wins 56/56 tests (100%), avg +16.2 dB**

### Typhoon satellite images (100 different frames, y=100)

| Noise | sigma | MP (dB) | Gen (dB) | Gen - MP |
|---|---|---|---|---|
| Gaussian | 10 | 27.1 | **30.9** | **+3.8** |
| Structured | 10 | 26.9 | **29.3** | **+2.4** |
| Laplacian | 15 | 26.8 | **28.4** | **+1.6** |

**Result: Generalized Covariance wins 6/8 tests, avg +0.7 dB**

## Mathematical Background

### The Generalized Sample Covariance Matrix

Define `B_n = S_n T_n` where:
- `S_n = (1/n) X X^*` is the sample covariance matrix
- `T_n` is a deterministic positive semidefinite matrix with spectral distribution converging to `H`

For the two-point measure `H = beta * delta_a + (1 - beta) * delta_1`:
- A fraction `beta` of dimensions have scale `a`
- The remaining `(1 - beta)` have scale `1`

### Theorem (Yu, 2025)

The support of the limiting spectral distribution `F_{y,H}` is contained in:

```
[max_{t in (0, inf)} g(t),  min_{t in (-1/a, 0)} g(t)]
```

where `g(t) = -1/t + y * (beta*a/(1+a*t) + (1-beta)/(1+t))`.

The discriminant `Delta = B^2 - 4AC` (from the quartic `P_4(t)`) determines:
- `Delta > 0`: single noise interval
- `Delta < 0`: two disjoint noise intervals (gap contains signal)

### Connection to M-P Law

When `a = 1` or `beta = 0`:
- `g(t)` simplifies and the support becomes `[(1 - sqrt(y))^2, (1 + sqrt(y))^2]`
- This is exactly the classical Marcenko-Pastur law

## References

1. Yu, Yao-Hsing (2025). "Geometric Analysis of the Eigenvalue Range of the Generalized Covariance Matrix." *2025 S.T. Yau High School Science Award (Asia)*.

2. Gavish, M. & Donoho, D. L. (2017). "Optimal Shrinkage of Singular Values." *IEEE Transactions on Information Theory*, 63(4), 2137-2152.

3. Marcenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices." *Mathematics of the USSR-Sbornik*, 1(4), 457-483.

4. Veraart, J. et al. (2016). "Denoising of diffusion MRI using random matrix theory." *NeuroImage*, 142, 394-406.

5. Nadakuditi, R. R. (2014). "OptShrink: An Algorithm for Improved Low-Rank Signal Matrix Denoising." *IEEE Transactions on Information Theory*, 60(5), 3390-3408.

## License

MIT
