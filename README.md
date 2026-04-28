# rmt-denoise

**Image denoising via Random Matrix Theory** — automatically pick the best generalized-covariance parameters `(a, β)` for every test image and reconstruct it.

| Method | Best for | Parameters |
|---|---|---|
| `MPLawDenoiser` | i.i.d. Gaussian noise | auto-estimates σ |
| `GeneralizedCovDenoiser` | Heteroscedastic / structured noise | **oracle search** for `(a, β)` |

## Installation

```bash
pip install rmt-denoise[images]   # `images` extra pulls in Pillow for load_folder
```

Or from source:
```bash
git clone https://github.com/yu314-coder/rmt-denoise.git
cd rmt-denoise && pip install -e .[images]
```

## Quick Start

`GeneralizedCovDenoiser` runs a single workflow: pick a folder of images, choose how many to use for training, choose which one to denoise, and the denoiser jointly optimises `(a, β)` via `differential_evolution` to maximise PSNR of that test image.

```python
from rmt_denoise import GeneralizedCovDenoiser, add_gaussian_noise

gc = GeneralizedCovDenoiser()    # always centres + T(a, β) + color-resize
denoised, clean, noisy, names = gc.denoise_folder(
    folder      = "path/to/images",                 # folder of (H, W) images
    n_train     = 300,                              # 300 training images (0 = use all)
    test        = "s12_1.png",                      # one filename — or an integer index
    size        = (100, 100),                       # resize on load (optional)
    noise_fn    = lambda imgs: add_gaussian_noise(imgs, sigma=0.04),
)
print(gc.a, gc.beta, gc.rank, gc.psnr_test)
# 0.27  0.95  12  34.5
```

The chosen `(a, β)` are exposed as direct attributes (`gc.a`, `gc.beta`, `gc.rank`, `gc.sigma2`, `gc.psnr_test`); the full diagnostic dict is still in `gc.info`.

`denoised` is the full `(n_train+1, H, W)` reconstructed stack — the test image lives at index `-1`. `clean` and `noisy` are the (clean / noisy) versions of that test image, returned for convenience.

### Lower-level API (already have an in-memory stack)

```python
from rmt_denoise import GeneralizedCovDenoiser

# noisy_stack: (n, H, W)   clean_test: (H, W)
gc = GeneralizedCovDenoiser()
denoised = gc.denoise(noisy_stack, clean=clean_test, test_index=-1)
```

The test column at `test_index` drives the (a, β) search; the rest of the stack is reconstructed at the same chosen rank with the same post-processing.

### Train + target convenience

```python
# train: (n_train, H, W) noisy   test_noisy: (H, W)   test_clean: (H, W)
gc = GeneralizedCovDenoiser()
denoised_test = gc.denoise_test(train, test_noisy, test_clean)   # (H, W)
print(gc.a, gc.beta, gc.rank, gc.psnr_test)
```

### GPU acceleration

`device='auto'` (default) picks Apple MPS, then CUDA, then CPU. CPU runs
`np.linalg.svd` in float64; MPS / CUDA runs `torch.linalg.svd` in float32.
The downstream optimisation runs on CPU regardless of `device`.

```python
gc = GeneralizedCovDenoiser(device='mps')   # Apple Silicon
gc = GeneralizedCovDenoiser(device='cuda')  # NVIDIA
gc = GeneralizedCovDenoiser(device='cpu')   # force CPU (slower, float64)
```

After every `.denoise()` call the lib prints the chosen `(â, β̂, r̂, σ̂², PSNR, device)` to stdout. The same fields are available as attributes on the denoiser:

```python
gc.a, gc.beta, gc.rank, gc.sigma2, gc.psnr_test
```

### MPLawDenoiser

Standard Marčenko–Pastur baseline, no oracle search:

```python
from rmt_denoise import MPLawDenoiser
mp = MPLawDenoiser()
denoised = mp.denoise(noisy_images)        # (n, H, W) -> (n, H, W)
```

## How `GeneralizedCovDenoiser` works

End-to-end:

1. **Centre** the data: `X̃ = X − X̄`.
2. **SVD once** (uses dual when `p > n`).
3. **`scipy.optimize.differential_evolution`** over `(log a, β)` with bounds `a ∈ [0.01, 1.0]`, `β ∈ [0.01, 0.99]`. Default settings: `popsize=20`, Sobol init, `seed=42`, `polish=True`.
4. **Per-candidate evaluation**: gen-cov acceptance test → rank `r̂` → cached projection → centring re-add → clip → `T(a, β)` → color-resize → PSNR vs clean.
5. **Reconstruct every column** at the best rank, applying the same `T` and color-resize to all of them.

`T(a, β)` is the diagonal matrix whose first `⌊p·β⌋` entries are `√a` and the rest are `1` (`p = H·W`); it embeds the noise-model scaling `H = β·δ_a + (1 − β)·δ_1` directly into the post-processed image.

`color_resize` rescales each image as `y = (x − min(x)) / max(x_before_subtract)` and clips to `[0, 1]`.

Both steps are part of the algorithm and are always applied — there are no flags to disable them.

## API Reference

### `GeneralizedCovDenoiser`

```python
GeneralizedCovDenoiser(
    a_bracket=(0.01, 1.0),
    beta_bracket=(0.01, 0.99),
    seed=42,
    de_kwargs=None,
    device='auto',     # 'auto' | 'cpu' | 'mps' | 'cuda'
)
```

| Method / attribute | Description |
|---|---|
| `.denoise(images, clean, test_index=-1)` | Run on a noisy `(n, H, W)` stack with a clean `(H, W)` reference for the test column. Returns the reconstructed stack. |
| `.denoise_test(train, test_noisy, test_clean)` | Single-target convenience: returns the denoised `(H, W)` test image given `(n_train, H, W)` training stack. |
| `.denoise_folder(folder, n_train, test, size=None, noise_fn=None, seed=42)` | End-to-end: load a folder, split into train + test, optionally inject noise, denoise. Returns `(denoised, clean, noisy, names)`. |
| `.a`, `.beta`, `.rank`, `.sigma2`, `.psnr_test` | Selected parameters from the most recent run. |
| `.info` | Full diagnostic dict (`a`, `beta`, `sigma2`, `rank`, `psnr_test`, `n_evals`, `y`, `p`, `n`, `device`, `method='best_a_beta_oracle'`). |

### `MPLawDenoiser(sigma2=None)`

| Method | Description |
|---|---|
| `.denoise(images)` | Denoise `(n, H, W)`. Returns `(n, H, W)`. |
| `.info` | `sigma2`, `threshold`, `rank`, `y`, `p`, `n`. |

### Folder loaders

```python
from rmt_denoise import load_folder, split_train_test

images, names = load_folder("path/", size=(H, W))            # (n, H, W) in [0, 1]
train_imgs, test_img, n_train = split_train_test(
    images, names, n_train=300, test="s12_1.png", seed=42,
)
```

### Noise utilities

```python
from rmt_denoise import (
    add_gaussian_noise, add_laplacian_noise,
    add_mixture_gaussian_noise, add_structured_noise,
)
```

### Metrics

```python
from rmt_denoise import compute_psnr, compute_ssim
psnr = compute_psnr(clean, denoised)   # dB
ssim = compute_ssim(clean, denoised)   # [-1, 1]
```

## Mathematical Background

`B_n = S_n T_n` where `T_n` has spectral distribution converging to `H = β·δ_a + (1 − β)·δ_1`. The noise eigenvalue support is bounded by

```
g(t) = -1/t + y · ( β·a/(1 + a·t) + (1 − β)/(1 + t) )
```

with bulk edges

```
λ_lower = σ² · max_{t > 0} g(t),
λ_upper = σ² · min_{−1/a < t < 0} g(t).
```

When `a = 1` or `β = 0`, this reduces to classical Marčenko–Pastur: `[(1 − √y)², (1 + √y)²]`.

The oracle workflow above selects `(a, β)` so that the corresponding acceptance-test rank gives the maximum PSNR against a clean reference, then bakes the resulting `T(a, β)` directly into the post-processed image.

## References

1. Yu, Yao-Hsing (2025). "Geometric Analysis of the Eigenvalue Range of the Generalized Covariance Matrix." *2025 S.T. Yau High School Science Award (Asia).*
2. Gavish, M. & Donoho, D. L. (2017). "Optimal Shrinkage of Singular Values." *IEEE Trans. Inf. Theory*, 63(4), 2137–2152.
3. Marčenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of random matrices." *Math. USSR-Sbornik*, 1(4), 457–483.
4. Veraart, J. et al. (2016). "Denoising of diffusion MRI using random matrix theory." *NeuroImage*, 142, 394–406.
5. Storn, R. & Price, K. (1997). "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *J. Global Optim.*, 11(4), 341–359.

## License

MIT
