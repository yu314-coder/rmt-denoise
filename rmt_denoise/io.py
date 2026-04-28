"""Folder-loading utilities for the best_a_beta oracle workflow."""

from __future__ import annotations
import os
from typing import List, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_DEFAULT_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif')


def load_folder(
    path: str,
    size: Tuple[int, int] | None = None,
    exts: Sequence[str] = _DEFAULT_EXTS,
    grayscale: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Load every image in *path* into a stack with shape (n, H, W) in [0, 1].

    Parameters
    ----------
    path : str
        Folder to scan (non-recursive).
    size : (H, W) or None
        If given, every image is resized to this exact (H, W). If None, every
        image must already share the same size.
    exts : sequence of str
        File extensions to include.
    grayscale : bool
        Convert to single-channel grayscale.

    Returns
    -------
    images : np.ndarray, shape (n, H, W), dtype float64, values in [0, 1]
    names : list[str]
        File names (basename only), aligned with `images[i]`.
    """
    if not _HAS_PIL:
        raise ImportError(
            "load_folder requires Pillow. Install with: pip install rmt-denoise[images]"
        )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"folder not found: {path}")

    names = sorted(
        f for f in os.listdir(path)
        if f.lower().endswith(tuple(e.lower() for e in exts))
    )
    if not names:
        raise ValueError(f"no images found in {path} with extensions {exts}")

    out: List[np.ndarray] = []
    for name in names:
        img = Image.open(os.path.join(path, name))
        if grayscale:
            img = img.convert('L')
        if size is not None:
            img = img.resize((size[1], size[0]))  # PIL takes (W, H)
        arr = np.asarray(img, dtype=np.float64) / 255.0
        out.append(arr)

    H = out[0].shape[0]; W = out[0].shape[1]
    for i, a in enumerate(out):
        if a.shape != (H, W):
            raise ValueError(
                f"image {names[i]} has shape {a.shape}, expected {(H, W)}; "
                "pass size=(H, W) to force resize"
            )
    return np.stack(out, axis=0), names


def split_train_test(
    images: np.ndarray,
    names: Sequence[str],
    n_train: int,
    test: int | str,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Carve a training set + a single test image out of `images`.

    Parameters
    ----------
    images : np.ndarray, shape (n_total, H, W)
    names : sequence of str
        File names aligned with images, used when *test* is a string.
    n_train : int
        Number of training images to keep. ``0`` means "all but the test image".
    test : int or str
        Test image identifier — either the integer index into *images* or the
        file name (matched against *names*).
    seed : int
        RNG seed for the random training subsample.

    Returns
    -------
    train_imgs : np.ndarray, shape (n_train, H, W)
    test_img   : np.ndarray, shape (H, W)   (clean reference)
    test_pos   : int
        Position the test image will occupy when concatenated as the LAST
        column of the training stack — i.e. index ``n_train`` in the combined
        (n_train+1, H, W) stack you would feed to `denoise(..., test_index=...)`.
    """
    n_total = images.shape[0]
    if isinstance(test, str):
        if test not in names:
            raise ValueError(f"test image '{test}' not in folder; available: {list(names)[:5]}...")
        test_idx = list(names).index(test)
    else:
        test_idx = int(test)
        if not 0 <= test_idx < n_total:
            raise ValueError(f"test index {test_idx} out of range [0, {n_total})")

    test_img = images[test_idx]
    pool = [i for i in range(n_total) if i != test_idx]
    n_train = int(n_train)
    if n_train <= 0 or n_train >= len(pool):
        train_idxs = pool
    else:
        rng = np.random.default_rng(seed)
        train_idxs = sorted(rng.choice(len(pool), n_train, replace=False).tolist())
        train_idxs = [pool[i] for i in train_idxs]

    train_imgs = images[train_idxs]
    return train_imgs, test_img, train_imgs.shape[0]
