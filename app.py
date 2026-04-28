"""
Matrix Analysis Lab - Desktop Application
==========================================
Uses the rmt-denoise library (pip install rmt-denoise) for denoising.
Web UI served via pywebview from the /web folder.

Usage:
    python app.py
"""
import json
import os
import io
import sys
import base64
import atexit
import shutil
import multiprocessing
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.optimize import minimize_scalar
from PIL import Image
import webview
import logging

# Import the rmt-denoise library
from rmt_denoise import (
    MPLawDenoiser,
    GeneralizedCovDenoiser,
    add_gaussian_noise,
    add_laplacian_noise,
    add_mixture_gaussian_noise,
    add_structured_noise,
    compute_psnr,
    compute_ssim,
)
from rmt_denoise.core import (
    g_function,
    compute_G_plus,
    compute_G_minus,
    compute_support_bounds,
    compute_discriminant,
    compute_P4_coefficients,
)

# --------- Paths --------- #
def get_random_matrix_folder():
    folder = Path.home() / ".random_matrix"
    folder.mkdir(exist_ok=True)
    return folder

def get_temp_folder():
    folder = get_random_matrix_folder() / "temp"
    folder.mkdir(exist_ok=True)
    return folder

def cleanup_temp_folder():
    pass  # Don't auto-delete — user manages via UI


# --------- Math helpers (cubic roots for plots) --------- #
def solve_cubic(a, b, c, d):
    """Solve ax^3 + bx^2 + cx + d = 0."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return np.array([complex(np.nan)] * 3)
        roots = np.roots([b, c, d])
        return np.concatenate([roots, [complex(np.inf)]])
    return np.roots([a, b, c, d])


def compute_ImS_vs_Z(a, y, beta, num_points, z_min, z_max, progress_cb=None):
    z_values = np.linspace(z_min, z_max, num_points)
    ims = np.zeros((num_points, 3))
    reals = np.zeros((num_points, 3))
    for i, z in enumerate(z_values):
        roots = solve_cubic(z * a, z * (a + 1) + a * (1 - y),
                            z + (a + 1) - y - y * beta * (a - 1), 1.0)
        for j in range(min(3, len(roots))):
            ims[i, j] = roots[j].imag
            reals[i, j] = roots[j].real
        if progress_cb:
            progress_cb(i + 1, num_points)
    return {"z_values": z_values.tolist(), "ims_values": ims.tolist(), "real_values": reals.tolist()}


def track_roots_consistently(grid, all_roots):
    tracked = [list(all_roots[0])]
    for i in range(1, len(all_roots)):
        prev = tracked[-1]
        curr = list(all_roots[i])
        assigned = [False] * len(curr)
        new_row = [None] * len(prev)
        for pi in range(len(prev)):
            best_j, best_d = 0, float('inf')
            for j in range(len(curr)):
                if not assigned[j]:
                    d = abs(prev[pi] - curr[j])
                    if d < best_d:
                        best_d, best_j = d, j
            new_row[pi] = curr[best_j]
            assigned[best_j] = True
        tracked.append(new_row)
    return tracked


def generate_discriminant(z, beta, a, y):
    ca = z * a
    cb = z * (a + 1) + a * (1 - y)
    cc = z + (a + 1) - y - y * beta * (a - 1)
    cd = 1.0
    return 18*ca*cb*cc*cd - 27*ca**2*cd**2 + cb**2*cc**2 - 4*cb**3*cd - 4*ca*cc**3


# --------- Image helpers --------- #
def img_to_base64(arr):
    """float [0,1] array -> base64 PNG string"""
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def images_to_matrix(images):
    n, H, W = images.shape
    return images.reshape(n, H * W).T  # (p, n)


def matrix_to_images(X, H, W):
    return np.clip(X.T.reshape(-1, H, W), 0, 1)


# --------- Bridge (Python API exposed to web UI) --------- #
class Bridge:
    def __init__(self, window):
        self.window = window
        self.processed_results = []

    def _update_progress_ui(self, bar_id, status_id, step, total):
        if self.window:
            pct = int(step / total * 100)
            try:
                self.window.evaluate_js(
                    f'document.getElementById("{bar_id}").style.width="{pct}%";')
            except:
                pass

    def _update_status_ui(self, elem_id, text):
        if self.window:
            safe = text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '<br>')
            try:
                self.window.evaluate_js(
                    f"document.getElementById('{elem_id}').innerHTML='{safe}';")
            except:
                pass

    # ---- Math analysis methods (same as original) ----
    def im_vs_z(self, params):
        a = float(params.get("a", 2.0))
        y = float(params.get("y", 1.0))
        beta = float(params.get("beta", 0.5))
        z_min = float(params.get("z_min", 0.01))
        z_max = float(params.get("z_max", 10))
        pts = int(params.get("points", 400))
        data = compute_ImS_vs_Z(a, y, beta, pts, z_min, z_max)
        return {"data": data, "params": {"a": a, "y": y, "beta": beta}}

    def roots_vs_beta(self, params):
        z = float(params.get("z", 1.0))
        y = float(params.get("y", 1.0))
        a = float(params.get("a", 2.0))
        b_min = float(params.get("beta_min", 0.0))
        b_max = float(params.get("beta_max", 1.0))
        pts = int(params.get("points", 400))

        betas = np.linspace(b_min, b_max, pts)
        all_roots = []
        discs = []
        for beta in betas:
            roots = solve_cubic(z * a, z * (a + 1) + a * (1 - y),
                                z + (a + 1) - y - y * beta * (a - 1), 1.0)
            all_roots.append(roots)
            discs.append(generate_discriminant(z, beta, a, y))

        tracked = track_roots_consistently(betas, all_roots)
        ims = [[r[j].imag for r in tracked] for j in range(3)]
        reals = [[r[j].real for r in tracked] for j in range(3)]

        return {
            "data": {
                "beta_values": betas.tolist(),
                "ims_values": ims,
                "real_values": reals,
                "discriminants": discs,
            },
            "params": {"z": z, "y": y, "a": a}
        }

    def eigen_distribution(self, params):
        beta = float(params.get("beta", 0.5))
        a = float(params.get("a", 2.0))
        n = int(params.get("n", 400))
        p = int(params.get("p", 200))
        seed = int(params.get("seed", 42))

        np.random.seed(seed)
        X = np.random.randn(p, n) / np.sqrt(n)
        num_spike = max(1, int(beta * p))
        T_diag = np.ones(p)
        T_diag[:num_spike] = a
        T = np.diag(T_diag)
        B = T @ (X @ X.T)
        eigenvalues = np.linalg.eigvalsh(B)
        eigenvalues = np.sort(eigenvalues)

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(eigenvalues, bw_method=0.05)
        x = np.linspace(max(0, eigenvalues.min() - 0.5), eigenvalues.max() + 0.5, 500)
        density = kde(x)

        return {
            "data": {
                "eigenvalues": eigenvalues.tolist(),
                "kde_x": x.tolist(),
                "kde_y": density.tolist(),
            },
            "stats": {
                "count": len(eigenvalues),
                "min": float(eigenvalues.min()),
                "max": float(eigenvalues.max()),
                "mean": float(eigenvalues.mean()),
                "std": float(eigenvalues.std()),
                "y": p / n,
            }
        }

    # ---- Folder management (same as original) ----
    def list_random_matrix_folders(self):
        base = get_random_matrix_folder()
        folders = [f.name for f in base.iterdir()
                   if f.is_dir() and f.name != "temp"]
        return {"folders": sorted(folders)}

    def get_folder_contents(self, folder_name):
        folder = get_random_matrix_folder() / folder_name
        files = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                with open(f, 'rb') as fh:
                    files.append({"name": f.name, "data": base64.b64encode(fh.read()).decode()})
        return {"files": files}

    def create_folder(self, params):
        name = params.get("folder_name", "").strip()
        if not name:
            return {"success": False, "error": "Empty name"}
        (get_random_matrix_folder() / name).mkdir(exist_ok=True)
        return {"success": True}

    def delete_folder(self, params):
        name = params.get("folder_name", "")
        if name == "temp":
            return {"success": False, "error": "Cannot delete temp"}
        path = get_random_matrix_folder() / name
        if path.exists():
            shutil.rmtree(path)
        return {"success": True}

    def upload_files_to_folder(self, params):
        folder = params.get("folder", "")
        files = params.get("files", [])
        folder_path = get_random_matrix_folder() / folder
        folder_path.mkdir(exist_ok=True)
        count = 0
        for fd in files:
            img_bytes = base64.b64decode(fd["data"])
            img = Image.open(io.BytesIO(img_bytes))
            img.save(folder_path / fd["name"])
            count += 1
        return {"success": True, "count": count}

    def import_local_folder(self, params):
        files = params.get("files", [])
        folder = params.get("folder", "imported")
        folder_path = get_random_matrix_folder() / folder
        folder_path.mkdir(exist_ok=True)
        count = 0
        for fd in files:
            img_bytes = base64.b64decode(fd["data"])
            img = Image.open(io.BytesIO(img_bytes))
            img.save(folder_path / fd["name"])
            count += 1
        return {"success": True, "count": count, "folder": folder}

    # ---- Image eigenvalue analysis ----
    def compute_image_eigenvalues(self, params):
        try:
            image_data = params.get("image_data", "")
            if not image_data:
                return {"success": False, "error": "No image data"}
            img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
            H, W = arr.shape
            if H < W:
                C = arr @ arr.T / W
            else:
                C = arr.T @ arr / H
            eigs = np.linalg.eigvalsh(C)
            eigs = np.sort(eigs[eigs > 1e-10])

            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(bandwidth=max(0.01, np.std(eigs) * 0.3))
            kde.fit(eigs.reshape(-1, 1))
            x = np.linspace(max(0, eigs.min() - np.std(eigs)), eigs.max() + np.std(eigs), 200)
            y = np.exp(kde.score_samples(x.reshape(-1, 1)))

            return {
                "success": True,
                "eigenvalues": eigs.tolist(),
                "kde_x": x.tolist(),
                "kde_y": y.tolist(),
                "stats": {
                    "count": len(eigs),
                    "min": float(eigs.min()), "max": float(eigs.max()),
                    "mean": float(eigs.mean()), "std": float(eigs.std()),
                },
                "image_size": {"height": H, "width": W}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ---- Main denoising pipeline (uses rmt-denoise library) ----
    def _apply_method(self, method, noisy_images, params, prefix):
        """Apply denoising using the rmt-denoise library."""
        n, H, W = noisy_images.shape

        if method == "mp_lower_bulk":
            mp = MPLawDenoiser()
            denoised = mp.denoise(noisy_images)
            info = mp.info
            info['method_label'] = 'M-P Law'
        elif method == "gen_lower_bulk":
            gc = GeneralizedCovDenoiser()
            denoised = gc.denoise(noisy_images)
            info = gc.info
            info['method_label'] = 'Gen. Cov (Auto)'
        elif method == "gen_manual":
            sigma2 = float(params.get(f"{prefix}_sigma2", 0.01))
            a = float(params.get(f"{prefix}_a", 2.0))
            beta = float(params.get(f"{prefix}_beta", 0.5))
            gc = GeneralizedCovDenoiser(sigma2=sigma2, a=a, beta=beta)
            denoised = gc.denoise(noisy_images)
            info = gc.info
            info['method_label'] = f'Gen. Cov (a={a}, β={beta})'
        else:
            mp = MPLawDenoiser()
            denoised = mp.denoise(noisy_images)
            info = mp.info
            info['method_label'] = 'M-P Law'

        return denoised, info

    def _format_method_details(self, method, info, method_name):
        rank = info.get('rank', info.get('num_signal', 0))
        s = f"<strong>{method_name}</strong><br>"
        s += f'<strong style="color:#2563eb;">Eigenvectors used: {rank}</strong><br>'
        if method.startswith("mp"):
            s += f"σ²={info.get('sigma2', 0):.6f}, λ₊={info.get('threshold', 0):.4f}"
        else:
            s += f"σ²={info.get('sigma2', 0):.6f}, a={info.get('a', '?')}, β={info.get('beta', '?')}"
        return s

    def process_images(self, params):
        """Main pipeline: load → add noise → denoise with 2 methods → compare."""
        try:
            folder_name = params.get("folder")
            method1 = params.get("method1", "mp_lower_bulk")
            method2 = params.get("method2", "gen_lower_bulk")
            noise_type = params.get("noise_type", "laplacian")
            laplacian_scale = float(params.get("laplacian_scale", 0.1))
            random_seed = params.get("random_seed", None)
            if random_seed is not None:
                random_seed = int(random_seed)

            # How many images to use (0 or empty = all)
            num_images = params.get("num_images", 0)
            if num_images is not None and num_images != "" and num_images != "0":
                num_images = int(num_images)
            else:
                num_images = 0  # means use all

            # Step 1: Load images
            self._update_status_ui('status_process', 'Step 1/5: Loading images...')
            self._update_progress_ui('bar_process', 'status_process', 1, 5)

            folder_path = get_random_matrix_folder() / folder_name
            all_files = sorted([f for f in folder_path.iterdir()
                                if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if not all_files:
                return {"success": False, "error": "No images in folder"}

            # Select subset: randomly pick num_images from the folder
            if num_images > 0 and num_images < len(all_files):
                rng = np.random.RandomState(random_seed if random_seed else 42)
                chosen_idx = rng.choice(len(all_files), size=num_images, replace=False)
                chosen_idx.sort()
                files = [all_files[i] for i in chosen_idx]
                selection_note = f" (randomly selected {num_images} of {len(all_files)})"
            else:
                files = all_files
                selection_note = ""

            images = np.stack([
                np.array(Image.open(f).convert('L'), dtype=np.float32) / 255.0
                for f in files
            ])
            n, H, W = images.shape
            p = H * W
            y = p / float(n)

            self._update_status_ui('processing_info',
                f'<strong>Using {n} images{selection_note}</strong> ({H}×{W}, p={p}, y={y:.4f})')

            # Step 2: Add noise
            self._update_status_ui('status_process', f'Step 2/5: Adding {noise_type} noise...')
            self._update_progress_ui('bar_process', 'status_process', 2, 5)

            if random_seed is not None:
                np.random.seed(random_seed)

            if noise_type == "laplacian":
                noisy_images, _ = add_laplacian_noise(images, laplacian_scale)
                noise_label = f"Laplacian (σ={laplacian_scale})"
            elif noise_type == "mixture_gaussian":
                weights = params.get("mog_weights", [0.6, 0.3, 0.1])
                means = params.get("mog_means", [0.0, 0.0, 0.0])
                sigmas = params.get("mog_sigmas", [0.1, 0.05, 0.3])
                # Use our library's mixture noise (2-component)
                noisy_images, _ = add_mixture_gaussian_noise(
                    images, sigma1=float(sigmas[0]), sigma2=float(sigmas[2]),
                    beta=float(weights[2]))
                noise_label = "Mixture of Gaussians"
            elif noise_type == "gaussian":
                sigma = float(params.get("gaussian_sigma", 0.1))
                noisy_images, _ = add_gaussian_noise(images, sigma)
                noise_label = f"Gaussian (σ={sigma})"
            elif noise_type == "structured":
                a_noise = float(params.get("struct_a", 5.0))
                beta_noise = float(params.get("struct_beta", 0.15))
                sigma_noise = float(params.get("struct_sigma", 0.1))
                noisy_images, _ = add_structured_noise(images, a=a_noise, beta=beta_noise, sigma=sigma_noise)
                noise_label = f"Structured (a={a_noise}, β={beta_noise})"
            else:
                noisy_images, _ = add_laplacian_noise(images, laplacian_scale)
                noise_label = f"Laplacian (σ={laplacian_scale})"

            # Step 3: Method 1
            method_names = {
                "mp_lower_bulk": "M-P Law",
                "gen_lower_bulk": "Gen. Cov (Auto)",
                "gen_manual": "Gen. Cov (Manual)",
            }
            m1_name = method_names.get(method1, method1)
            self._update_status_ui('status_process', f'Step 3/5: {m1_name}...')
            self._update_progress_ui('bar_process', 'status_process', 3, 5)

            den1, info1 = self._apply_method(method1, noisy_images, params, "method1")
            self._update_status_ui('method1_info', self._format_method_details(method1, info1, m1_name))

            # Step 4: Method 2
            m2_name = method_names.get(method2, method2)
            self._update_status_ui('status_process', f'Step 4/5: {m2_name}...')
            self._update_progress_ui('bar_process', 'status_process', 4, 5)

            den2, info2 = self._apply_method(method2, noisy_images, params, "method2")
            self._update_status_ui('method2_info', self._format_method_details(method2, info2, m2_name))

            # Step 5: Save results
            self._update_status_ui('status_process', 'Step 5/5: Saving results...')
            self._update_progress_ui('bar_process', 'status_process', 5, 5)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp = get_temp_folder()
            noisy_dir = temp / f"noisy_{noise_type}_{ts}"
            m1_dir = temp / f"{method1}_{ts}"
            m2_dir = temp / f"{method2}_{ts}"
            noisy_dir.mkdir(exist_ok=True)
            m1_dir.mkdir(exist_ok=True)
            m2_dir.mkdir(exist_ok=True)

            results = []
            for i in range(n):
                # Save to disk
                Image.fromarray(np.clip(noisy_images[i] * 255, 0, 255).astype(np.uint8), 'L').save(noisy_dir / f"{i:03d}.png")
                Image.fromarray(np.clip(den1[i] * 255, 0, 255).astype(np.uint8), 'L').save(m1_dir / f"{i:03d}.png")
                Image.fromarray(np.clip(den2[i] * 255, 0, 255).astype(np.uint8), 'L').save(m2_dir / f"{i:03d}.png")

                results.append({
                    "original": img_to_base64(images[i]),
                    "blurred": img_to_base64(noisy_images[i]),
                    "method1": img_to_base64(den1[i]),
                    "method2": img_to_base64(den2[i]),
                    "method1_name": m1_name,
                    "method2_name": m2_name,
                    "method1_eigvec": info1.get('rank', info1.get('num_signal', 0)),
                    "method2_eigvec": info2.get('rank', info2.get('num_signal', 0)),
                    "noise_type_label": noise_label,
                })

            self.processed_results = results

            return {
                "success": True,
                "total_images": len(results),
                "results": results,
                "processing_info": {
                    "n": n, "H": H, "W": W, "p": p, "y": y,
                    "method1": self._format_method_details(method1, info1, m1_name),
                    "method2": self._format_method_details(method2, info2, m2_name),
                }
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def get_processed_image(self, index):
        if hasattr(self, 'processed_results') and index < len(self.processed_results):
            return self.processed_results[index]
        return {"error": "Not found"}

    # ---- Temp file management ----
    def list_temp_folders(self):
        temp = get_temp_folder()
        if not temp.exists():
            return {"folders": []}
        folders = []
        for f in sorted(temp.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_dir():
                size = sum(p.stat().st_size for p in f.rglob("*") if p.is_file())
                folders.append({
                    "name": f.name,
                    "size_mb": round(size / 1048576, 2),
                    "file_count": sum(1 for _ in f.iterdir()),
                    "modified_str": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
        return {"folders": folders}

    def delete_temp_folder(self, folder_name):
        path = get_temp_folder() / folder_name
        if path.exists():
            shutil.rmtree(path)
        return {"success": True}

    def delete_all_temp_folders(self):
        temp = get_temp_folder()
        for f in temp.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
        return {"success": True}

    def open_temp_folder(self, folder_name):
        path = get_temp_folder() / folder_name
        if path.exists():
            os.startfile(str(path))
        return {"success": True}

    def download_images_playwright(self, params):
        """Download images using playwright."""
        try:
            from playwright.sync_api import sync_playwright
            import time, random

            url = params.get("url")
            count = int(params.get("count", 10))
            scale = int(params.get("scale", 100))
            folder = params.get("folder", "downloaded")

            folder_path = get_random_matrix_folder() / folder
            folder_path.mkdir(exist_ok=True)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_context().new_page()
                page.goto(url, timeout=60000)
                time.sleep(random.uniform(2, 4))

                for i in range(count):
                    ss = page.screenshot()
                    img = Image.open(io.BytesIO(ss)).convert('L').resize((scale, scale), Image.BICUBIC)
                    img.save(folder_path / f"{i:03d}.png")
                    if i < count - 1:
                        time.sleep(random.uniform(1, 2))
                browser.close()

            return {"success": True, "count": count, "folder": folder}
        except ImportError:
            return {"success": False, "error": "Install playwright: pip install playwright && playwright install chromium"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# --------- Main --------- #
def main():
    import warnings
    logging.getLogger('pywebview').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')

    # Redirect stderr to suppress Windows accessibility errors
    if sys.platform == 'win32':
        try:
            sys.stderr = open(os.devnull, 'w')
        except Exception:
            pass

    try:
        get_random_matrix_folder()
    except Exception as e:
        print(f"Warning: Could not create .random_matrix folder: {e}")

    atexit.register(cleanup_temp_folder)

    api = Bridge(None)

    # Load HTML from web/ folder — read and inline everything
    web_dir = Path(__file__).parent / "web"
    html_path = web_dir / "index.html"
    css_path = web_dir / "styles.css"
    js_path = web_dir / "app.js"

    if html_path.exists():
        html = html_path.read_text(encoding='utf-8')

        # Inline CSS
        if css_path.exists():
            css = css_path.read_text(encoding='utf-8')
            html = html.replace(
                '<link rel="stylesheet" href="styles.css">',
                f'<style>\n{css}\n</style>'
            )

        # Inline JS
        if js_path.exists():
            js = js_path.read_text(encoding='utf-8')
            html = html.replace(
                '<script src="app.js"></script>',
                f'<script>\n{js}\n</script>'
            )
    else:
        html = "<html><body><h1>Error: web/ folder not found</h1><p>Make sure the web/ folder is next to app.py</p></body></html>"

    # Create window — this is the same pattern as the original working app.py
    window = webview.create_window(
        "Matrix Analysis Lab (rmt-denoise)",
        html=html,
        js_api=api,
        width=1300,
        height=900,
    )

    api.window = window

    # Start the webview - this blocks until the window is closed
    webview.start()


if __name__ == "__main__":
    # CRITICAL: Must be called before any other code for Windows exe
    multiprocessing.freeze_support()
    main()
