"""
Metrics used in "Underwater image enhancement via multiscale disentanglement
strategy" (MD-Net, Sci. Rep. 2025): PSNR, SSIM (full-reference) and
UCIQE, UIQM (no-reference).

Usage:
    python uie_metrics.py --pred results/ --gt reference/        # PSNR/SSIM/UCIQE/UIQM
    python uie_metrics.py --pred results/                        # UCIQE/UIQM only

Requirements: numpy, opencv-python, scikit-image
Note: all functions expect uint8 RGB images (H, W, 3) in [0, 255].
"""
import argparse
import math
import os

import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def tensor_to_uint8(t):
    """torch tensor (C,H,W) or (1,C,H,W) in ~[0,1] -> uint8 RGB (H,W,3).
    CLAMPING is essential: without it, values >1 overflow when cast to uint8
    and produce bright/red/colored speckles."""
    t = t.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return np.round(t * 255.0).astype(np.uint8)


def load_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


# --------------------------------------------------------------------------
# Full-reference metrics
# --------------------------------------------------------------------------
def psnr(pred, gt):
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(pred, gt):
    return structural_similarity(pred, gt, channel_axis=2, data_range=255)


# --------------------------------------------------------------------------
# UCIQE  (Yang & Sowmya, 2015)
#   UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
#   sigma_c: std of chroma, con_l: luminance contrast (top1% - bottom1%),
#   mu_s: mean saturation, in CIELab.  c1=0.4680, c2=0.2745, c3=0.2576
# --------------------------------------------------------------------------
def uciqe(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l = lab[..., 0] / 255.0
    a = lab[..., 1] / 255.0
    b = lab[..., 2] / 255.0

    chroma = np.sqrt(a ** 2 + b ** 2)
    sat = chroma / np.sqrt(chroma ** 2 + l ** 2 + 1e-12)

    mean_chroma = np.mean(chroma)
    sigma_c = np.sqrt(np.mean(np.abs(1.0 - (mean_chroma / (chroma + 1e-12)) ** 2)))

    l_sorted = np.sort(l.flatten())
    n = len(l_sorted)
    con_l = l_sorted[min(int(n * 0.99), n - 1)] - l_sorted[int(n * 0.01)]

    mu_s = np.mean(sat)
    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


# --------------------------------------------------------------------------
# UIQM  (Panetta et al., 2016)
#   UIQM = 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM
# --------------------------------------------------------------------------
def _mu_a(x, alpha_l=0.1, alpha_r=0.1):
    """Asymmetric alpha-trimmed mean."""
    x = np.sort(x.flatten())
    k = len(x)
    t_l = math.ceil(alpha_l * k)
    t_r = math.floor(alpha_r * k)
    return np.sum(x[t_l + 1: k - t_r]) / (k - t_l - t_r)


def _uicm(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = (r + g) / 2.0 - b
    mu_rg, mu_yb = _mu_a(rg), _mu_a(yb)
    var_rg = np.mean((rg - mu_rg) ** 2)
    var_yb = np.mean((yb - mu_yb) ** 2)
    left = math.sqrt(mu_rg ** 2 + mu_yb ** 2)
    right = math.sqrt(var_rg + var_yb)
    return -0.0268 * left + 0.1586 * right


def _sobel_mag(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    m = np.max(mag)
    return mag * (255.0 / m) if m > 0 else mag


def _blocks(x, win):
    h, w = x.shape
    h2, w2 = (h // win) * win, (w // win) * win
    x = x[:h2, :w2]
    return x.reshape(h2 // win, win, w2 // win, win).transpose(0, 2, 1, 3).reshape(-1, win, win)


def _eme(x, win=8):
    blk = _blocks(x, win)
    mx = blk.max(axis=(1, 2))
    mn = blk.min(axis=(1, 2))
    valid = (mx > 0) & (mn > 0)
    vals = np.zeros_like(mx)
    vals[valid] = np.log(mx[valid] / mn[valid])
    return 2.0 * vals.sum() / len(mx)


def _uism(img):
    weights = (0.299, 0.587, 0.114)
    total = 0.0
    for c, lam in enumerate(weights):
        ch = img[..., c]
        edge = _sobel_mag(ch) * ch
        total += lam * _eme(edge, 8)
    return total


def _uiconm(gray, win=10):
    blk = _blocks(gray, win)
    mx = blk.max(axis=(1, 2))
    mn = blk.min(axis=(1, 2))
    top, bot = mx - mn, mx + mn
    valid = (top > 0) & (bot > 0)
    vals = np.zeros_like(mx)
    ratio = top[valid] / bot[valid]
    vals[valid] = ratio * np.log(ratio)
    return -vals.sum() / len(mx)


def uiqm(img_rgb):
    img = img_rgb.astype(np.float64)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(gray, 10)


# --------------------------------------------------------------------------
# Folder evaluation
# --------------------------------------------------------------------------
def evaluate_folder(pred_dir, gt_dir=None, size=None):
    names = sorted(f for f in os.listdir(pred_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    rows = []
    for n in names:
        pred = load_rgb(os.path.join(pred_dir, n), size)
        row = {"name": n, "UCIQE": uciqe(pred), "UIQM": uiqm(pred)}
        if gt_dir:
            gt_path = os.path.join(gt_dir, n)
            if not os.path.exists(gt_path):
                continue
            gt = load_rgb(gt_path, size)
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_CUBIC)
            row["PSNR"] = psnr(pred, gt)
            row["SSIM"] = ssim(pred, gt)
        rows.append(row)

    if not rows:
        print("No images evaluated (check folders / file names).")
        return rows
    keys = [k for k in rows[0] if k != "name"]
    print(f"Evaluated {len(rows)} images")
    for k in keys:
        print(f"  {k:6s}: {np.mean([r[k] for r in rows]):.4f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="folder with enhanced images")
    ap.add_argument("--gt", default=None, help="folder with reference images (same file names)")
    ap.add_argument("--size", type=int, default=None, help="resize both to size x size (paper: 350)")
    args = ap.parse_args()
    evaluate_folder(args.pred, args.gt, args.size)
